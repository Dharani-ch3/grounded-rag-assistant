"""Decide whether a question goes to the knowledge graph, to document search, or both.

Relationship questions ("who owns…", "who approved…", "which documents can nurses read?",
"who do I report to?") go to the graph. "Who should I contact about X?" uses both: search
finds the document about X, and the graph finds its owner. Everything else is document search.

The rules below are deterministic and fast. With an LLM provider configured, the model may
pick the intent instead, but only from the same fixed menu, and its choice is validated
before use; anything unexpected falls back to these rules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .knowledge import KnowledgeBase, User
from .text import tokenize

GRAPH_INTENTS = (
    "doc_owner",
    "doc_approver",
    "topic_contact",
    "role_documents",
    "user_manager",
    "org_members",
    "user_documents",
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("doc_approver", re.compile(r"\bwho (approved|signed off( on)?)\b", re.I)),
    (
        "topic_contact",
        re.compile(
            r"\bwho (should|do|can) i (ask|contact|talk to|call|email|reach out to)\b|\bwho to (ask|contact)\b", re.I
        ),
    ),
    (
        "user_manager",
        re.compile(
            r"\bwho (do|does) (\w+) report to\b|\bwho is (my|\w+'s) (manager|boss|supervisor)\b|\b(manager|boss) of \w+",
            re.I,
        ),
    ),
    (
        "user_documents",
        re.compile(r"\b(which|what) (documents|docs|policies|procedures)\b.*\b(does|do) (\w+) own\b", re.I),
    ),
    ("doc_owner", re.compile(r"\bwho (owns|maintains|wrote|is responsible for)\b|\bowner of\b", re.I)),
    (
        "role_documents",
        re.compile(
            r"\b(which|what) (documents|docs|policies|procedures)\b.*\b(can|may|do)\b.*\b(read|see|access|view)\b", re.I
        ),
    ),
    ("org_members", re.compile(r"\bwho (works|is|are) (in|on)\b|\bmembers of\b", re.I)),
]

# Words that appear in many titles and say nothing about which document is meant.
_GENERIC = {"policy", "procedure", "guide", "guideline", "runbook", "document", "sop", "monitor"}


@dataclass
class Route:
    kind: str  # documents | graph | both
    intent: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    by: str = "rules"


def _find_user(question: str, asker: User, kb: KnowledgeBase) -> tuple[str | None, bool]:
    q = question.lower()
    if re.search(r"\b(i|my|me)\b", q):
        return asker.id, True
    for u in kb.directory.users.values():
        first = u.name.split()[0].lower()
        if u.name.lower() in q or re.search(rf"\b{re.escape(first)}('s)?\b", q):
            return u.id, u.id == asker.id
    return None, False


def _find_alias(question: str, items: dict[str, dict[str, Any]]) -> str | None:
    q = f" {question.lower()} "
    best: tuple[int, str] | None = None
    for item_id, item in items.items():
        for alias in [item["name"].lower(), *item.get("aliases", [])]:
            if re.search(rf"\b{re.escape(alias)}\b", q) and (best is None or len(alias) > best[0]):
                best = (len(alias), item_id)
    return best[1] if best else None


def find_document(question: str, kb: KnowledgeBase, candidates: list[str]) -> str | None:
    """Match a document by its title. Candidates are the documents the asker may use."""
    q = set(tokenize(question))
    best: tuple[float, int, str] | None = None
    for doc_id in candidates:
        doc = kb.docs[doc_id]
        title = set(tokenize(doc.title))
        specific = title - _GENERIC
        hits = q & specific
        if not hits:
            continue
        score = len(q & title) / len(title)
        key = (score, doc.version, doc_id)
        if best is None or key > best:
            best = key
    return best[2] if best and best[0] >= 0.5 else None


def route(question: str, asker: User, kb: KnowledgeBase, candidates: list[str]) -> Route:
    for intent, pattern in _PATTERNS:
        if not pattern.search(question):
            continue
        if intent in {"doc_owner", "doc_approver"}:
            doc_id = find_document(question, kb, candidates)
            if doc_id:
                return Route("graph", intent, {"doc_id": doc_id}, f"Asks who {intent.split('_')[1]}s a named document")
            # No title match: search finds the document, the graph answers the same question.
            return Route(
                "both", intent, {"topic": question}, "Asks about a document's people; search finds the document"
            )
        if intent == "topic_contact":
            return Route(
                "both",
                "topic_contact",
                {"topic": question},
                "Asks who to contact; search finds the document, the graph finds its owner",
            )
        if intent in {"user_manager", "user_documents"}:
            user_id, is_self = _find_user(question, asker, kb)
            if user_id:
                return Route(
                    "graph", intent, {"user_id": user_id, "is_self": is_self}, "Asks about a person's relationships"
                )
        if intent == "role_documents":
            role = _find_alias(question, kb.directory.roles)
            if role:
                return Route("graph", intent, {"role": role}, "Asks which documents a role can read")
        if intent == "org_members":
            org = _find_alias(question, kb.directory.orgs)
            if org:
                return Route("graph", intent, {"org_id": org}, "Asks who belongs to a department")
    return Route("documents", None, {}, "Asks what the documents say")


ROUTER_PROMPT = """You route questions for an enterprise knowledge assistant.
Pick ONE intent:
- documents: the answer is in the text of a policy or procedure
- doc_owner / doc_approver: who owns / approved a named document (give "document": its title words)
- topic_contact: who to contact about a topic
- role_documents: which documents a role can read (give "role": one of {roles})
- user_manager: who someone reports to (give "user": one of {users}, or "me")
- org_members: who works in a department (give "org": one of {orgs})
- user_documents: which documents a person owns (give "user")
Reply with JSON only: {{"intent": "...", "document": "", "role": "", "user": "", "org": ""}}"""


def route_with_llm(provider: Any, question: str, asker: User, kb: KnowledgeBase, candidates: list[str]) -> Route:
    """Let a model choose from the fixed menu; validate everything it returns."""
    fallback = route(question, asker, kb, candidates)
    try:
        raw = provider.complete(
            ROUTER_PROMPT.format(
                roles=", ".join(kb.directory.roles),
                users=", ".join(kb.directory.users),
                orgs=", ".join(kb.directory.orgs),
            ),
            question,
            json_mode=True,
        )
        choice = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except Exception:  # any provider or parsing failure: use the rules
        return fallback
    intent = choice.get("intent")
    if intent == "documents":
        return Route("documents", None, {}, "Model: answer is in document text", "model")
    if intent in {"doc_owner", "doc_approver"}:
        doc_id = find_document(str(choice.get("document", "")) or question, kb, candidates)
        if doc_id:
            return Route("graph", intent, {"doc_id": doc_id}, "Model: asks about a named document", "model")
        return Route("both", intent, {"topic": question}, "Model: asks about a document's people", "model")
    if intent == "topic_contact":
        return Route("both", "topic_contact", {"topic": question}, "Model: asks who to contact", "model")
    if intent == "role_documents" and choice.get("role") in kb.directory.roles:
        return Route("graph", intent, {"role": choice["role"]}, "Model: documents for a role", "model")
    if intent in {"user_manager", "user_documents"}:
        user = choice.get("user")
        user = asker.id if user == "me" else user
        if user in kb.directory.users:
            return Route(
                "graph", intent, {"user_id": user, "is_self": user == asker.id}, "Model: about a person", "model"
            )
    if intent == "org_members" and choice.get("org") in kb.directory.orgs:
        return Route("graph", intent, {"org_id": choice["org"]}, "Model: members of a department", "model")
    return fallback
