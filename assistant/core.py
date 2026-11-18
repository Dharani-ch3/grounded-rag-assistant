"""The assistant: route a question, retrieve what the asker may use, answer, and verify.

    question ──► router ──► graph templates ─────────────┐
                   │                                     ├─► answer + citations + trace
                   └──► permission-aware hybrid search ──┤
                          └─► draft answer ─► grounding check (drop unsupported, or abstain)

`mode="naive"` turns every guardrail off (no access filter, no lifecycle filter, no
abstention, no grounding check). It exists only so the demo and the evaluation can show
what those guardrails prevent.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .gateway import LLMProvider, get_embedder, get_provider
from .graph import MemoryGraph, explain
from .grounding import Source, extractive_answer, verify
from .knowledge import Document, KnowledgeBase, KnowledgeError, User
from .retrieval import AccessFilter, HybridRetriever, MemoryVectorStore, VectorStore
from .router import Route, route, route_with_llm
from .text import content_terms, tokenize

MODES = ("grounded", "naive")

ANSWER_SYSTEM = """You answer employee questions for Ridgeway Health using ONLY the numbered sources.
Rules:
- After every sentence, cite the source it came from in square brackets, like [2].
- Copy numbers, amounts, dates and time limits exactly as the source writes them.
- Do not add advice, background or anything the sources don't say.
- If the sources don't answer the question, reply with exactly: NOT_FOUND
- Answer in 1 to 3 short sentences."""

_CONTACT_WORDS = set(
    tokenize(
        "who should do can i ask contact talk call email reach out about with problems problem issues issue questions "
        "approved approve signed sign off owns own owner maintains responsible wrote"
    )
)


# Greetings, thanks and "what can you do?" aren't policy questions. Answering them with
# "I can't answer that" is unfriendly, and searching for them returns random passages.
_GREETING = re.compile(
    r"^(hi|hello|hey|hiya|howdy|yo|good (morning|afternoon|evening)|greetings)\b[\s!.,]*(there|team|all)?[\s!.]*$", re.I
)
_THANKS = re.compile(r"^(thanks|thank you|thx|ty|ok|okay|cool|great|got it|bye|goodbye)\b[\s!.,a-z]*$", re.I)
_HELP = re.compile(r"^(help|what can you do|what do you (do|know)|how does this work|who are you)\b[\s?!.]*$", re.I)


class NotFound(LookupError):
    pass


class Forbidden(PermissionError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "document"


class Assistant:
    def __init__(
        self,
        settings: Settings | None = None,
        kb: KnowledgeBase | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        root = Path(self.settings.knowledge_path)
        self.kb = kb or KnowledgeBase.load(root)
        self.graph = MemoryGraph(self.kb)
        self.retriever = HybridRetriever(self.kb, get_embedder(self.settings), vector_store or self._vector_store())
        suggestions = root / "suggestions.json"
        self.suggestions: dict[str, list[dict[str, str]]] = (
            json.loads(suggestions.read_text()) if suggestions.exists() else {}
        )
        self.audit: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _vector_store(self) -> VectorStore:
        if self.settings.vector_store == "pinecone":
            from .backends.pinecone import PineconeStore

            return PineconeStore.from_env()
        if self.settings.vector_store == "weaviate":
            from .backends.weaviate import WeaviateStore

            return WeaviateStore.from_env()
        return MemoryVectorStore()

    # ---------- helpers ----------

    def user(self, user_id: str) -> User:
        try:
            return self.kb.directory.user(user_id)
        except KnowledgeError as exc:
            raise NotFound(str(exc)) from exc

    def _log(self, actor: str, action: str, **detail: Any) -> None:
        event = {"at": _now(), "actor": actor, "action": action, **detail}
        with self._lock:
            self.audit.append(event)
            if self.settings.audit_path:
                Path(self.settings.audit_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.settings.audit_path, "a") as f:
                    f.write(json.dumps(event) + "\n")

    def _person(self, user_id: str | None) -> dict[str, str] | None:
        if not user_id or user_id not in self.kb.directory.users:
            return None
        u = self.kb.directory.users[user_id]
        return {"id": u.id, "name": u.name, "title": u.title, "org": self.kb.directory.org_name(u.org)}

    def _doc_info(self, doc: Document, viewer: User) -> dict[str, Any]:
        info = doc.summary()
        info["owner"] = self._person(doc.owner)
        info["approved_by"] = self._person(doc.approved_by)
        info["org_name"] = self.kb.directory.org_name(doc.org)
        info["audience_names"] = [self.kb.directory.role_name(r) for r in doc.audience]
        info["policy"] = self.kb.decide(viewer, doc).reason  # ok | draft | archived | no_access
        return info

    # ---------- asking ----------

    def ask(self, user_id: str, question: str, mode: str = "grounded", provider: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")
        if len(question) > 1000:
            raise ValueError("Question is too long (1000 characters max)")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        asker = self.user(user_id)
        guarded = mode == "grounded"
        llm = get_provider(provider or self.settings.provider, self.settings)
        flt = AccessFilter(tuple(asker.roles) if guarded else None)
        candidates = [d.id for d in self.kb.docs.values() if not guarded or self.kb.decide(asker, d).allowed]

        chat = self._small_talk(question, asker)
        if chat is not None:
            return chat

        if llm is not None and self.settings.llm_router:
            rt = route_with_llm(llm, question, asker, self.kb, candidates)
        else:
            rt = route(question, asker, self.kb, candidates)

        result: dict[str, Any] = {
            "id": f"ask_{uuid.uuid4().hex[:10]}",
            "user": self._person(asker.id),
            "question": question,
            "mode": mode,
            "provider": llm.name if llm else "offline",
            "route": rt.__dict__,
            "answer": "",
            "abstained": False,
            "citations": [],
            "graph": None,
            "trace": {"retrieved": [], "excluded": {}, "confidence": None, "verification": [], "draft": ""},
        }
        if rt.kind == "documents":
            self._answer_from_documents(result, asker, question, flt, llm, guarded)
        else:
            self._answer_from_graph(result, asker, question, rt, flt, guarded)

        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        self._log(
            asker.id,
            "ask",
            question=question,
            mode=mode,
            route=rt.kind,
            abstained=result["abstained"],
            cited=sorted({c["doc_id"] for c in result["citations"]}),
        )
        return result

    def _small_talk(self, question: str, asker: User) -> dict[str, Any] | None:
        """A friendly reply for greetings, thanks and "help", or None for real questions.
        Text with no searchable words at all ("???", "hmm") is treated as small talk too."""
        q = question.strip()
        if _THANKS.match(q):
            text = "You're welcome! Ask me anything else about Ridgeway Health's policies and procedures."
        elif _GREETING.match(q) or _HELP.match(q) or not content_terms(q):
            topics = sorted({d.title for d in self.kb.docs.values() if self.kb.decide(asker, d).allowed})
            listed = ", ".join(topics[:-1]) + f" and {topics[-1]}" if len(topics) > 1 else "".join(topics)
            text = (
                f"Hi {asker.name.split()[0]}! I answer questions about Ridgeway Health's policies and procedures, "
                f"using only approved documents you're allowed to read, and I cite every answer. "
                f"You can ask about the {listed}"
                + (", or relationship questions like who owns a policy or who you report to." if topics else ".")
            )
        else:
            return None
        self._log(asker.id, "ask", question=q, mode="chat", route="chat", abstained=False, cited=[])
        return {
            "id": f"ask_{uuid.uuid4().hex[:10]}",
            "user": self._person(asker.id),
            "question": q,
            "mode": "grounded",
            "provider": "offline",
            "route": {
                "kind": "chat",
                "intent": None,
                "params": {},
                "reason": "Greeting or small talk, not a policy question",
                "by": "rules",
            },
            "answer": text,
            "abstained": False,
            "citations": [],
            "graph": None,
            "trace": {"retrieved": [], "excluded": {}, "confidence": None, "verification": [], "draft": ""},
            "latency_ms": 0.0,
        }

    def _retrieve(self, result: dict[str, Any], asker: User, question: str, flt: AccessFilter) -> list[Source]:
        hits = self.retriever.search(question, flt, k=self.settings.top_k)
        sources = []
        for n, h in enumerate(hits, 1):
            c, doc = self.kb.chunks[h.chunk_id], self.kb.docs[h.doc_id]
            sources.append(Source(n, c.id, doc.id, doc.title, c.heading, c.text))
            result["trace"]["retrieved"].append(
                {
                    "n": n,
                    "chunk_id": c.id,
                    "doc_id": doc.id,
                    "title": doc.title,
                    "version": doc.version,
                    "heading": c.heading,
                    "score": h.score,
                    "bm25_rank": h.bm25_rank,
                    "vector_rank": h.vector_rank,
                    "policy": self.kb.decide(asker, doc).reason,
                }
            )
        if flt.active:
            # What the filter kept out, among passages this person can read. Passages they
            # can't read are not counted: even their existence is not disclosed.
            excluded: dict[str, int] = {}
            for h in self.retriever.search(question, AccessFilter(None), k=self.settings.top_k):
                reason = self.kb.decide(asker, self.kb.docs[h.doc_id]).reason
                if reason in {"draft", "archived"}:
                    excluded[reason] = excluded.get(reason, 0) + 1
            result["trace"]["excluded"] = excluded
        return sources

    def _cite(self, asker: User, src: Source, n: int) -> dict[str, Any]:
        doc = self.kb.docs[src.doc_id]
        return {
            "n": n,
            "kind": "passage",
            "chunk_id": src.chunk_id,
            "doc_id": doc.id,
            "title": doc.title,
            "version": doc.version,
            "status": doc.status,
            "heading": src.heading,
            "text": src.text,
            "owner": self._person(doc.owner),
            "policy": self.kb.decide(asker, doc).reason,
        }

    def _answer_from_documents(
        self,
        result: dict[str, Any],
        asker: User,
        question: str,
        flt: AccessFilter,
        llm: LLMProvider | None,
        guarded: bool,
    ) -> None:
        sources = self._retrieve(result, asker, question, flt)
        trace = result["trace"]
        if not sources:
            self._abstain(result, asker, "No passage you can use matches this question.")
            return
        if llm is None:
            draft, confidence = extractive_answer(question, sources, self.retriever.idf_of)
            trace["confidence"] = confidence
            if guarded and confidence < self.settings.abstain_below:
                trace["draft"] = draft
                self._abstain(result, asker, "The closest passages don't answer the question.", sources)
                return
        else:
            context = "\n\n".join(f"[{s.n}] {s.title} — {s.heading}: {s.text}" for s in sources)
            draft = llm.complete(ANSWER_SYSTEM, f"Sources:\n{context}\n\nQuestion: {question}").strip()
            if "NOT_FOUND" in draft:
                trace["draft"] = draft
                if guarded:
                    self._abstain(result, asker, "The model found no answer in the sources.", sources)
                    return
        trace["draft"] = draft
        check = verify(draft, sources, self.settings.min_support)
        trace["verification"] = [v.__dict__ for v in check.verdicts]
        if guarded:
            if not check.kept:
                self._abstain(result, asker, "No sentence of the draft answer was supported by its sources.", sources)
                return
            kept = check.kept
        else:
            kept = check.verdicts  # naive: whatever the draft said, verified or not
        by_n = {s.n: s for s in sources}
        renumber: dict[int, int] = {}
        parts = []
        for v in kept:
            cites = []
            for c in v.cites:
                if c in by_n:
                    renumber.setdefault(c, len(renumber) + 1)
                    cites.append(renumber[c])
            parts.append(v.sentence + "".join(f" [{c}]" for c in cites))
        result["answer"] = " ".join(parts) if parts else draft
        result["citations"] = [self._cite(asker, by_n[old], new) for old, new in renumber.items()]

    def _answer_from_graph(
        self, result: dict[str, Any], asker: User, question: str, rt: Route, flt: AccessFilter, guarded: bool
    ) -> None:
        params = {**rt.params, "roles": list(asker.roles), "guarded": guarded}
        intent = rt.intent or ""
        if "topic" in rt.params:  # find the document by search first, then ask the graph about it
            sources = self._retrieve(result, asker, question, flt)
            topic = [t for t in content_terms(question) if t not in _CONTACT_WORDS]
            best = None
            for s in sources:
                toks = set(tokenize(f"{s.title} {s.heading} {s.text}"))
                total = sum(self.retriever.idf_of(t) for t in topic) or 1.0
                cov = sum(self.retriever.idf_of(t) for t in topic if t in toks) / total
                if best is None or cov > best[0]:
                    best = (cov, s)
            result["trace"]["confidence"] = round(best[0], 3) if best else 0.0
            if not best or (guarded and best[0] < self.settings.abstain_below):
                self._abstain(
                    result, asker, "No document you can use covers this topic, so there's no owner to point to."
                )
                return
            if intent == "topic_contact":
                intent = "doc_owner"
                params = {**params, "doc_id": best[1].doc_id, "topic": True}
            else:
                params = {**params, "doc_id": best[1].doc_id, "topic": False}
        rows = self.graph.run(intent, params)
        answer = explain(intent, params, rows, self.kb)
        result["graph"] = answer.as_dict()
        if not rows:
            self._abstain(result, asker, "The knowledge graph has no answer you have access to.")
            return
        result["answer"] = answer.text
        for i, doc_id in enumerate(dict.fromkeys(answer.doc_ids), 1):
            doc = self.kb.docs[doc_id]
            result["citations"].append(
                {
                    "n": i,
                    "kind": "graph",
                    "doc_id": doc.id,
                    "title": doc.title,
                    "version": doc.version,
                    "status": doc.status,
                    "owner": self._person(doc.owner),
                    "policy": self.kb.decide(asker, doc).reason,
                }
            )

    def _abstain(self, result: dict[str, Any], asker: User, why: str, sources: list[Source] | None = None) -> None:
        result["abstained"] = True
        result["trace"]["abstain_reason"] = why
        text = "I can't answer that from the approved documents you have access to."
        # Point to a person only when an allowed document is clearly on the same topic.
        if sources and (result["trace"].get("confidence") or 0) >= 0.4:
            doc = self.kb.docs[sources[0].doc_id]
            owner = self._person(doc.owner)
            if owner and self.kb.decide(asker, doc).allowed:
                text += f" The closest document is the {doc.title}; its owner, {owner['name']} ({owner['org']}), may be able to help."
        result["answer"] = text
        result["citations"] = []

    # ---------- knowledge management ----------

    def documents(self, user_id: str) -> list[dict[str, Any]]:
        viewer = self.user(user_id)
        return [self._doc_info(d, viewer) for d in self.kb.visible_docs(viewer)]

    def document(self, user_id: str, doc_id: str) -> dict[str, Any]:
        viewer = self.user(user_id)
        doc = self.kb.docs.get(doc_id)
        if not doc or doc not in self.kb.visible_docs(viewer):
            raise NotFound("Document not found")  # same answer whether it exists or not
        info = self._doc_info(doc, viewer)
        info["body"] = doc.body
        info["chunks"] = [c.__dict__ for c in self.kb.chunks.values() if c.doc_id == doc_id]
        return info

    def add_document(
        self, user_id: str, title: str, body: str, audience: list[str], supersedes: str | None = None
    ) -> dict[str, Any]:
        author = self.user(user_id)
        title, body = title.strip(), body.strip()
        if not title or not body:
            raise ValueError("Title and text are required")
        version = 1
        if supersedes:
            old = self.kb.docs.get(supersedes)
            if not old or old not in self.kb.visible_docs(author):
                raise NotFound("The document to replace was not found")
            version = old.version + 1
        base = f"{_slug(title)}-v{version}"
        doc_id, i = base, 2
        while doc_id in self.kb.docs:
            doc_id, i = f"{base}-{i}", i + 1
        doc = Document(
            id=doc_id,
            title=title,
            version=version,
            status="draft",
            owner=author.id,
            org=author.org,
            audience=audience or ["employee"],
            effective=datetime.now(UTC).date().isoformat(),
            body=body if body.startswith("#") or "\n## " in body else f"## {title}\n{body}",
            supersedes=supersedes,
        )
        try:
            with self._lock:
                self.kb.add(doc)
        except KnowledgeError as exc:
            raise ValueError(str(exc)) from exc
        self._log(author.id, "add_document", doc_id=doc.id, supersedes=supersedes)
        return self._doc_info(doc, author)

    def approve(self, user_id: str, doc_id: str) -> dict[str, Any]:
        approver = self.user(user_id)
        doc = self.kb.docs.get(doc_id)
        if not doc or doc not in self.kb.visible_docs(approver):
            raise NotFound("Document not found")
        if not approver.is_admin:
            raise Forbidden("Only knowledge admins can approve documents")
        if doc.owner == approver.id:
            raise Forbidden("Owners can't approve their own documents; ask another knowledge admin")
        try:
            with self._lock:
                self.kb.approve(doc_id, approver.id)
        except KnowledgeError as exc:
            raise ValueError(str(exc)) from exc
        self._log(approver.id, "approve", doc_id=doc_id, archived=doc.supersedes)
        return self._doc_info(doc, approver)

    def graph_snapshot(self, user_id: str) -> dict[str, Any]:
        return self.graph.snapshot(self.user(user_id))

    def users(self) -> list[dict[str, Any]]:
        out = []
        for uid in self.kb.directory.demo_users:
            u = self.kb.directory.users[uid]
            out.append(
                {
                    **(self._person(uid) or {}),
                    "roles": [self.kb.directory.role_name(r) for r in u.roles if r != "employee"],
                    "admin": u.is_admin,
                    "suggestions": self.suggestions.get(uid, []),
                }
            )
        return out

    def evaluate(self, provider: str | None = None) -> dict[str, Any]:
        from .evaluation import run_eval

        return run_eval(self, provider)
