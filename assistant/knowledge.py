"""The knowledge base: documents with lifecycle metadata, the people directory, and access policy.

The access policy lives here, in one function, and every path to content goes through it:
retrieval, graph lookups, source previews and the document list. A passage the asker can't
see is never ranked, never sent to a model and never counted in a response.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .text import chunk_markdown

STATUSES = ("draft", "approved", "archived")


class KnowledgeError(ValueError):
    pass


@dataclass
class User:
    id: str
    name: str
    title: str
    org: str
    roles: list[str]
    manager: str | None = None

    @property
    def is_admin(self) -> bool:
        return "knowledge_admin" in self.roles


@dataclass
class Document:
    id: str
    title: str
    version: int
    status: str
    owner: str
    org: str
    audience: list[str]
    effective: str
    body: str
    approved_by: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("body")
        return d


@dataclass
class Chunk:
    id: str
    doc_id: str
    ord: int
    heading: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.heading}. {self.text}" if self.heading else self.text


@dataclass
class Decision:
    allowed: bool
    reason: str  # ok | draft | archived | superseded | no_access


@dataclass
class Directory:
    users: dict[str, User]
    orgs: dict[str, dict[str, Any]]
    roles: dict[str, dict[str, Any]]
    demo_users: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Directory:
        raw = json.loads(path.read_text())
        return cls(
            users={u["id"]: User(**u) for u in raw["users"]},
            orgs={o["id"]: o for o in raw["orgs"]},
            roles={r["id"]: r for r in raw["roles"]},
            demo_users=raw.get("demo_users", []),
        )

    def user(self, user_id: str) -> User:
        if user_id not in self.users:
            raise KnowledgeError(f"Unknown user {user_id!r}")
        return self.users[user_id]

    def name(self, user_id: str | None) -> str:
        return self.users[user_id].name if user_id in self.users else (user_id or "")

    def org_name(self, org_id: str) -> str:
        return self.orgs.get(org_id, {}).get("name", org_id)

    def role_name(self, role_id: str) -> str:
        return self.roles.get(role_id, {}).get("name", role_id)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.S)
    if not m:
        raise KnowledgeError("Document must start with a --- front matter block")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, m.group(2).strip()


def document_from_markdown(raw: str) -> Document:
    meta, body = parse_front_matter(raw)
    missing = {"id", "title", "status", "owner", "org", "audience"} - set(meta)
    if missing:
        raise KnowledgeError(f"Front matter is missing: {', '.join(sorted(missing))}")
    if meta["status"] not in STATUSES:
        raise KnowledgeError(f"status must be one of {STATUSES}")
    return Document(
        id=meta["id"],
        title=meta["title"],
        version=int(meta.get("version", "1")),
        status=meta["status"],
        owner=meta["owner"],
        org=meta["org"],
        audience=[a.strip() for a in meta["audience"].split(",") if a.strip()],
        effective=meta.get("effective", ""),
        body=body,
        approved_by=meta.get("approved_by") or None,
        supersedes=meta.get("supersedes") or None,
    )


class KnowledgeBase:
    def __init__(self, directory: Directory, chunk_words: int = 90) -> None:
        self.directory = directory
        self.docs: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.chunk_words = chunk_words
        self.version = 0  # bumps on every change so indexes know to rebuild

    @classmethod
    def load(cls, root: Path) -> KnowledgeBase:
        kb = cls(Directory.load(root / "directory.json"))
        for path in sorted((root / "documents").glob("*.md")):
            kb.add(document_from_markdown(path.read_text()))
        kb.link_versions()
        return kb

    def add(self, doc: Document) -> Document:
        if doc.id in self.docs:
            raise KnowledgeError(f"A document with id {doc.id!r} already exists")
        if doc.owner not in self.directory.users:
            raise KnowledgeError(f"Unknown owner {doc.owner!r}")
        unknown = set(doc.audience) - set(self.directory.roles)
        if unknown:
            raise KnowledgeError(f"Unknown audience role(s): {', '.join(sorted(unknown))}")
        self.docs[doc.id] = doc
        for i, (heading, text) in enumerate(chunk_markdown(doc.body, self.chunk_words)):
            c = Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, ord=i, heading=heading, text=text)
            self.chunks[c.id] = c
        self.version += 1
        return doc

    def link_versions(self) -> None:
        for doc in self.docs.values():
            if doc.supersedes and doc.supersedes in self.docs and doc.status == "approved":
                self.docs[doc.supersedes].superseded_by = doc.id

    def approve(self, doc_id: str, approver: str) -> Document:
        doc = self.docs[doc_id]
        if doc.status != "draft":
            raise KnowledgeError(f"Only drafts can be approved; this document is {doc.status}")
        doc.status, doc.approved_by = "approved", approver
        if doc.supersedes and doc.supersedes in self.docs:
            old = self.docs[doc.supersedes]
            old.status, old.superseded_by = "archived", doc.id
        self.version += 1
        return doc

    # ---------- the access policy ----------

    def can_read(self, user: User, doc: Document) -> bool:
        return bool(set(user.roles) & set(doc.audience))

    def decide(self, user: User, doc: Document) -> Decision:
        """May this passage be used to answer this user? Order matters: access is checked first,
        so a restricted draft reports `no_access` and its lifecycle state is never revealed."""
        if not self.can_read(user, doc):
            return Decision(False, "no_access")
        if doc.status == "draft":
            return Decision(False, "draft")
        if doc.status == "archived" or doc.superseded_by:
            return Decision(False, "archived")
        return Decision(True, "ok")

    def visible_docs(self, user: User) -> list[Document]:
        """Documents listed for this user. Knowledge admins also see drafts and archives they can read."""
        out = []
        for doc in self.docs.values():
            d = self.decide(user, doc)
            if d.allowed or (user.is_admin and d.reason in {"draft", "archived"}):
                out.append(doc)
        return sorted(out, key=lambda d: (d.title, -d.version))
