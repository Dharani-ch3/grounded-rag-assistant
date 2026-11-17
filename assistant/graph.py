"""Knowledge graph for questions that span people, roles, departments and documents.

Vector search is good at "what does the policy say?" and bad at "who owns it?" or "which
documents can nurses read?". Those are relationship questions, so they are answered from a
graph:

    (User)-[:HAS_ROLE]->(Role)          (User)-[:MEMBER_OF]->(Org)
    (User)-[:REPORTS_TO]->(User)        (User)-[:OWNS|APPROVED]->(Document)
    (Role)-[:CAN_READ]->(Document)      (Document)-[:BELONGS_TO]->(Org)
    (Document)-[:SUPERSEDES]->(Document)

The model never writes graph queries. A question is mapped to one of the fixed templates in
`CYPHER` and only its parameters vary, so there is nothing to inject. `MemoryGraph` answers
them in Python (the default and the browser demo); `backends.neo4j.Neo4jGraph` runs the same
Cypher against Neo4j and returns rows with the same keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .knowledge import Document, KnowledgeBase, User

# Shared access predicate. With $guarded = false (naive mode, for comparison) it is a no-op.
ACCESS = (
    "(NOT $guarded OR (d.status = 'approved' AND d.superseded_by IS NULL AND any(r IN d.audience WHERE r IN $roles)))"
)

CYPHER: dict[str, str] = {
    "doc_owner": f"""MATCH (u:User)-[:OWNS]->(d:Document {{id: $doc_id}})
WHERE {ACCESS}
MATCH (u)-[:MEMBER_OF]->(o:Org)
RETURN u.id AS person_id, u.name AS person, u.title AS title, o.name AS org,
       d.id AS doc_id, d.title AS document, d.version AS version""",
    "doc_approver": f"""MATCH (u:User)-[:APPROVED]->(d:Document {{id: $doc_id}})
WHERE {ACCESS}
MATCH (u)-[:MEMBER_OF]->(o:Org)
RETURN u.id AS person_id, u.name AS person, u.title AS title, o.name AS org,
       d.id AS doc_id, d.title AS document, d.version AS version""",
    "role_documents": f"""MATCH (:Role {{id: $role}})-[:CAN_READ]->(d:Document)
WHERE {ACCESS}
MATCH (d)-[:BELONGS_TO]->(o:Org)
RETURN d.id AS doc_id, d.title AS document, d.version AS version, o.name AS org
ORDER BY document""",
    "user_manager": """MATCH (u:User {id: $user_id})-[:REPORTS_TO]->(m:User)-[:MEMBER_OF]->(o:Org)
RETURN u.name AS subject, m.id AS person_id, m.name AS person, m.title AS title, o.name AS org""",
    "org_members": """MATCH (u:User)-[:MEMBER_OF]->(o:Org {id: $org_id})
RETURN u.id AS person_id, u.name AS person, u.title AS title, o.name AS org
ORDER BY person""",
    "user_documents": f"""MATCH (u:User {{id: $user_id}})-[:OWNS]->(d:Document)
WHERE {ACCESS}
RETURN u.name AS subject, d.id AS doc_id, d.title AS document, d.version AS version
ORDER BY document""",
}


class GraphStore(Protocol):
    def run(self, intent: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...


@dataclass
class Edge:
    source: str
    rel: str
    target: str
    source_kind: str = ""
    target_kind: str = ""


@dataclass
class GraphAnswer:
    intent: str
    text: str
    rows: list[dict[str, Any]]
    path: list[Edge]
    cypher: str
    params: dict[str, Any]
    doc_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["path"] = [e.__dict__ for e in self.path]
        return d


class MemoryGraph:
    """In-process graph built from the knowledge base. Runs the CYPHER templates' logic in Python."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    def _allowed(self, doc: Document, params: dict[str, Any]) -> bool:
        if not params.get("guarded", True):
            return True
        roles = set(params.get("roles", []))
        return doc.status == "approved" and not doc.superseded_by and bool(roles & set(doc.audience))

    def _person(self, user: User) -> dict[str, Any]:
        return {
            "person_id": user.id,
            "person": user.name,
            "title": user.title,
            "org": self.kb.directory.org_name(user.org),
        }

    def run(self, intent: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        kb, users = self.kb, self.kb.directory.users
        if intent in {"doc_owner", "doc_approver"}:
            doc = kb.docs.get(params["doc_id"])
            who = (doc.owner if intent == "doc_owner" else doc.approved_by) if doc else None
            if not doc or who not in users or not self._allowed(doc, params):
                return []
            return [{**self._person(users[who]), "doc_id": doc.id, "document": doc.title, "version": doc.version}]
        if intent == "role_documents":
            rows = [
                {"doc_id": d.id, "document": d.title, "version": d.version, "org": kb.directory.org_name(d.org)}
                for d in kb.docs.values()
                if params["role"] in d.audience and self._allowed(d, params)
            ]
            return sorted(rows, key=lambda r: (r["document"], r["version"]))
        if intent == "user_manager":
            user = users.get(params["user_id"])
            if not user or user.manager not in users:
                return []
            return [{"subject": user.name, **self._person(users[user.manager])}]
        if intent == "org_members":
            return sorted(
                (self._person(u) for u in users.values() if u.org == params["org_id"]), key=lambda r: r["person"]
            )
        if intent == "user_documents":
            user = users.get(params["user_id"])
            if not user:
                return []
            rows = [
                {"subject": user.name, "doc_id": d.id, "document": d.title, "version": d.version}
                for d in kb.docs.values()
                if d.owner == user.id and self._allowed(d, params)
            ]
            return sorted(rows, key=lambda r: (r["document"], r["version"]))
        raise ValueError(f"Unknown graph intent {intent!r}")

    # ---------- whole-graph view for the UI ----------

    def snapshot(self, viewer: User | None, guarded: bool = True) -> dict[str, Any]:
        """Nodes and edges, with documents the viewer can't use left out entirely."""
        kb = self.kb
        params = {"guarded": guarded, "roles": viewer.roles if viewer else []}
        docs = [d for d in kb.docs.values() if self._allowed(d, params)]
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []
        for u in kb.directory.users.values():
            nodes.append({"id": f"user:{u.id}", "kind": "User", "label": u.name, "sub": u.title})
            edges.append({"source": f"user:{u.id}", "rel": "MEMBER_OF", "target": f"org:{u.org}"})
            if u.manager:
                edges.append({"source": f"user:{u.id}", "rel": "REPORTS_TO", "target": f"user:{u.manager}"})
            for r in u.roles:
                edges.append({"source": f"user:{u.id}", "rel": "HAS_ROLE", "target": f"role:{r}"})
        for o in kb.directory.orgs.values():
            nodes.append({"id": f"org:{o['id']}", "kind": "Org", "label": o["name"], "sub": ""})
        for r in kb.directory.roles.values():
            nodes.append({"id": f"role:{r['id']}", "kind": "Role", "label": r["name"], "sub": ""})
        for d in docs:
            nodes.append(
                {"id": f"doc:{d.id}", "kind": "Document", "label": d.title, "sub": f"v{d.version} · {d.status}"}
            )
            edges.append({"source": f"user:{d.owner}", "rel": "OWNS", "target": f"doc:{d.id}"})
            if d.approved_by:
                edges.append({"source": f"user:{d.approved_by}", "rel": "APPROVED", "target": f"doc:{d.id}"})
            edges.append({"source": f"doc:{d.id}", "rel": "BELONGS_TO", "target": f"org:{d.org}"})
            for r in d.audience:
                edges.append({"source": f"role:{r}", "rel": "CAN_READ", "target": f"doc:{d.id}"})
            if d.supersedes and d.supersedes in {x.id for x in docs}:
                edges.append({"source": f"doc:{d.id}", "rel": "SUPERSEDES", "target": f"doc:{d.supersedes}"})
        return {"nodes": nodes, "edges": edges}


def _doc_label(row: dict[str, Any]) -> str:
    return f"{row['document']} (v{row['version']})"


def explain(intent: str, params: dict[str, Any], rows: list[dict[str, Any]], kb: KnowledgeBase) -> GraphAnswer:
    """Turn template rows into an answer sentence and the graph path behind it."""
    cypher = CYPHER[intent]
    shown = {k: v for k, v in params.items() if k != "guarded"}
    path: list[Edge] = []
    doc_ids = [r["doc_id"] for r in rows if "doc_id" in r]
    if not rows:
        return GraphAnswer(intent, "", [], [], cypher, shown, [])
    r0 = rows[0]
    if intent in {"doc_owner", "doc_approver"}:
        verb = "owns" if intent == "doc_owner" else "approved"
        rel = "OWNS" if intent == "doc_owner" else "APPROVED"
        text = f"{r0['person']} ({r0['title']}, {r0['org']}) {verb} the {_doc_label(r0)}."
        if params.get("topic"):
            text = f"Contact {r0['person']} ({r0['title']}, {r0['org']}), who owns the {_doc_label(r0)}."
        path = [
            Edge(r0["person"], rel, _doc_label(r0), "User", "Document"),
            Edge(r0["person"], "MEMBER_OF", r0["org"], "User", "Org"),
        ]
    elif intent == "role_documents":
        role = kb.directory.role_name(params["role"])
        names = ", ".join(_doc_label(r) for r in rows)
        text = f"The {role} role can read {len(rows)} document{'s' if len(rows) != 1 else ''} you also have access to: {names}."
        path = [Edge(role, "CAN_READ", _doc_label(r), "Role", "Document") for r in rows]
    elif intent == "user_manager":
        subject = "You report" if params.get("is_self") else f"{r0['subject']} reports"
        text = f"{subject} to {r0['person']} ({r0['title']}, {r0['org']})."
        path = [Edge(r0["subject"], "REPORTS_TO", r0["person"], "User", "User")]
    elif intent == "org_members":
        names = ", ".join(f"{r['person']} ({r['title']})" for r in rows)
        text = f"{r0['org']} has {len(rows)} member{'s' if len(rows) != 1 else ''}: {names}."
        path = [Edge(r["person"], "MEMBER_OF", r["org"], "User", "Org") for r in rows]
    elif intent == "user_documents":
        names = ", ".join(_doc_label(r) for r in rows)
        text = f"{r0['subject']} owns {len(rows)} document{'s' if len(rows) != 1 else ''} you can see: {names}."
        path = [Edge(r0["subject"], "OWNS", _doc_label(r), "User", "Document") for r in rows]
    else:
        raise ValueError(intent)
    return GraphAnswer(intent, text, rows, path, cypher, shown, doc_ids)
