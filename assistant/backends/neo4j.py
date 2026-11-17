"""Neo4j graph store: the same question templates, run as parameterized Cypher.

    ASSISTANT_GRAPH_STORE=neo4j
    NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=...

`sync` MERGEs users, roles, orgs and documents from the knowledge base. Queries only ever
run the fixed templates in `graph.CYPHER` with parameters, never model-written Cypher.
"""

from __future__ import annotations

import os
from typing import Any

from ..graph import CYPHER
from ..knowledge import KnowledgeBase

SYNC = [
    "MATCH (n) WHERE n:User OR n:Role OR n:Org OR n:Document DETACH DELETE n",
    "UNWIND $orgs AS o MERGE (:Org {id: o.id, name: o.name})",
    "UNWIND $roles AS r MERGE (:Role {id: r.id, name: r.name})",
    """UNWIND $users AS u
       MERGE (x:User {id: u.id}) SET x.name = u.name, x.title = u.title
       WITH x, u MATCH (o:Org {id: u.org}) MERGE (x)-[:MEMBER_OF]->(o)
       WITH x, u UNWIND u.roles AS rid MATCH (r:Role {id: rid}) MERGE (x)-[:HAS_ROLE]->(r)""",
    """UNWIND $users AS u WITH u WHERE u.manager IS NOT NULL
       MATCH (x:User {id: u.id}), (m:User {id: u.manager}) MERGE (x)-[:REPORTS_TO]->(m)""",
    """UNWIND $docs AS d
       MERGE (x:Document {id: d.id})
       SET x.title = d.title, x.version = d.version, x.status = d.status,
           x.audience = d.audience, x.superseded_by = d.superseded_by
       WITH x, d MATCH (o:Org {id: d.org}) MERGE (x)-[:BELONGS_TO]->(o)
       WITH x, d MATCH (u:User {id: d.owner}) MERGE (u)-[:OWNS]->(x)
       WITH x, d UNWIND d.audience AS rid MATCH (r:Role {id: rid}) MERGE (r)-[:CAN_READ]->(x)""",
    """UNWIND $docs AS d WITH d WHERE d.approved_by IS NOT NULL
       MATCH (u:User {id: d.approved_by}), (x:Document {id: d.id}) MERGE (u)-[:APPROVED]->(x)""",
    """UNWIND $docs AS d WITH d WHERE d.supersedes IS NOT NULL
       MATCH (new:Document {id: d.id}), (old:Document {id: d.supersedes}) MERGE (new)-[:SUPERSEDES]->(old)""",
]


class Neo4jGraph:
    def __init__(self, driver: Any, database: str | None = None) -> None:
        self.driver = driver
        self.database = database

    @classmethod
    def from_env(cls) -> Neo4jGraph:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Neo4j needs the driver: pip install neo4j") from exc
        uri = os.environ.get("NEO4J_URI")
        if not uri:
            raise RuntimeError("Set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD to use Neo4j")
        auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
        return cls(GraphDatabase.driver(uri, auth=auth), os.environ.get("NEO4J_DATABASE"))

    def sync(self, kb: KnowledgeBase) -> None:
        d = kb.directory
        params = {
            "orgs": [{"id": o["id"], "name": o["name"]} for o in d.orgs.values()],
            "roles": [{"id": r["id"], "name": r["name"]} for r in d.roles.values()],
            "users": [u.__dict__ for u in d.users.values()],
            "docs": [doc.summary() for doc in kb.docs.values()],
        }
        with self.driver.session(database=self.database) as session:
            for statement in SYNC:
                session.run(statement, params)

    def run(self, intent: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if intent not in CYPHER:
            raise ValueError(f"Unknown graph intent {intent!r}")
        cypher_params = {"guarded": True, "roles": [], **params}
        with self.driver.session(database=self.database) as session:
            return [dict(record) for record in session.run(CYPHER[intent], cypher_params)]
