import json

import httpx

from assistant.backends.neo4j import Neo4jGraph
from assistant.backends.pinecone import PineconeStore, pinecone_filter
from assistant.backends.weaviate import WeaviateStore, weaviate_where
from assistant.config import Settings
from assistant.core import Assistant
from assistant.graph import CYPHER
from assistant.retrieval import AccessFilter

NURSE = AccessFilter(("employee", "nurse"))


def mock_client(base: str, handler) -> httpx.Client:
    return httpx.Client(base_url=base, transport=httpx.MockTransport(handler))


def test_pinecone_pushes_access_filter_down():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append((req.url.path, body))
        if req.url.path == "/query":
            return httpx.Response(200, json={"matches": [{"id": "pto-policy-v3#0", "score": 0.9}]})
        return httpx.Response(200, json={"upsertedCount": len(body["vectors"])})

    store = PineconeStore("https://idx.example", "key", "ns", client=mock_client("https://idx.example", handler))
    store.upsert(
        [("a#0", [0.1, 0.2], {"status": "approved", "current": True, "audience": ["employee"], "doc_id": "a"})]
    )
    assert store.query([0.1, 0.2], 3, NURSE) == [("pto-policy-v3#0", 0.9)]
    query = seen[-1][1]
    assert query["filter"] == pinecone_filter(NURSE)
    assert {"audience": {"$in": ["employee", "nurse"]}} in query["filter"]["$and"]
    store.query([0.1], 3, AccessFilter(None))
    assert "filter" not in seen[-1][1]


def test_weaviate_query_has_where_filter():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        if req.url.path == "/v1/graphql":
            return httpx.Response(
                200,
                json={"data": {"Get": {"KnowledgeChunk": [{"chunk_id": "x#0", "_additional": {"certainty": 0.8}}]}}},
            )
        return httpx.Response(200, json=[])

    store = WeaviateStore("http://w", client=mock_client("http://w", handler))
    store.upsert([("x#0", [1.0, 0.0], {"status": "approved"})])
    assert seen[0]["objects"][0]["id"] == WeaviateStore.object_id("x#0")
    assert store.query([1.0, 0.0], 2, NURSE) == [("x#0", 0.8)]
    q = seen[-1]["query"]
    assert "where: {operator: And" in q and 'valueText: ["employee", "nurse"]' in q and "ContainsAny" in q
    assert weaviate_where(AccessFilter(None)) is None


class FakeSession:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, params):
        self.log.append((cypher, params))
        return self.rows


class FakeDriver:
    def __init__(self, rows=()):
        self.log, self.rows = [], list(rows)

    def session(self, database=None):
        return FakeSession(self.log, self.rows)


def test_neo4j_runs_fixed_templates_with_parameters():
    row = {
        "person_id": "dev",
        "person": "Dev Patel",
        "title": "Finance Analyst",
        "org": "Finance",
        "doc_id": "expense-policy-v2",
        "document": "Expense Reimbursement Policy",
        "version": 2,
    }
    driver = FakeDriver([row])
    graph = Neo4jGraph(driver)
    a = Assistant(Settings(), graph_store=graph)
    r = a.ask("ana", "Who owns the expense reimbursement policy?")
    assert r["answer"].startswith("Dev Patel")
    cypher, params = driver.log[-1]
    assert cypher == CYPHER["doc_owner"] and params["doc_id"] == "expense-policy-v2" and params["guarded"] is True
    graph.sync(a.kb)
    assert any("MERGE (u)-[:OWNS]->(x)" in c for c, _ in driver.log)
    synced = driver.log[-1][1]
    assert {d["id"] for d in synced["docs"]} == set(a.kb.docs)


def test_cypher_templates_only_use_parameters():
    for cypher in CYPHER.values():
        assert "{" not in cypher.replace("{id: $", "").replace("}", "") or "$" in cypher
        assert "%s" not in cypher and "format" not in cypher
