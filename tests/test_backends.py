import json

import httpx

from assistant.backends.pinecone import PineconeStore, pinecone_filter
from assistant.backends.weaviate import WeaviateStore, weaviate_where
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
