"""Pinecone vector store over its REST data-plane API.

    ASSISTANT_VECTOR_STORE=pinecone
    PINECONE_API_KEY=...
    PINECONE_INDEX_HOST=https://my-index-abc123.svc.us-east1-gcp.pinecone.io
    PINECONE_NAMESPACE=ridgeway            # optional

The index's dimension must match the embedder (e.g. 1536 for text-embedding-3-small).
The access filter is sent as Pinecone metadata filtering, so restricted, draft and
archived passages are excluded inside Pinecone, before ranking.
"""

from __future__ import annotations

import os
from typing import Any

from ..retrieval import AccessFilter


def pinecone_filter(flt: AccessFilter) -> dict[str, Any] | None:
    if not flt.active:
        return None
    return {
        "$and": [
            {"status": {"$eq": "approved"}},
            {"current": {"$eq": True}},
            # `audience` is a list field: $in matches when any listed role is in it.
            {"audience": {"$in": list(flt.roles or ())}},
        ]
    }


class PineconeStore:
    name = "pinecone"

    def __init__(self, host: str, api_key: str, namespace: str = "", client: Any = None) -> None:
        import httpx

        self.namespace = namespace
        self.client = client or httpx.Client(
            base_url=host.rstrip("/"), headers={"Api-Key": api_key, "X-Pinecone-API-Version": "2025-04"}, timeout=30
        )

    @classmethod
    def from_env(cls) -> PineconeStore:
        host, key = os.environ.get("PINECONE_INDEX_HOST"), os.environ.get("PINECONE_API_KEY")
        if not host or not key:
            raise RuntimeError("Set PINECONE_INDEX_HOST and PINECONE_API_KEY to use Pinecone")
        return cls(host, key, os.environ.get("PINECONE_NAMESPACE", ""))

    def upsert(self, items: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        for i in range(0, len(items), 100):
            vectors = [{"id": cid, "values": vec, "metadata": meta} for cid, vec, meta in items[i : i + 100]]
            resp = self.client.post("/vectors/upsert", json={"vectors": vectors, "namespace": self.namespace})
            resp.raise_for_status()

    def query(self, vector: list[float], k: int, flt: AccessFilter) -> list[tuple[str, float]]:
        body: dict[str, Any] = {"vector": vector, "topK": k, "namespace": self.namespace, "includeMetadata": False}
        f = pinecone_filter(flt)
        if f:
            body["filter"] = f
        resp = self.client.post("/query", json=body)
        resp.raise_for_status()
        return [(m["id"], float(m["score"])) for m in resp.json().get("matches", [])]
