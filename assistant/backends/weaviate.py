"""Weaviate vector store over its REST and GraphQL APIs (bring-your-own vectors).

    ASSISTANT_VECTOR_STORE=weaviate
    WEAVIATE_URL=https://my-cluster.weaviate.network
    WEAVIATE_API_KEY=...                    # optional for local Weaviate
    WEAVIATE_CLASS=KnowledgeChunk           # optional

The access filter becomes a Weaviate `where` filter on the nearVector query.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from ..retrieval import AccessFilter


def weaviate_where(flt: AccessFilter) -> dict[str, Any] | None:
    if not flt.active:
        return None
    return {
        "operator": "And",
        "operands": [
            {"path": ["status"], "operator": "Equal", "valueText": "approved"},
            {"path": ["current"], "operator": "Equal", "valueBoolean": True},
            {"path": ["audience"], "operator": "ContainsAny", "valueText": list(flt.roles or ())},
        ],
    }


def _graphql_value(value: Any) -> str:
    """Render a Python value as a GraphQL input literal (keys unquoted, enums bare)."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_graphql_value(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_graphql_value(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and value in {"And", "Or", "Equal", "ContainsAny"}:
        return value
    return json.dumps(value)


class WeaviateStore:
    name = "weaviate"

    def __init__(self, url: str, api_key: str = "", class_name: str = "KnowledgeChunk", client: Any = None) -> None:
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.class_name = class_name
        self.client = client or httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=30)

    @classmethod
    def from_env(cls) -> WeaviateStore:
        url = os.environ.get("WEAVIATE_URL")
        if not url:
            raise RuntimeError("Set WEAVIATE_URL to use Weaviate")
        return cls(url, os.environ.get("WEAVIATE_API_KEY", ""), os.environ.get("WEAVIATE_CLASS", "KnowledgeChunk"))

    @staticmethod
    def object_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert(self, items: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        for i in range(0, len(items), 100):
            objects = [
                {
                    "class": self.class_name,
                    "id": self.object_id(cid),
                    "vector": vec,
                    "properties": {"chunk_id": cid, **meta},
                }
                for cid, vec, meta in items[i : i + 100]
            ]
            self.client.post("/v1/batch/objects", json={"objects": objects}).raise_for_status()

    def query(self, vector: list[float], k: int, flt: AccessFilter) -> list[tuple[str, float]]:
        args = f"nearVector: {{vector: {json.dumps(vector)}}}, limit: {int(k)}"
        where = weaviate_where(flt)
        if where:
            args += f", where: {_graphql_value(where)}"
        query = f"{{ Get {{ {self.class_name}({args}) {{ chunk_id _additional {{ certainty }} }} }} }}"
        resp = self.client.post("/v1/graphql", json={"query": query})
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            raise RuntimeError(f"Weaviate query failed: {data['errors']}")
        rows = data["data"]["Get"][self.class_name] or []
        return [(r["chunk_id"], float(r["_additional"]["certainty"])) for r in rows]
