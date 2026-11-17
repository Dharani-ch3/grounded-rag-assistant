"""Permission-aware hybrid retrieval.

BM25 catches exact terms ("Sev1", "PTO"); dense vectors catch paraphrases. The two rankings
are merged with Reciprocal Rank Fusion. The access filter is applied *before* ranking, in
both retrievers, so a passage the asker can't use never competes for a slot in the context.

Dense search goes through the `VectorStore` interface. `MemoryVectorStore` is the default;
`backends.pinecone` and `backends.weaviate` push the same filter down into those databases.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .knowledge import Chunk, KnowledgeBase
from .text import tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Local, deterministic embeddings via feature hashing of unigrams and bigrams.

    No model download and no API cost, so the demo runs anywhere, including in a browser.
    Swap in OpenAI, Bedrock Titan or Ollama embeddings with ASSISTANT_EMBEDDER.
    """

    name = "hashing-512"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            toks = tokenize(text)
            feats = Counter(toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)])
            vec = [0.0] * self.dim
            for feat, tf in feats.items():
                idx, sign = self._bucket(feat)
                vec[idx] += sign * (1.0 + math.log(tf))
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


# ---------- access filter, expressed as data so vector databases can apply it ----------


@dataclass(frozen=True)
class AccessFilter:
    """None roles = no filtering (naive mode, used only for comparison and evaluation)."""

    roles: tuple[str, ...] | None

    @property
    def active(self) -> bool:
        return self.roles is not None

    def matches(self, meta: dict[str, Any]) -> bool:
        if self.roles is None:
            return True
        return meta["status"] == "approved" and meta["current"] and bool(set(self.roles) & set(meta["audience"]))


def chunk_metadata(kb: KnowledgeBase, chunk: Chunk) -> dict[str, Any]:
    doc = kb.docs[chunk.doc_id]
    return {
        "doc_id": doc.id,
        "status": doc.status,
        "current": doc.superseded_by is None,
        "audience": list(doc.audience),
        "org": doc.org,
    }


class VectorStore(Protocol):
    name: str

    def upsert(self, items: list[tuple[str, list[float], dict[str, Any]]]) -> None: ...

    def query(self, vector: list[float], k: int, flt: AccessFilter) -> list[tuple[str, float]]: ...


class MemoryVectorStore:
    name = "memory"

    def __init__(self) -> None:
        self.items: dict[str, tuple[list[float], dict[str, Any]]] = {}

    def upsert(self, items: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        for cid, vec, meta in items:
            self.items[cid] = (vec, meta)

    def query(self, vector: list[float], k: int, flt: AccessFilter) -> list[tuple[str, float]]:
        scored = []
        for cid, (vec, meta) in self.items.items():
            if not flt.matches(meta):
                continue
            s = sum(x * y for x, y in zip(vector, vec, strict=False))
            if s > 0.05:
                scored.append((cid, s))
        return sorted(scored, key=lambda x: -x[1])[:k]


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    score: float
    bm25_rank: int | None
    vector_rank: int | None


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, embedder: Embedder, store: VectorStore, k1: float = 1.5, b: float = 0.75):
        self.kb, self.embedder, self.store = kb, embedder, store
        self.k1, self.b = k1, b
        self._lock = threading.Lock()
        self._built_version = -1
        self._embedded: set[str] = set()

    def _build(self) -> None:
        """(Re)index when the knowledge base changes. Only new chunks are embedded;
        metadata (status, audience) is re-sent for all, since approvals change it."""
        kb = self.kb
        self.ids = list(kb.chunks)
        self.meta = {cid: chunk_metadata(kb, kb.chunks[cid]) for cid in self.ids}
        self.tokens = {
            cid: tokenize(kb.chunks[cid].full_text + " " + kb.docs[kb.chunks[cid].doc_id].title) for cid in self.ids
        }
        self.tf = {cid: Counter(t) for cid, t in self.tokens.items()}
        self.avgdl = sum(len(t) for t in self.tokens.values()) / max(1, len(self.tokens))
        df: Counter[str] = Counter()
        for t in self.tokens.values():
            df.update(set(t))
        n = len(self.ids)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}
        new = [cid for cid in self.ids if cid not in self._embedded]
        vectors = dict(zip(new, self.embedder.embed([kb.chunks[c].full_text for c in new]), strict=True)) if new else {}
        if isinstance(self.store, MemoryVectorStore):
            self.store.upsert([(cid, vectors[cid], self.meta[cid]) for cid in new])
            for cid in self.ids:  # refresh metadata in place
                vec, _ = self.store.items[cid]
                self.store.items[cid] = (vec, self.meta[cid])
        else:
            missing = [cid for cid in self.ids if cid not in vectors]
            if missing:  # external stores need the vector re-sent with new metadata
                vectors.update(
                    zip(missing, self.embedder.embed([kb.chunks[c].full_text for c in missing]), strict=True)
                )
            self.store.upsert([(cid, vectors[cid], self.meta[cid]) for cid in self.ids])
        self._embedded.update(new)
        self._built_version = kb.version

    def ensure_built(self) -> None:
        with self._lock:
            if self._built_version != self.kb.version:
                self._build()

    def idf_of(self, term: str) -> float:
        return self.idf.get(term, math.log(1 + len(self.ids)))

    def _bm25(self, query: str, flt: AccessFilter) -> list[tuple[str, float]]:
        q = tokenize(query)
        scores = []
        for cid in self.ids:
            if not flt.matches(self.meta[cid]):
                continue
            tf, dl, s = self.tf[cid], len(self.tokens[cid]), 0.0
            for term in q:
                f = tf.get(term)
                if f:
                    s += (
                        self.idf[term]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
                    )
            if s > 0:
                scores.append((cid, s))
        return sorted(scores, key=lambda x: -x[1])

    def search(self, query: str, flt: AccessFilter, k: int = 6, rrf_k: int = 60) -> list[Hit]:
        self.ensure_built()
        if not query.strip():
            return []
        bm25 = self._bm25(query, flt)[: k * 3]
        dense = self.store.query(self.embedder.embed([query])[0], k * 3, flt)
        b_rank = {cid: r for r, (cid, _) in enumerate(bm25, 1)}
        v_rank = {cid: r for r, (cid, _) in enumerate(dense, 1)}
        fused: dict[str, float] = {}
        for ranks in (b_rank, v_rank):
            for cid, r in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + r)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        best = top[0][1] if top else 1.0
        return [
            Hit(cid, self.kb.chunks[cid].doc_id, round(s / best, 3), b_rank.get(cid), v_rank.get(cid)) for cid, s in top
        ]
