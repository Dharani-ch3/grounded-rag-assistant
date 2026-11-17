"""Settings read from environment variables, with safe defaults for local demos."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_KNOWLEDGE = str(Path(__file__).resolve().parent.parent / "knowledge")


def _parse_keys(raw: str) -> dict[str, str]:
    """ASSISTANT_API_KEYS="key1:ana,key2:carla" -> {key: user_id}."""
    keys: dict[str, str] = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        key, sep, user = entry.partition(":")
        if not sep or not key or not user:
            raise ValueError(f"ASSISTANT_API_KEYS entry must be key:user_id, got {entry!r}")
        keys[key] = user
    return keys


@dataclass(frozen=True)
class Settings:
    knowledge_path: str = DEFAULT_KNOWLEDGE
    provider: str = "offline"  # offline | openai | anthropic | bedrock | ollama
    embedder: str = "hashing"  # hashing | openai | ollama | bedrock
    vector_store: str = "memory"  # memory | pinecone | weaviate
    graph_store: str = "memory"  # memory | neo4j
    llm_router: bool = False  # let the model pick graph intents (from a fixed menu)
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    anthropic_model: str = "claude-sonnet-5-5"
    bedrock_model: str = "anthropic.claude-3-5-haiku-20241022-v1:0"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"
    aws_region: str = "us-east-1"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embed_model: str = "nomic-embed-text"
    top_k: int = 5
    abstain_below: float = 0.6  # minimum question coverage for an extractive answer
    min_support: float = 0.6  # minimum share of a sentence's words found in its cited passage
    audit_path: str = ""  # optional JSONL audit log file
    api_keys: dict[str, str] = field(default_factory=dict)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


@lru_cache
def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        knowledge_path=env("ASSISTANT_KNOWLEDGE_PATH", DEFAULT_KNOWLEDGE),
        provider=env("ASSISTANT_PROVIDER", "offline").lower(),
        embedder=env("ASSISTANT_EMBEDDER", "hashing").lower(),
        vector_store=env("ASSISTANT_VECTOR_STORE", "memory").lower(),
        graph_store=env("ASSISTANT_GRAPH_STORE", "memory").lower(),
        llm_router=env("ASSISTANT_LLM_ROUTER", "false").lower() in {"1", "true", "yes"},
        openai_model=env("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embed_model=env("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        anthropic_model=env("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        bedrock_model=env("BEDROCK_MODEL", "anthropic.claude-3-5-haiku-20241022-v1:0"),
        bedrock_embed_model=env("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0"),
        aws_region=env("AWS_REGION", "us-east-1"),
        ollama_url=env("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=env("OLLAMA_MODEL", "llama3.1"),
        ollama_embed_model=env("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        audit_path=env("ASSISTANT_AUDIT_PATH", ""),
        api_keys=_parse_keys(env("ASSISTANT_API_KEYS", "")),
    )
