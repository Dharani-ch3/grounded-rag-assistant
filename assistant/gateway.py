"""Model gateway: one interface for every model, cloud or local.

Engine code calls `provider.complete(system, user)` and never imports a vendor SDK, so
switching Bedrock -> Anthropic -> OpenAI -> a local model on Ollama is a config change.
`offline` uses no model: answers are quoted sentences, which keeps the public demo free
and lets the assistant run fully air-gapped.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .config import Settings


class ProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


def _post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    import httpx  # imported lazily so the browser demo doesn't need it

    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=httpx.Timeout(90.0, connect=10.0))
    except httpx.HTTPError as exc:
        raise ProviderError(f"Could not reach {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise ProviderError(f"{url} returned {resp.status_code}: {resp.text[:300]}")
    return resp.json()


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.openai_model
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = _post(f"{self.base}/chat/completions", {"Authorization": f"Bearer {self.key}"}, payload)
        return data["choices"][0]["message"]["content"]


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.anthropic_model
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        if not self.model:
            raise ProviderError("Set ANTHROPIC_MODEL to the Anthropic model id you want to use")
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            {
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0,
                "system": system + ("\nRespond with a single JSON object and nothing else." if json_mode else ""),
                "messages": [{"role": "user", "content": user}],
            },
        )
        return "".join(block.get("text", "") for block in data.get("content", []))


class BedrockProvider:
    """Amazon Bedrock via the Converse API (works with Amazon Nova, Llama, Mistral and other Bedrock models)."""

    name = "bedrock"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.bedrock_model
        self.region = settings.aws_region
        self._client: Any = None

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ProviderError("Bedrock needs boto3: pip install boto3") from exc
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        try:
            resp = self._client.converse(
                modelId=self.model,
                system=[{"text": system + ("\nRespond with JSON only." if json_mode else "")}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"temperature": 0, "maxTokens": 1024},
            )
        except Exception as exc:  # botocore raises many types
            raise ProviderError(f"Bedrock call failed: {exc}") from exc
        return "".join(part.get("text", "") for part in resp["output"]["message"]["content"])


class OllamaProvider:
    """Local models (Llama, Gemma, Mistral) for offline or air-gapped deployments."""

    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.ollama_model
        self.url = settings.ollama_url

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            payload["format"] = "json"
        return _post(f"{self.url}/api/chat", {}, payload)["message"]["content"]


PROVIDERS: dict[str, Callable[[Settings], LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "bedrock": BedrockProvider,
    "ollama": OllamaProvider,
}


def is_configured(name: str) -> bool:
    env = os.environ.get
    return {
        "offline": True,
        "openai": bool(env("OPENAI_API_KEY")),
        "anthropic": bool(env("ANTHROPIC_API_KEY") and env("ANTHROPIC_MODEL")),
        "bedrock": bool(env("AWS_ACCESS_KEY_ID") or env("AWS_PROFILE") or env("AWS_ROLE_ARN")),
        "ollama": bool(env("OLLAMA_URL")),
    }.get(name, name in PROVIDERS)


def get_provider(name: str, settings: Settings) -> LLMProvider | None:
    """The LLM provider, or None for the offline (extractive) answerer."""
    name = name.lower()
    if name == "offline":
        return None
    if name not in PROVIDERS:
        raise ProviderError(f"Unknown provider {name!r}. Choose from: offline, {', '.join(PROVIDERS)}")
    return PROVIDERS[name](settings)


def list_providers(settings: Settings) -> list[dict[str, Any]]:
    models = {
        "offline": "extractive (no model)",
        "openai": settings.openai_model,
        "anthropic": settings.anthropic_model or "set ANTHROPIC_MODEL",
        "bedrock": settings.bedrock_model,
        "ollama": settings.ollama_model,
    }
    return [
        {"name": n, "model": models.get(n, "custom"), "configured": is_configured(n), "default": n == settings.provider}
        for n in ["offline", *PROVIDERS]
    ]


# ---------- embedders ----------


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.openai_embed_model
        self.name = f"openai:{self.model}"
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):
            data = _post(
                f"{self.base}/embeddings",
                {"Authorization": f"Bearer {self.key}"},
                {"model": self.model, "input": list(texts[i : i + 96])},
            )
            out.extend(item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"]))
        return [_unit(v) for v in out]


class OllamaEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.ollama_embed_model
        self.name = f"ollama:{self.model}"
        self.url = settings.ollama_url

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            _unit(v)
            for v in _post(f"{self.url}/api/embed", {}, {"model": self.model, "input": list(texts)})["embeddings"]
        ]


class BedrockEmbedder:
    """Amazon Titan text embeddings on Bedrock."""

    def __init__(self, settings: Settings) -> None:
        self.model = settings.bedrock_embed_model
        self.name = f"bedrock:{self.model}"
        self.region = settings.aws_region
        self._client: Any = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        out = []
        for text in texts:
            resp = self._client.invoke_model(modelId=self.model, body=json.dumps({"inputText": text}))
            out.append(_unit(json.loads(resp["body"].read())["embedding"]))
        return out


def _unit(v: Sequence[float]) -> list[float]:
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def get_embedder(settings: Settings) -> Any:
    from .retrieval import HashingEmbedder

    return {
        "openai": OpenAIEmbedder,
        "ollama": OllamaEmbedder,
        "bedrock": BedrockEmbedder,
    }.get(settings.embedder, lambda _s: HashingEmbedder())(settings)
