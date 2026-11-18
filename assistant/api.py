"""REST API and web console.

Who is asking decides what can be answered, so every route needs a user:

* Demo mode (no keys configured): the caller says who they are with `user`, so the web
  console can switch between employees. Never expose this mode on a real network.
* Keyed mode: set ASSISTANT_API_KEYS="KEY1:ana,KEY2:carla" and send `X-API-Key`. The user
  comes from the key and any `user` field is ignored. In production, put SSO in front and
  map the identity provider's user to a directory user the same way.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .core import Assistant, Forbidden, NotFound
from .evaluation import load_cases, run_eval
from .gateway import ProviderError, list_providers

STATIC = Path(__file__).parent / "static"


@lru_cache
def get_assistant() -> Assistant:
    return Assistant()


AssistantDep = Annotated[Assistant, Depends(get_assistant)]


def caller(a: Assistant, x_api_key: str | None, claimed: str | None) -> str:
    keys = a.settings.api_keys
    if keys:
        if not x_api_key or x_api_key not in keys:
            raise HTTPException(401, "Missing or invalid X-API-Key")
        return keys[x_api_key]
    if not claimed:
        raise HTTPException(422, "Say who is asking with `user` (demo mode)")
    return claimed


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    user: str | None = None
    mode: Literal["grounded", "naive"] = "grounded"
    provider: str | None = None


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    audience: list[str] = Field(default_factory=list)
    supersedes: str | None = None
    user: str | None = None


class UserIn(BaseModel):
    user: str | None = None


class EvalIn(BaseModel):
    holdout: bool = False
    provider: str | None = None


app = FastAPI(
    title="Grounded RAG Assistant",
    version=__version__,
    description="Answers only from approved knowledge the asker may see, cites every sentence, and abstains instead of guessing.",
)


@app.exception_handler(NotFound)
async def _not_found(_: Request, exc: NotFound) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(Forbidden)
async def _forbidden(_: Request, exc: Forbidden) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=403)


@app.exception_handler(ProviderError)
async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(ValueError)
async def _invalid(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=422)


Key = Annotated[str | None, Header()]


@app.get("/api/health")
def health(a: AssistantDep) -> dict:
    s = a.settings
    return {
        "status": "ok",
        "version": __version__,
        "provider": s.provider,
        "embedder": s.embedder,
        "vector_store": s.vector_store,
        "auth": s.auth_enabled,
        "documents": len(a.kb.docs),
    }


@app.get("/api/providers")
def providers(a: AssistantDep) -> list[dict]:
    return list_providers(a.settings)


@app.get("/api/users")
def users(a: AssistantDep) -> list[dict]:
    return a.users()


@app.post("/api/ask")
def ask(body: AskIn, a: AssistantDep, x_api_key: Key = None) -> dict:
    return a.ask(caller(a, x_api_key, body.user), body.question, body.mode, body.provider)


@app.get("/api/documents")
def documents(a: AssistantDep, user: str | None = None, x_api_key: Key = None) -> list[dict]:
    return a.documents(caller(a, x_api_key, user))


@app.post("/api/documents")
def add_document(body: DocumentIn, a: AssistantDep, x_api_key: Key = None) -> dict:
    return a.add_document(caller(a, x_api_key, body.user), body.title, body.text, body.audience, body.supersedes)


@app.get("/api/documents/{doc_id}")
def document(doc_id: str, a: AssistantDep, user: str | None = None, x_api_key: Key = None) -> dict:
    return a.document(caller(a, x_api_key, user), doc_id)


@app.post("/api/documents/{doc_id}/approve")
def approve(doc_id: str, body: UserIn, a: AssistantDep, x_api_key: Key = None) -> dict:
    return a.approve(caller(a, x_api_key, body.user), doc_id)


@app.get("/api/graph")
def graph(a: AssistantDep, user: str | None = None, x_api_key: Key = None) -> dict:
    return a.graph_snapshot(caller(a, x_api_key, user))


@app.post("/api/eval")
def evaluate(body: EvalIn, a: AssistantDep) -> dict:
    name = "eval_holdout.json" if body.holdout else "eval.json"
    fresh = Assistant(a.settings)  # documents added through the API don't change the score
    return run_eval(fresh, body.provider, load_cases(Path(a.settings.knowledge_path) / name))


@app.get("/api/audit")
def audit(a: AssistantDep, limit: Annotated[int, Query(ge=1, le=1000)] = 200, x_api_key: Key = None) -> list[dict]:
    if a.settings.auth_enabled:
        who = a.user(caller(a, x_api_key, None))
        if not who.is_admin:
            raise HTTPException(403, "Only knowledge admins can read the audit log")
    return a.audit[-limit:]


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
