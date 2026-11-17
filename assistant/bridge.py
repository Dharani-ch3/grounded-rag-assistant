"""In-browser bridge: the REST API's routes as a plain function call.

The GitHub Pages demo runs this assistant inside the visitor's browser with Pyodide
(Python compiled to WebAssembly). There is no server there, so the web console sends its
requests to `dispatch()` instead of over HTTP. Same assistant code, same JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .config import Settings
from .core import Assistant, Forbidden, NotFound
from .evaluation import load_cases, run_eval
from .gateway import ProviderError, list_providers

_assistant: Assistant | None = None


def assistant() -> Assistant:
    global _assistant
    if _assistant is None:
        _assistant = Assistant(Settings())
    return _assistant


def reset() -> None:
    global _assistant
    _assistant = None


def _need(body: dict[str, Any], *keys: str) -> list[Any]:
    missing = [k for k in keys if not body.get(k)]
    if missing:
        raise ValueError(f"Missing field(s): {', '.join(missing)}")
    return [body[k] for k in keys]


def _route(method: str, path: str, query: dict[str, str], body: dict[str, Any]) -> Any:
    a = assistant()
    m = re.fullmatch
    if method == "GET" and path == "/api/health":
        return {"status": "ok", "provider": "offline", "runtime": "browser", "auth": False}
    if method == "GET" and path == "/api/providers":
        return [p for p in list_providers(a.settings) if p["name"] == "offline"]
    if method == "GET" and path == "/api/users":
        return a.users()
    if method == "POST" and path == "/api/ask":
        user, question = _need(body, "user", "question")
        return a.ask(user, question, body.get("mode", "grounded"))
    if path == "/api/documents":
        if method == "GET":
            return a.documents(query.get("user", ""))
        user, title, text = _need(body, "user", "title", "text")
        return a.add_document(user, title, text, body.get("audience") or [], body.get("supersedes") or None)
    if (g := m(r"/api/documents/([\w-]+)/approve", path)) and method == "POST":
        return a.approve(_need(body, "user")[0], g.group(1))
    if (g := m(r"/api/documents/([\w-]+)", path)) and method == "GET":
        return a.document(query.get("user", ""), g.group(1))
    if method == "GET" and path == "/api/graph":
        return a.graph_snapshot(query.get("user", ""))
    if method == "POST" and path == "/api/eval":
        name = "eval_holdout.json" if body.get("holdout") else "eval.json"
        # A fresh assistant, so documents added or approved in the demo don't change the score.
        return run_eval(Assistant(a.settings), cases=load_cases(Path(a.settings.knowledge_path) / name))
    if method == "GET" and path == "/api/audit":
        return a.audit[-200:]
    raise NotFound(f"No route for {method} {path}")


def dispatch(method: str, url: str, body_json: str = "null") -> str:
    """Handle one request. Returns JSON: {"status": int, "data": ...}."""
    parts = urlsplit(url)
    query = {k: v[-1] for k, v in parse_qs(parts.query).items()}
    try:
        body = json.loads(body_json) or {}
        data = _route(method.upper(), unquote(parts.path), query, body)
        status = 200
    except NotFound as exc:
        status, data = 404, {"detail": str(exc)}
    except Forbidden as exc:
        status, data = 403, {"detail": str(exc)}
    except ProviderError as exc:
        status, data = 400, {"detail": str(exc)}
    except ValueError as exc:
        status, data = 422, {"detail": str(exc)}
    return json.dumps({"status": status, "data": data})
