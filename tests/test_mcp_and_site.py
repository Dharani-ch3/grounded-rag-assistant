import asyncio
import json
import subprocess
import sys
from pathlib import Path

from assistant import mcp_server
from assistant.config import Settings
from assistant.core import Assistant

ROOT = Path(__file__).resolve().parent.parent


def test_mcp_tools(monkeypatch):
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"ask_knowledge_base", "list_documents", "read_document"} <= names
    monkeypatch.setattr(mcp_server, "_assistant", Assistant(Settings()))
    monkeypatch.setenv("ASSISTANT_MCP_USER", "dev")
    r = mcp_server.ask_knowledge_base("What is the annual bonus target for executives?")
    assert r["abstained"] and r["citations"] == []
    assert "exec-compensation-guidelines" not in {d["id"] for d in mcp_server.list_documents()}


def test_static_site_build(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_site.py"), "--out", str(tmp_path)], check=True)
    html = (tmp_path / "index.html").read_text()
    assert 'src="./boot.js"' in html and 'href="./style.css"' in html
    bundle = json.loads((tmp_path / "bundle.json").read_text())
    assert "assistant/bridge.py" in bundle and "knowledge/documents/pto-policy-v3.md" in bundle
    assert "assistant/api.py" not in bundle
    # The browser bundle must not need third-party packages.
    for path, text in bundle.items():
        if path.endswith(".py"):
            assert "\nimport httpx" not in text and "\nfrom pydantic" not in text, path
