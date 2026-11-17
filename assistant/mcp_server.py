"""MCP server: lets Claude, Cursor or any MCP client ask the knowledge base as a tool.

The MCP client acts as one directory user, set with ASSISTANT_MCP_USER (default: ana),
so an agent gets exactly the access that person has and no more.

Run over stdio:   python -m assistant.mcp_server
Claude Desktop / Cursor config:
    {"mcpServers": {"grounded-rag": {"command": "python", "args": ["-m", "assistant.mcp_server"],
                                     "cwd": "/path/to/grounded-rag-assistant",
                                     "env": {"ASSISTANT_MCP_USER": "ana"}}}}
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from .core import Assistant

mcp = MCPServer(
    name="grounded-rag-assistant",
    instructions=(
        "Answers questions from Ridgeway Health's approved policies and procedures, with citations. "
        "If ask_knowledge_base returns abstained=true, tell the user the knowledge base doesn't answer it; "
        "do not fill the gap from your own knowledge."
    ),
)
_assistant: Assistant | None = None


def assistant() -> Assistant:
    global _assistant
    if _assistant is None:
        _assistant = Assistant()
    return _assistant


def acting_user() -> str:
    return os.environ.get("ASSISTANT_MCP_USER", "ana")


@mcp.tool()
def ask_knowledge_base(question: str) -> dict:
    """Ask a question about company policies, procedures, owners or approvers.
    Returns the answer, its citations, and abstained=true when the approved documents don't answer it."""
    r = assistant().ask(acting_user(), question)
    return {
        "answer": r["answer"],
        "abstained": r["abstained"],
        "citations": [
            {
                "n": c["n"],
                "document": c["title"],
                "version": c["version"],
                "section": c.get("heading", ""),
                "text": c.get("text", ""),
            }
            for c in r["citations"]
        ],
        "route": r["route"]["kind"],
    }


@mcp.tool()
def list_documents() -> list[dict]:
    """List the documents the acting user may read, with owner, version and status."""
    return [
        {
            "id": d["id"],
            "title": d["title"],
            "version": d["version"],
            "status": d["status"],
            "owner": (d["owner"] or {}).get("name"),
        }
        for d in assistant().documents(acting_user())
    ]


@mcp.tool()
def read_document(doc_id: str) -> dict:
    """Read one document's full text (only documents the acting user may read)."""
    d = assistant().document(acting_user(), doc_id)
    return {"id": d["id"], "title": d["title"], "version": d["version"], "status": d["status"], "text": d["body"]}


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
