from fastapi.testclient import TestClient

from assistant import api
from assistant.config import Settings
from assistant.core import Assistant


def client(settings: Settings | None = None) -> TestClient:
    instance = Assistant(settings or Settings())
    api.app.dependency_overrides[api.get_assistant] = lambda: instance
    return TestClient(api.app)


def test_demo_mode_flow():
    c = client()
    assert c.get("/api/health").json()["documents"] == 11
    assert len(c.get("/api/users").json()) == 5
    r = c.post("/api/ask", json={"user": "ana", "question": "What extension is the IT service desk?"}).json()
    assert "4357" in r["answer"]
    assert c.post("/api/ask", json={"question": "hi"}).status_code == 422  # must say who is asking
    assert c.get("/api/documents/exec-compensation-guidelines", params={"user": "ana"}).status_code == 404
    assert c.post("/api/documents/remote-work-policy-draft/approve", json={"user": "carla"}).status_code == 403
    d = c.post(
        "/api/documents", json={"user": "dev", "title": "Parking", "text": "Parking is free.", "audience": ["employee"]}
    ).json()
    assert d["status"] == "draft" and d["id"] == "parking-v1"
    assert (
        c.post("/api/documents", json={"user": "dev", "title": "X", "text": "y", "audience": ["pirate"]}).status_code
        == 422
    )
    assert "nodes" in c.get("/api/graph", params={"user": "dev"}).json()
    assert c.post("/api/eval", json={"holdout": True}).json()["grounded"]["grounding_errors"] == 0
    assert c.get("/").status_code == 200


def test_keyed_mode_ignores_claimed_user():
    c = client(Settings(api_keys={"k-ana": "ana", "k-omar": "omar"}))
    q = {"question": "What is the annual bonus target for executives?", "user": "carla"}
    assert c.post("/api/ask", json=q).status_code == 401
    r = c.post("/api/ask", json=q, headers={"X-API-Key": "k-ana"}).json()
    assert r["user"]["id"] == "ana" and r["abstained"]
    assert c.get("/api/audit", headers={"X-API-Key": "k-ana"}).status_code == 403
    assert c.get("/api/audit", headers={"X-API-Key": "k-omar"}).json()[-1]["actor"] == "ana"
