import json

from assistant import bridge


def call(method, url, body=None):
    return json.loads(bridge.dispatch(method, url, json.dumps(body)))


def test_bridge_routes():
    bridge.reset()
    assert call("GET", "/api/health")["data"]["runtime"] == "browser"
    r = call("POST", "/api/ask", {"user": "dev", "question": "What is the daily meal limit when traveling for work?"})
    assert r["status"] == 200 and "$75" in r["data"]["answer"]
    assert call("POST", "/api/ask", {"user": "dev"})["status"] == 422
    assert call("GET", "/api/documents/exec-compensation-guidelines?user=dev")["status"] == 404
    assert call("POST", "/api/documents/remote-work-policy-draft/approve", {"user": "carla"})["status"] == 403
    assert call("POST", "/api/documents/remote-work-policy-draft/approve", {"user": "omar"})["status"] == 200
    ev = call("POST", "/api/eval", {})["data"]
    assert ev["grounded"]["grounding_errors"] == 0  # scored on a fresh copy, unaffected by the approval
    assert call("GET", "/api/nope")["status"] == 404
