import pytest

from assistant import core
from assistant.core import Forbidden, NotFound
from tests.conftest import FakeLLM


def ask(a, user, q, mode="grounded"):
    return a.ask(user, q, mode)


def test_current_version_beats_archived(assistant):
    r = ask(assistant, "ana", "How many PTO days do full-time employees get per year?")
    assert "20 days" in r["answer"] and not r["abstained"]
    assert [c["doc_id"] for c in r["citations"]] == ["pto-policy-v3"]
    assert r["trace"]["excluded"].get("archived", 0) >= 1


def test_naive_mode_shows_what_guardrails_prevent(assistant):
    r = ask(assistant, "ana", "How many PTO days do full-time employees get per year?", "naive")
    assert "15 days" in r["answer"] and r["citations"][0]["policy"] == "archived"
    r = ask(assistant, "ana", "How quickly must on-call respond to a Sev1 security incident?", "naive")
    assert r["citations"][0]["policy"] == "no_access"


def test_restricted_content_is_not_used_or_hinted(assistant):
    r = ask(assistant, "ana", "How quickly must on-call respond to a Sev1 security incident?")
    assert r["abstained"] and not r["citations"]
    assert "15 minutes" not in r["answer"] and "Runbook" not in r["answer"] and "Sam" not in r["answer"]
    assert all(h["doc_id"] != "incident-response-runbook" for h in r["trace"]["retrieved"])
    assert "no_access" not in r["trace"]["excluded"]


def test_drafts_are_not_used_until_approved(assistant):
    q = "How many days per week can I work remotely?"
    assert ask(assistant, "ben", q)["abstained"]
    with pytest.raises(NotFound):
        assistant.approve("ben", "remote-work-policy-draft")  # not an admin: drafts are invisible to him
    with pytest.raises(Forbidden):
        assistant.approve("ana", "pto-policy-v3")  # not a knowledge admin
    with pytest.raises(Forbidden):
        assistant.approve("carla", "remote-work-policy-draft")  # admin, but the author
    assistant.approve("omar", "remote-work-policy-draft")
    r = ask(assistant, "ben", q)
    assert "3 days per week" in r["answer"]
    assert [e["action"] for e in assistant.audit].count("approve") == 1


def test_new_version_archives_the_old_one_on_approval(assistant):
    doc = assistant.add_document(
        "dev",
        "Expense Reimbursement Policy",
        "## Travel meals\nMeals are reimbursed up to $90 per day.",
        ["employee"],
        "expense-policy-v2",
    )
    assert doc["status"] == "draft" and doc["version"] == 3
    assert "$75" in ask(assistant, "ana", "What is the daily meal limit when traveling for work?")["answer"]
    assistant.approve("omar", doc["id"])
    assert assistant.kb.docs["expense-policy-v2"].status == "archived"
    assert "$90" in ask(assistant, "ana", "What is the daily meal limit when traveling for work?")["answer"]


def test_documents_list_and_read_respect_access(assistant):
    assert "exec-compensation-guidelines" not in {d["id"] for d in assistant.documents("dev")}
    with pytest.raises(NotFound):
        assistant.document("dev", "exec-compensation-guidelines")
    assert assistant.document("carla", "exec-compensation-guidelines")["chunks"]


def test_graph_answers(assistant):
    r = ask(assistant, "ana", "Who owns the expense reimbursement policy?")
    assert r["answer"].startswith("Dev Patel") and r["graph"]["path"][0]["rel"] == "OWNS"
    assert "OWNS" in r["graph"]["lookup"] and r["graph"]["params"]["doc_id"] == "expense-policy-v2"
    r = ask(assistant, "ana", "Which documents can nurses read?")
    assert "Patient Intake Procedure" in r["answer"] and "v2) , " not in r["answer"]
    assert "Paid Time Off Policy (v2)" not in r["answer"]  # archived versions aren't listed


def test_graph_respects_access(assistant):
    assert ask(assistant, "dev", "Who should I contact about vaccine refrigerator problems?")["abstained"]
    r = ask(assistant, "dev", "Who should I contact about vaccine refrigerator problems?", "naive")
    assert "Maria Okafor" in r["answer"] and r["citations"][0]["policy"] == "no_access"


def test_llm_answers_are_verified(assistant, monkeypatch):
    llm = FakeLLM(
        "Full-time employees accrue 20 days of paid time off per calendar year [1]. "
        "Unused days can be cashed out at 50% of their value [1]."
    )
    monkeypatch.setattr(core, "get_provider", lambda name, settings: llm)
    r = assistant.ask("ana", "How many PTO days do full-time employees get per year?")
    assert "20 days" in r["answer"] and "cashed out" not in r["answer"]
    dropped = [v for v in r["trace"]["verification"] if not v["kept"]]
    assert dropped and dropped[0]["reason"] == "number_mismatch"
    assert "[1] Paid Time Off Policy" in llm.calls[0][1]  # sources are numbered in the prompt


def test_llm_hallucination_with_no_support_abstains(assistant, monkeypatch):
    monkeypatch.setattr(core, "get_provider", lambda n, s: FakeLLM("Employees get 30 days of PTO [1]."))
    assert assistant.ask("ana", "How many PTO days do full-time employees get per year?")["abstained"]
    monkeypatch.setattr(core, "get_provider", lambda n, s: FakeLLM("NOT_FOUND"))
    assert assistant.ask("ana", "What is the dress code?")["abstained"]


def test_llm_router_choice_is_validated(assistant, monkeypatch):
    object.__setattr__(assistant.settings, "llm_router", True)
    monkeypatch.setattr(core, "get_provider", lambda n, s: FakeLLM(route='{"intent": "org_members", "org": "finance"}'))
    assert assistant.ask("ana", "Who is on the money team?")["route"]["by"] == "model"
    # Invalid choices fall back to the rules.
    monkeypatch.setattr(core, "get_provider", lambda n, s: FakeLLM("x", route='{"intent": "drop_database"}'))
    assert assistant.ask("ana", "Who works in Finance?")["route"] == {
        "kind": "graph",
        "intent": "org_members",
        "params": {"org_id": "finance"},
        "reason": "Asks who belongs to a department",
        "by": "rules",
    }


def test_input_validation(assistant):
    with pytest.raises(ValueError):
        assistant.ask("ana", "   ")
    with pytest.raises(ValueError):
        assistant.ask("ana", "x", mode="yolo")
    with pytest.raises(NotFound):
        assistant.ask("mallory", "Hi")
