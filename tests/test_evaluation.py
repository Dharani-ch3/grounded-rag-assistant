from pathlib import Path

from assistant.evaluation import load_cases, run_eval, to_markdown

KNOWLEDGE = Path(__file__).resolve().parent.parent / "knowledge"


def test_tuning_set(assistant):
    r = run_eval(assistant)
    assert r["grounded"]["grounding_errors"] == 0
    assert r["grounded"]["accuracy"] >= 0.95
    assert r["naive"]["grounding_errors"] >= 10  # the traps really are traps
    assert "Naive RAG" in to_markdown(r)


def test_holdout_set_has_no_grounding_errors(assistant):
    r = run_eval(assistant, cases=load_cases(KNOWLEDGE / "eval_holdout.json"))
    assert r["grounded"]["grounding_errors"] == 0
    assert r["grounded"]["accuracy"] > r["naive"]["accuracy"]


def test_judge_classifies_errors(assistant):
    cases = [
        {
            "id": 1,
            "user": "ana",
            "question": "How quickly must on-call respond to a Sev1 security incident?",
            "abstain": True,
        }
    ]
    r = run_eval(assistant, cases=cases)
    assert r["rows"][0]["naive"]["outcome"] == "leak"
    assert r["rows"][0]["grounded"]["outcome"] == "correct_abstain"
