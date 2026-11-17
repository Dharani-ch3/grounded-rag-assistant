"""Runs only in CI's `backends` job, against real Neo4j and Weaviate containers."""

import os

import pytest

from assistant.config import Settings
from assistant.core import Assistant
from assistant.evaluation import run_eval

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.environ.get("NEO4J_URI") and os.environ.get("WEAVIATE_URL")), reason="needs Neo4j and Weaviate"
    ),
]


@pytest.fixture(scope="module")
def live() -> Assistant:
    return Assistant(Settings(vector_store="weaviate", graph_store="neo4j"))


def test_graph_questions_on_neo4j(live):
    r = live.ask("ana", "Who owns the expense reimbursement policy?")
    assert "Dev Patel" in r["answer"]
    assert live.ask("ana", "Who owns the security incident response runbook?")["abstained"]
    r = live.ask("ana", "Which documents can nurses read?")
    assert "Patient Intake Procedure" in r["answer"] and "(v2), Paid Time Off Policy (v2)" not in r["answer"]


def test_weaviate_filters_restricted_passages(live):
    assert live.ask("ana", "How quickly must on-call respond to a Sev1 security incident?")["abstained"]
    assert "20 days" in live.ask("ana", "How many PTO days do full-time employees get per year?")["answer"]


def test_full_evaluation_matches_in_memory(live):
    memory = run_eval(Assistant(Settings()))
    report = run_eval(live)
    assert report["grounded"]["grounding_errors"] == 0
    assert report["grounded"]["accuracy"] >= memory["grounded"]["accuracy"] - 0.05
