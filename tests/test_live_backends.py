"""Runs only in CI's `backends` job, against a real Weaviate container."""

import os

import pytest

from assistant.config import Settings
from assistant.core import Assistant
from assistant.evaluation import run_eval

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("WEAVIATE_URL"), reason="needs Weaviate"),
]


@pytest.fixture(scope="module")
def live() -> Assistant:
    return Assistant(Settings(vector_store="weaviate"))


def test_weaviate_filters_restricted_passages(live):
    assert live.ask("ana", "How quickly must on-call respond to a Sev1 security incident?")["abstained"]
    assert "20 days" in live.ask("ana", "How many PTO days do full-time employees get per year?")["answer"]


def test_full_evaluation_matches_in_memory(live):
    memory = run_eval(Assistant(Settings()))
    report = run_eval(live)
    assert report["grounded"]["grounding_errors"] == 0
    assert report["grounded"]["accuracy"] >= memory["grounded"]["accuracy"] - 0.05
