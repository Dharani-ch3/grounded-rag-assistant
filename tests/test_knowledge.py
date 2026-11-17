import pytest

from assistant.knowledge import KnowledgeError, document_from_markdown
from assistant.text import chunk_markdown, content_terms, numbers, split_sentences, strip_citations, tokenize


def test_front_matter_and_validation():
    doc = document_from_markdown(
        "---\nid: x\ntitle: X\nstatus: draft\nowner: ana\norg: hr\naudience: nurse, employee\n---\n## A\nText."
    )
    assert doc.audience == ["nurse", "employee"] and doc.status == "draft" and doc.version == 1
    with pytest.raises(KnowledgeError):
        document_from_markdown("no front matter")
    with pytest.raises(KnowledgeError):
        document_from_markdown("---\nid: x\ntitle: X\nstatus: live\nowner: a\norg: b\naudience: c\n---\nbody")


def test_chunks_follow_sections():
    chunks = chunk_markdown("# Title\n\n## One\nFirst sentence. Second.\n\n## Two\nOther topic.")
    assert chunks == [("One", "First sentence. Second."), ("Two", "Other topic.")]


def test_text_helpers():
    assert numbers("Up to $75 per day, 30% or 0.65") == {"75", "30", "0.65"}
    assert split_sentences("One is here [1]. Two [2] Three.") == ["One is here [1].", "Two [2]", "Three."]
    assert strip_citations("Meals up to $75 [2] [3]") == ("Meals up to $75", [2, 3])
    assert "pto" in tokenize("How much vacation?")  # synonym
    assert content_terms("What is the daily meal limit?") == ["day", "meal"]  # frame word "limit" dropped


def test_access_is_checked_before_lifecycle(assistant):
    kb = assistant.kb
    ben, ana, carla = (kb.directory.users[u] for u in ("ben", "ana", "carla"))
    assert kb.decide(ana, kb.docs["incident-response-runbook"]).reason == "no_access"
    assert kb.decide(ben, kb.docs["remote-work-policy-draft"]).reason == "draft"
    assert kb.decide(ben, kb.docs["pto-policy-v2"]).reason == "archived"
    assert kb.decide(ana, kb.docs["pto-policy-v3"]).allowed
    # Admins see drafts they can read; others don't see drafts at all.
    assert "remote-work-policy-draft" in {d.id for d in kb.visible_docs(carla)}
    assert "remote-work-policy-draft" not in {d.id for d in kb.visible_docs(ben)}
    assert "exec-compensation-guidelines" not in {d.id for d in kb.visible_docs(ben)}


def test_superseded_versions_are_linked(assistant):
    assert assistant.kb.docs["pto-policy-v2"].superseded_by == "pto-policy-v3"
    assert assistant.kb.docs["expense-policy-v1"].superseded_by == "expense-policy-v2"
