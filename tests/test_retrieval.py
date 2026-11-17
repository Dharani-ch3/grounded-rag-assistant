from assistant.retrieval import AccessFilter, HashingEmbedder


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.embed(["vaccine fridge temperature"]), e.embed(["vaccine fridge temperature"])
    assert a == b and abs(sum(x * x for x in a[0]) - 1) < 1e-9


def test_filter_is_applied_before_ranking(assistant):
    r = assistant.retriever
    nurse = AccessFilter(("employee", "nurse"))
    hits = r.search("Sev1 security incident on-call response", nurse, k=10)
    assert all(h.doc_id != "incident-response-runbook" for h in hits)
    naive = r.search("Sev1 security incident on-call response", AccessFilter(None), k=10)
    assert naive[0].doc_id == "incident-response-runbook"


def test_archived_versions_never_returned_when_guarded(assistant):
    hits = assistant.retriever.search("PTO days per year carry over", AccessFilter(("employee",)), k=20)
    assert hits and {h.doc_id for h in hits}.isdisjoint({"pto-policy-v2", "expense-policy-v1"})


def test_approval_updates_index_metadata(assistant):
    flt = AccessFilter(("employee",))
    before = assistant.retriever.search("remote days per week", flt)
    assert all(h.doc_id != "remote-work-policy-draft" for h in before)
    assistant.approve("omar", "remote-work-policy-draft")
    after = assistant.retriever.search("remote days per week", flt)
    assert after[0].doc_id == "remote-work-policy-draft"
