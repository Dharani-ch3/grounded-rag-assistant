import pytest

from assistant.router import route


@pytest.mark.parametrize(
    "question,kind,intent",
    [
        ("Who owns the expense reimbursement policy?", "graph", "doc_owner"),
        ("Who approved the paid time off policy?", "graph", "doc_approver"),
        ("Who should I contact about vaccine refrigerator problems?", "both", "topic_contact"),
        ("Who do I report to?", "graph", "user_manager"),
        ("Who does Ben report to?", "graph", "user_manager"),
        ("Which documents can nurses read?", "graph", "role_documents"),
        ("Who works in Finance?", "graph", "org_members"),
        ("Which documents does Carla own?", "graph", "user_documents"),
        ("How many PTO days do I get?", "documents", None),
    ],
)
def test_routes(assistant, question, kind, intent):
    kb = assistant.kb
    ana = kb.directory.users["ana"]
    allowed = [d.id for d in kb.docs.values() if kb.decide(ana, d).allowed]
    r = route(question, ana, kb, allowed)
    assert (r.kind, r.intent) == (kind, intent)


def test_self_reference_resolves_to_asker(assistant):
    kb = assistant.kb
    r = route("Who is my manager?", kb.directory.users["dev"], kb, [])
    assert r.params == {"user_id": "dev", "is_self": True}


def test_document_names_only_match_allowed_documents(assistant):
    kb = assistant.kb
    ana = kb.directory.users["ana"]
    allowed = [d.id for d in kb.docs.values() if kb.decide(ana, d).allowed]
    r = route("Who owns the incident response runbook?", ana, kb, allowed)
    assert r.kind == "both" and "doc_id" not in r.params  # not resolved to a document Ana can't read
