from assistant.grounding import Source, extractive_answer, verify

SRC = [
    Source(
        1,
        "c1",
        "d1",
        "Expense Reimbursement Policy",
        "Travel meals",
        "While traveling for work, meals are reimbursed up to $75 per day. Alcohol is not reimbursable.",
    ),
    Source(
        2,
        "c2",
        "d1",
        "Expense Reimbursement Policy",
        "Receipts",
        "Itemized receipts are required for any expense over $25.",
    ),
]


def test_supported_sentences_are_kept():
    v = verify("Meals are reimbursed up to $75 per day while traveling for work [1].", SRC)
    assert [x.reason for x in v.verdicts] == ["supported"] and v.cited == [1]


def test_wrong_numbers_are_dropped():
    v = verify("Meals are reimbursed up to $100 per day while traveling for work [1].", SRC)
    assert v.verdicts[0].reason == "number_mismatch" and not v.kept


def test_missing_or_unknown_citations_are_dropped():
    v = verify("Meals are reimbursed up to $75 per day. Receipts are needed over $25 [7].", SRC)
    assert [x.reason for x in v.verdicts] == ["no_citation", "unknown_source"]


def test_unsupported_content_is_dropped_but_supported_part_kept():
    v = verify(
        "Itemized receipts are required for any expense over $25 [2]. Managers can waive this for executives [2].", SRC
    )
    assert [x.reason for x in v.verdicts] == ["supported", "low_support"]


def test_extractive_answer_quotes_with_citation():
    answer, confidence = extractive_answer("What is the daily meal limit when traveling?", SRC, lambda t: 1.0)
    assert answer.startswith("While traveling for work, meals are reimbursed up to $75 per day. [1]")
    assert confidence == 1.0
