"""Answer drafting and the grounding check that every answer must pass.

`extractive_answer` is the offline answerer: it quotes the sentences that best answer the
question, with no model involved. LLM answers are written by a model instead. Either way the
draft then goes through `verify`, which checks every sentence against the passages it cites:

* a sentence with no citation, or citing a source that wasn't provided, is dropped;
* a sentence whose content words aren't mostly in the cited passage is dropped;
* a sentence with a number, amount or percentage that isn't in the cited passage is dropped.
  (Wrong numbers are the most common and most harmful hallucination in policy answers.)

If nothing survives, the assistant abstains instead of answering.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .text import content_terms, numbers, split_sentences, strip_citations, tokenize


@dataclass
class Source:
    n: int  # citation number shown to the model and the user
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    text: str


@dataclass
class Verdict:
    sentence: str
    cites: list[int]
    kept: bool
    reason: str  # supported | no_citation | unknown_source | low_support | number_mismatch
    support: float = 0.0


@dataclass
class Verification:
    answer: str
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def kept(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.kept]

    @property
    def dropped(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.kept]

    @property
    def cited(self) -> list[int]:
        seen: list[int] = []
        for v in self.kept:
            for c in v.cites:
                if c not in seen:
                    seen.append(c)
        return seen


def coverage(question_tokens: list[str], text_tokens: set[str], idf: Callable[[str], float]) -> float:
    """IDF-weighted share of the question's terms that appear in the text."""
    q = set(question_tokens)
    if not q:
        return 0.0
    total = sum(idf(t) for t in q)
    return sum(idf(t) for t in q if t in text_tokens) / total if total else 0.0


def extractive_answer(
    question: str, sources: list[Source], idf: Callable[[str], float], max_sentences: int = 2
) -> tuple[str, float]:
    """Quote the best-matching sentence(s), each with its citation. Returns (answer, confidence).

    Confidence is the question-term coverage of the best sentence plus its section heading,
    so "Carryover: Employees may carry over up to 5 days" can answer "how many days carry over?".
    """
    q = content_terms(question)
    scored: list[tuple[float, int, int, str, Source]] = []
    for rank, src in enumerate(sources):
        head = set(tokenize(src.heading + " " + src.title))
        for i, sentence in enumerate(split_sentences(src.text)):
            toks = set(tokenize(sentence))
            cov = coverage(q, toks | head, idf)
            own = coverage(q, toks, idf)
            # Rank bonus is small: retrieval order breaks ties, sentence content decides.
            score = 0.75 * cov + 0.25 * own - 0.02 * rank
            scored.append((score, rank, i, sentence, src))
    if not scored:
        return "", 0.0
    scored.sort(key=lambda x: -x[0])
    best = scored[0]
    chosen = [best]
    for cand in scored[1:]:
        if len(chosen) >= max_sentences:
            break
        # A follow-up sentence must come from the same passage, right after the best one,
        # and still be about the question.
        if cand[4] is best[4] and cand[2] == best[2] + 1 and cand[0] >= 0.6 * best[0] and cand[0] > 0.25:
            chosen.append(cand)
    chosen.sort(key=lambda x: (x[1], x[2]))
    answer = " ".join(f"{s} [{src.n}]" for _, _, _, s, src in chosen)
    head = set(tokenize(best[4].heading + " " + best[4].title))
    confidence = coverage(q, set(tokenize(best[3])) | head, idf)
    return answer, round(confidence, 3)


def verify(answer: str, sources: list[Source], min_support: float = 0.6) -> Verification:
    by_n = {s.n: s for s in sources}
    result = Verification(answer=answer)
    for raw in split_sentences(answer):
        sentence, cites = strip_citations(raw)
        if not sentence:
            continue
        if not cites:
            result.verdicts.append(Verdict(sentence, cites, False, "no_citation"))
            continue
        if any(c not in by_n for c in cites):
            result.verdicts.append(Verdict(sentence, cites, False, "unknown_source"))
            continue
        cited_text = " ".join(by_n[c].heading + ". " + by_n[c].text for c in cites)
        s_toks = set(tokenize(sentence))
        support = len(s_toks & set(tokenize(cited_text))) / len(s_toks) if s_toks else 0.0
        if numbers(sentence) - numbers(cited_text):
            result.verdicts.append(Verdict(sentence, cites, False, "number_mismatch", round(support, 3)))
        elif support < min_support:
            result.verdicts.append(Verdict(sentence, cites, False, "low_support", round(support, 3)))
        else:
            result.verdicts.append(Verdict(sentence, cites, True, "supported", round(support, 3)))
    return result
