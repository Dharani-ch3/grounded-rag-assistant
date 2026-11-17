"""Text utilities shared by retrieval, answering and grounding checks."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from had has have he her his i if in into
    is it its me my no not of on or our she so than that the their them then there these they this to
    too up us was we were what when where which who whom will with would you your just also very really
    how many much should get per any am i'm im about tell please need know there's what's each every""".split()
)

_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_NUMBER = re.compile(r"\$?\d+(?:[.,]\d+)*%?")
# A sentence ends at . ! or ? followed by a capital, digit, quote or $; a following "[n]"
# citation stays attached to the sentence before it, and the next sentence starts after it.
_SENTENCE = re.compile(r"(?:(?<=[.!?])|(?<=\]))\s+(?=[A-Z0-9\"'($])")
_CITE = re.compile(r"\s*\[(\d+)\]")

# A few domain synonyms so short questions still reach the right passages
# ("vacation" -> PTO). Real deployments get this from dense embeddings.
SYNONYMS = {
    "vacation": "pto",
    "daily": "day",
    "reimburse": "reimburs",
    "reimbursed": "reimburs",
    "reimbursement": "reimburs",
    "reimbursable": "reimburs",
    "2fa": "multi-factor",
    "mfa": "multi-factor",
    "fridge": "refrigerator",
    "temp": "temperature",
    "sev1": "sev1",
    "keep": "retain",
    "kept": "retain",
    "retained": "retain",
    "retention": "retain",
}


def stem(word: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ies", "edly", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if w in STOPWORDS or len(w) < 2:
            continue
        w = SYNONYMS.get(w, w)
        out.append(stem(w))
    return out


# Words that frame a question ("what is the meal *limit*", "how *long*") but rarely appear
# in the passage that answers it. They still drive retrieval; they don't count against coverage.
FRAME_WORDS = frozenset(
    stem(w)
    for w in "limit rate range long often quick quickly fast amount maximum minimum number allowed rule rules required requirement".split()
)


def content_terms(question: str) -> list[str]:
    """Question terms that an answering passage should contain (no frame words, no bare numbers)."""
    return [t for t in tokenize(question) if t not in FRAME_WORDS and not t.replace(".", "").isdigit()]


def numbers(text: str) -> set[str]:
    """Numbers, money and percentages, normalised so "$75" and "75" compare equal."""
    return {n.strip("$%").replace(",", "") for n in _NUMBER.findall(text)}


def split_sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n|\n(?=#)", text):
        para = " ".join(line.strip() for line in para.splitlines() if not line.strip().startswith("#")).strip()
        if para:
            out.extend(s.strip() for s in _SENTENCE.split(para) if s.strip())
    return out


def strip_citations(sentence: str) -> tuple[str, list[int]]:
    cites = [int(n) for n in _CITE.findall(sentence)]
    return _CITE.sub("", sentence).strip(), cites


def chunk_markdown(body: str, max_words: int = 90) -> list[tuple[str, str]]:
    """Split a markdown body into (section heading, text) chunks.

    Chunks follow `##` sections so a passage never mixes two topics. Long sections are
    split on sentence boundaries. The heading is kept so the chunk reads on its own.
    """
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in body.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif line.startswith("# "):
            continue
        else:
            sections[-1][1].append(line)
    chunks: list[tuple[str, str]] = []
    for heading, lines in sections:
        text = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if not text:
            continue
        current: list[str] = []
        count = 0
        for sentence in _SENTENCE.split(text):
            n = len(sentence.split())
            if current and count + n > max_words:
                chunks.append((heading, " ".join(current)))
                current, count = [], 0
            current.append(sentence)
            count += n
        if current:
            chunks.append((heading, " ".join(current)))
    return chunks
