"""Oracle helpers for retrieval-augmented (RAG) agents.

Decorate your retriever with ``@crashlab.tool()`` so its results are recorded (and replayed
offline: no vector store is contacted on replay). These helpers read those recorded
results from a :class:`~crashlab.models.Run` and return a failure message (``str``) or
``None``, so they can be returned straight from a ``@crashlab.check`` oracle::

    from crashlab import rag

    @crashlab.check("support-rag")
    def citations_ok(run):
        return rag.citations_were_retrieved(run, run.output)

They are deterministic string checks, not semantic judges: ``answer_grounded`` only
verifies that numbers in the answer appear in a retrieved document.

A retrieval result may be a list of documents (dicts) or a dict wrapping that list under
``documents`` / ``results`` / ``docs`` / ``hits`` / ``matches``. Documents are dicts; the id
and text keys are configurable (``id_key`` / ``text_key``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from .models import Run, Step

__all__ = [
    "abstains_without_context",
    "claims_supported",
    "answer_grounded",
    "cited_ids",
    "citations_were_retrieved",
    "no_stale_docs",
    "retrieved",
    "retrieved_ids",
]

CITATION = r"\[([^\[\]\s]+)\]"
_WRAPPERS = ("documents", "results", "docs", "hits", "matches")
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
ABSTAIN = (
    "i don't know",
    "i do not know",
    "not sure",
    "could not find",
    "couldn't find",
    "no information",
    "no documents",
    "not in the documents",
    "unable to find",
    "could not answer",
    "couldn't answer",
    "cannot answer",
    "can't answer",
)


_TEXT_FALLBACKS = ("text", "page_content", "content", "chunk", "document", "snippet")
_ID_FALLBACKS = ("id", "doc_id", "document_id", "source")


def _scalar(v: Any) -> bool:
    return isinstance(v, (str, int)) and not isinstance(v, bool)


def _docs_of(result: Any, id_key: str, text_key: str) -> list[dict[str, Any]]:
    if isinstance(result, str) and result.lstrip()[:1] in ("[", "{"):
        try:
            result = json.loads(result)
        except ValueError:
            return []
    if isinstance(result, dict):
        for key in _WRAPPERS:
            if isinstance(result.get(key), list):
                result = result[key]
                break
        else:
            return []
    if not isinstance(result, list):
        return []
    docs: list[dict[str, Any]] = []
    for item in result:
        if isinstance(item, str):
            item = {text_key: item}
        if not isinstance(item, dict):
            continue
        doc = dict(item)
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        if not _scalar(doc.get(id_key)):
            found = [m.get(k) for m in (doc, meta) for k in (id_key, *_ID_FALLBACKS)]
            doc[id_key] = next((v for v in found if _scalar(v)), None)
        if not isinstance(doc.get(text_key), str):
            texts = [doc.get(k) for k in (text_key, *_TEXT_FALLBACKS)]
            doc[text_key] = next((t for t in texts if isinstance(t, str)), "")
        docs.append(doc)
    return docs


def retrieved(
    run: Run, tool: str = "search_docs", *, id_key: str = "id", text_key: str = "text"
) -> list[tuple[Step, dict[str, Any]]]:
    """``(step, document)`` for every document returned by successful ``tool`` calls.

    Documents are copies normalized so ``doc[id_key]`` is a str/int or ``None`` (falling
    back to ``doc_id`` / ``document_id`` / ``source`` and the same keys under
    ``metadata``) and ``doc[text_key]`` is a str (falling back to ``page_content`` /
    ``content`` / ``chunk`` / ``document`` / ``snippet``). Plain strings become
    ``{text_key: s}``; JSON text results are parsed; anything else is skipped."""
    return [
        (s, d)
        for s in run.tool_calls(tool, ok_only=True)
        for d in _docs_of(s.result, id_key, text_key)
    ]


def _ids(docs: Iterable[tuple[Step, dict[str, Any]]], id_key: str) -> list[str]:
    out: list[str] = []
    for _, d in docs:
        if _scalar(d.get(id_key)) and str(d[id_key]) not in out:
            out.append(str(d[id_key]))
    return out


def _where(run: Run, tool: str) -> str:
    steps = [s.idx for s in run.tool_calls(tool)]
    return f"step {steps[-1]}" if steps else f"no {tool} call"


def _answer_step(run: Run) -> str:
    chats = run.chat_steps()
    return f"step {chats[-1].idx}" if chats else "final answer"


def cited_ids(text: str | None, pattern: str = CITATION) -> list[str]:
    """Unique citation ids in ``text`` (default ``[doc-id]``), in order."""
    out: list[str] = []
    for m in re.finditer(pattern, text or ""):
        if m.group(1) not in out:
            out.append(m.group(1))
    return out


def retrieved_ids(
    run: Run, expected: Iterable[str], *, tool: str = "search_docs", id_key: str = "id"
) -> str | None:
    """Wrong retrieval: every id in ``expected`` (your gold documents) was retrieved."""
    got = _ids(retrieved(run, tool, id_key=id_key), id_key)
    missing = [e for e in expected if e not in got]
    if missing:
        return f"{_where(run, tool)}: {tool} returned {got}, expected {missing}"
    return None


def citations_were_retrieved(
    run: Run,
    answer: str | None,
    *,
    tool: str = "search_docs",
    id_key: str = "id",
    pattern: str = CITATION,
    require: bool = True,
) -> str | None:
    """Citation mismatch: every cited id was actually retrieved (and, with ``require``,
    the answer cites something whenever documents were retrieved)."""
    got = _ids(retrieved(run, tool, id_key=id_key), id_key)
    cites = cited_ids(answer, pattern)
    bad = [c for c in cites if c not in got]
    if bad:
        return f"{_answer_step(run)}: answer cites {bad}, but {tool} returned {got}"
    if require and got and not cites and not _abstained(answer):
        return f"{_answer_step(run)}: answer has no citation; {tool} returned {got}"
    return None


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text or "")}


def answer_grounded(
    run: Run,
    answer: str | None,
    *,
    tool: str = "search_docs",
    id_key: str = "id",
    text_key: str = "text",
    pattern: str = CITATION,
) -> str | None:
    """Unsupported answer: every number in the answer (citations removed) appears in a
    retrieved document's ``text_key`` (in a cited one when the answer cites any)."""
    docs = retrieved(run, tool, id_key=id_key, text_key=text_key)
    cites = cited_ids(answer, pattern)
    if cites:
        docs = [(s, d) for s, d in docs if str(d.get(id_key)) in cites] or docs
    evidence: set[str] = set()
    for _, d in docs:
        evidence |= _numbers(str(d.get(text_key, "")))
    claimed = _numbers(re.sub(pattern, " ", answer or ""))
    unsupported = sorted(claimed - evidence)
    if unsupported:
        return (
            f"{_answer_step(run)}: answer states {unsupported}, not found in the retrieved "
            f"documents {_ids(docs, id_key)} ({_where(run, tool)})"
        )
    return None


def no_stale_docs(
    run: Run,
    *,
    tool: str = "search_docs",
    id_key: str = "id",
    superseded_key: str = "superseded_by",
    date_key: str | None = None,
    not_before: str | None = None,
) -> str | None:
    """Stale documents: no retrieved document is marked superseded (``superseded_key`` set)
    or, with ``date_key`` + ``not_before`` (ISO dates compare as strings), older than that."""
    stale: list[str] = []
    for s, d in retrieved(run, tool, id_key=id_key):
        why = None
        if d.get(superseded_key):
            why = f"superseded by {d[superseded_key]}"
        elif date_key and not_before and str(d.get(date_key, "")) < not_before:
            why = f"{date_key}={d.get(date_key)!r} < {not_before}"
        if why:
            stale.append(f"step {s.idx}: {d.get(id_key)} ({why})")
    return f"{tool} returned stale documents: " + "; ".join(stale) if stale else None


def _abstained(answer: str | None, phrases: Iterable[str] = ABSTAIN) -> bool:
    low = (answer or "").lower().replace("\u2019", "'")
    return any(p in low for p in phrases)


def abstains_without_context(
    run: Run,
    answer: str | None,
    *,
    tool: str = "search_docs",
    id_key: str = "id",
    phrases: Iterable[str] = ABSTAIN,
) -> str | None:
    """Missing context: when retrieval returned nothing, the answer says it does not know
    (contains one of ``phrases``) instead of answering anyway."""
    empty = run.tool_calls(tool) and not retrieved(run, tool, id_key=id_key)
    if empty and not _abstained(answer, phrases):
        return (
            f"{_where(run, tool)}: {tool} returned no documents, but the answer "
            f"({_answer_step(run)}) did not abstain: {(answer or '')[:120]!r}"
        )
    return None


# --- word-overlap grounding -------------------------------------------------------------

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "within",
        "into",
        "onto",
        "over",
        "under",
        "above",
        "below",
        "after",
        "before",
        "about",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "having",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "it's",
        "they",
        "them",
        "their",
        "there",
        "here",
        "you",
        "your",
        "yours",
        "we",
        "our",
        "us",
        "i",
        "me",
        "my",
        "he",
        "she",
        "his",
        "her",
        "him",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "each",
        "every",
        "some",
        "such",
        "only",
        "also",
        "just",
        "than",
        "too",
        "very",
        "more",
        "most",
        "get",
        "gets",
        "got",
        "instead",
        "still",
        "yes",
        "ok",
        "okay",
        "please",
    ]
)
NEGATIONS = frozenset(
    [
        "not",
        "no",
        "never",
        "none",
        "nor",
        "cannot",
        "without",
        "neither",
        "nothing",
        "nobody",
        "nowhere",
    ]
)
_WORD = re.compile(r"[a-z][a-z']*")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _stem(w: str) -> str:
    w = w.removesuffix("'s")
    for suf in ("ing", "ed", "ies", "es", "s", "y", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _negated(text: str) -> bool:
    words = _WORD.findall(text.lower().replace("\u2019", "'"))
    return any(w in NEGATIONS or w.endswith("n't") for w in words)


def _content(text: str, stopwords: frozenset[str]) -> dict[str, str]:
    """stem -> first surface word, for content words (stopwords/negations dropped)."""
    out: dict[str, str] = {}
    for w in _WORD.findall(text.lower().replace("\u2019", "'")):
        if w in stopwords or w in NEGATIONS or w.endswith("n't") or len(w) < 3:
            continue
        out.setdefault(_stem(w), w)
    return out


def claims_supported(
    run: Run,
    answer: str | None,
    *,
    tool: str = "search_docs",
    id_key: str = "id",
    text_key: str = "text",
    pattern: str = CITATION,
    min_overlap: float = 0.5,
    min_words: int = 3,
    stopwords: frozenset[str] = STOPWORDS,
) -> str | None:
    """Unsupported (non-numeric) claims, by word overlap.

    Each answer sentence with at least ``min_words`` content words is compared with the
    documents it cites (the answer's citations if the sentence has none; all retrieved
    documents if nothing cited was retrieved). It fails when fewer than ``min_overlap`` of
    its content words (lower-cased, stopwords removed, light suffix stemming) occur in
    those documents, or when it matches document sentences only with the opposite
    negation (``not`` / ``no`` / ``never`` / ``n't`` ...). Abstentions pass; with no
    retrieved documents it defers to ``abstains_without_context``.

    A heuristic, not entailment: paraphrases with different vocabulary can fail (lower
    ``min_overlap``) and wrong claims reusing document words can pass."""
    if _abstained(answer):
        return None
    docs = retrieved(run, tool, id_key=id_key, text_key=text_key)
    if not docs:
        return None
    all_cites = cited_ids(answer, pattern)
    for sentence in _SENT.split(answer or ""):
        cites = cited_ids(sentence, pattern) or all_cites
        words = _content(re.sub(pattern, " ", sentence), stopwords)
        if len(words) < min_words:
            continue
        used = [d for _, d in docs if str(d.get(id_key)) in cites] or [d for _, d in docs]
        doc_sents = [ds for d in used for ds in _SENT.split(str(d.get(text_key, ""))) if ds]
        vocab: set[str] = set()
        for ds in doc_sents:
            vocab |= set(_content(ds, stopwords))
        where = f"[{'], ['.join(cites)}]" if cites else "the retrieved documents"
        claim = re.sub(pattern, "", sentence).strip()
        missing = [surface for st, surface in words.items() if st not in vocab]
        if (len(words) - len(missing)) / len(words) < min_overlap:
            return (
                f"{_answer_step(run)}: claim {claim!r} is not supported by {where} "
                f"({_where(run, tool)}): words {missing} not found"
            )
        neg = _negated(claim)

        def overlap(ds: str, w: dict[str, str] = words) -> float:
            return len(set(w) & set(_content(ds, stopwords))) / len(w)

        same = [ds for ds in doc_sents if _negated(ds) == neg and overlap(ds) >= min_overlap]
        flipped = [ds for ds in doc_sents if _negated(ds) != neg and overlap(ds) >= min_overlap]
        if flipped and not same:
            return (
                f"{_answer_step(run)}: claim {claim!r} contradicts {where} "
                f"({_where(run, tool)}): negation mismatch with {flipped[0].strip()!r}"
            )
    return None
