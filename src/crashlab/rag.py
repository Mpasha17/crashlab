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

import re
from collections.abc import Iterable
from typing import Any

from .models import Run, Step

__all__ = [
    "abstains_without_context",
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
    "cannot answer",
    "can't answer",
)


def _docs_of(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        for key in _WRAPPERS:
            if isinstance(result.get(key), list):
                result = result[key]
                break
        else:
            return []
    if not isinstance(result, list):
        return []
    return [d for d in result if isinstance(d, dict)]


def retrieved(
    run: Run, tool: str = "search_docs", *, id_key: str = "id"
) -> list[tuple[Step, dict[str, Any]]]:
    """``(step, document)`` for every document returned by successful ``tool`` calls."""
    return [(s, d) for s in run.tool_calls(tool, ok_only=True) for d in _docs_of(s.result)]


def _ids(docs: Iterable[tuple[Step, dict[str, Any]]], id_key: str) -> list[str]:
    out: list[str] = []
    for _, d in docs:
        if d.get(id_key) is not None and str(d[id_key]) not in out:
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
    docs = retrieved(run, tool, id_key=id_key)
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
