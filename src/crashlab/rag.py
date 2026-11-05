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
import unicodedata
from collections.abc import Iterable
from typing import Any

from .models import Run, Step

__all__ = [
    "abstains_without_context",
    "claims_supported",
    "DUPLICATES",
    "answer_grounded",
    "cited_ids",
    "citations_were_retrieved",
    "no_stale_docs",
    "retrieved",
    "retrieved_ids",
]

CITATION = r"\[([^\[\]]+)\](?!\()|\u3010([^\u3010\u3011]+)\u3011"
_WRAPPERS = ("documents", "results", "docs", "hits", "matches")
_NUM = re.compile(r"(?<![A-Za-z0-9.])\d+(?:[.,]\d+)*(?![A-Za-z0-9])")
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
    "don't have that information",
    "do not have that information",
    "don't have enough information",
    "do not have enough information",
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
    """Unique citation ids in ``text``, in order. Default form ``[doc-id]``; ``[a, b]`` /
    ``[a; b]`` cite several; bracketed text with spaces and ``[text](url)`` links are not
    citations."""
    out: list[str] = []
    for m in re.finditer(pattern, text if isinstance(text, str) else ""):
        body = next((g for g in m.groups() if g is not None), m.group(0))
        for part in re.split(r"[,;]", body):
            # 'doc†L1' (line anchor) / '4†doc' (cursor): keep the segment that is an id
            segs = [g.strip() for g in part.split("\u2020")]
            if len(segs) > 1:  # plain '[1]' footnotes stay ids
                part = next((g for g in segs if g and not _ANCHOR.match(g)), "")
            else:
                part = segs[0]
            if part and not re.search(r"\s", part) and part not in out:
                out.append(part)
    return out


_ANCHOR = re.compile(r"^(?:\d+|L\d+(?:-L?\d+)?)$")
_SUFFIX = re.compile(r"(?:#|:)(?:L?\d+(?:-L?\d+)?|page=\d+)$")


def _resolve(cites: list[str], got: Iterable[str]) -> list[str]:
    """Map citations with a line/page suffix (``doc.md#L10``, ``doc.md:3-5``,
    ``doc.md#page=2``) to the retrieved id they point at; exact ids win."""
    known = set(got)
    out: list[str] = []
    for c in cites:
        base = _SUFFIX.sub("", c)
        c = c if c in known or base not in known else base
        if c not in out:
            out.append(c)
    return out


def retrieved_ids(
    run: Run,
    expected: Iterable[str],
    *,
    tool: str = "search_docs",
    id_key: str = "id",
    top_k: int | None = None,
) -> str | None:
    """Wrong retrieval: every id in ``expected`` (your gold documents) was retrieved; with
    ``top_k``, within the first ``top_k`` documents of some ``tool`` call (the recorded
    order is the ranking; score fields are not read)."""
    docs = retrieved(run, tool, id_key=id_key)
    if top_k is not None:
        rank: dict[int, int] = {}
        kept = []
        for s, d in docs:
            rank[id(s)] = rank.get(id(s), 0) + 1
            if rank[id(s)] <= top_k:
                kept.append((s, d))
        docs = kept
    got = _ids(docs, id_key)
    missing = [str(e) for e in expected if str(e) not in got]
    if missing:
        top = f" (top {top_k})" if top_k is not None else ""
        return f"{_where(run, tool)}: {tool} returned {got}{top}, expected {missing}"
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
    cites = _resolve(cited_ids(answer, pattern), got)
    bad = [c for c in cites if c not in got]
    if bad:
        return f"{_answer_step(run)}: answer cites {bad}, but {tool} returned {got}"
    if require and got and not cites and not _abstained(answer):
        return f"{_answer_step(run)}: answer has no citation; {tool} returned {got}"
    return None


def _numbers(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text) if isinstance(text, str) else ""
    return {n.replace(",", "") for n in _NUM.findall(text)}


DUPLICATES = ("versions", "latest", "first", "any", "strict")
VERSION_KEYS = ("version", "updated", "updated_at", "last_modified", "modified", "date")


def _version(d: dict[str, Any], superseded_key: str) -> tuple[int, Any]:
    meta = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
    value = next(
        (src[k] for k in VERSION_KEYS for src in (d, meta) if src.get(k) not in (None, "")),
        None,
    )
    current = 0 if (d.get(superseded_key) or meta.get(superseded_key)) else 1
    return current, value


def _newer(a: Any, b: Any) -> bool:
    """a > b for version values (numbers numerically, else as strings; None is oldest)."""
    if b is None:
        return a is not None
    if a is None:
        return False
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool):
        return a > b
    return str(a) > str(b)


def _select(
    docs: list[dict[str, Any]],
    id_key: str,
    text_key: str,
    duplicates: str,
    superseded_key: str = "superseded_by",
) -> tuple[list[dict[str, Any]], str | None]:
    """Resolve documents sharing an id but differing in text (see ``duplicates``)."""
    if duplicates not in DUPLICATES:
        raise ValueError(f"duplicates must be one of {DUPLICATES}, not {duplicates!r}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for d in docs:
        if _scalar(d.get(id_key)):
            groups.setdefault(str(d[id_key]), []).append(d)
    drop: set[int] = set()
    for key, group in groups.items():
        texts = list(dict.fromkeys(str(d.get(text_key, "")) for d in group))
        if len(texts) < 2 or duplicates == "any":
            continue
        versions = [_version(d, superseded_key) for d in group]
        versioned = any(c == 0 for c, _ in versions) or len({repr(v) for _, v in versions}) > 1
        if duplicates == "strict" or (duplicates == "versions" and versioned):
            what = f"{len(texts)} versions" if versioned else f"{len(texts)} different texts"
            return docs, (
                f"id {key!r} was retrieved with {what}; the evidence is ambiguous "
                f"(pass duplicates='latest', 'first' or 'any' to choose)"
            )
        if duplicates == "versions":  # unversioned chunks of one document: keep all
            continue
        keep = group[0]
        if duplicates == "latest":
            for d, (cur, ver) in zip(group, versions, strict=True):
                kc, kv = _version(keep, superseded_key)
                if cur > kc or (cur == kc and _newer(ver, kv)):
                    keep = d
        drop |= {id(d) for d in group if str(d.get(text_key, "")) != str(keep.get(text_key, ""))}
    return [d for d in docs if id(d) not in drop], None


def answer_grounded(
    run: Run,
    answer: str | None,
    *,
    tool: str = "search_docs",
    id_key: str = "id",
    text_key: str = "text",
    pattern: str = CITATION,
    duplicates: str = "versions",
    exclude_stale: bool = False,
    superseded_key: str = "superseded_by",
    date_key: str | None = None,
    not_before: str | None = None,
) -> str | None:
    """Unsupported answer: every number in the answer (citations removed) appears in a
    retrieved document's ``text_key`` (in a cited one when the answer cites any).

    ``duplicates`` decides what happens when the evidence holds one id with different
    texts: ``"versions"`` (default) fails as ambiguous when they are versions (differing
    ``version`` / ``updated`` / ... fields, or one has ``superseded_by``) and merges
    unversioned chunks; ``"latest"`` keeps the current/newest version; ``"first"`` the
    first retrieved; ``"any"`` merges all; ``"strict"`` fails on any differing text.

    By default an uncited number is grounded by any retrieved document, stale ones
    included. ``exclude_stale=True`` drops stale documents (``superseded_key`` set, or
    ``date_key`` older than ``not_before``, as in :func:`no_stale_docs`) from the evidence,
    so a number only a stale document supports fails, cited or not."""
    _select([], id_key, text_key, duplicates)  # validate early
    found = retrieved(run, tool, id_key=id_key, text_key=text_key)
    cites = _resolve(cited_ids(answer, pattern), _ids(found, id_key))
    if cites:
        found = [(s, d) for s, d in found if str(d.get(id_key)) in cites] or found
    claimed = _numbers(re.sub(pattern, " ", answer if isinstance(answer, str) else ""))
    if not claimed:
        return None
    kept, conflict = _select([d for _, d in found], id_key, text_key, duplicates)
    if conflict:
        return f"{_answer_step(run)}: {conflict} ({_where(run, tool)})"
    docs = [(s, d) for s, d in found if any(d is k for k in kept)]
    ignored: list[str] = []
    if exclude_stale:
        stale = [(s, d) for s, d in docs if _stale_why(d, superseded_key, date_key, not_before)]
        ignored = _ids(stale, id_key)
        docs = [(s, d) for s, d in docs if not any(d is x for _, x in stale)]
    evidence: set[str] = set()
    for _, d in docs:
        evidence |= _numbers(str(d.get(text_key, "")))
    unsupported = sorted(claimed - evidence)
    if unsupported:
        return (
            f"{_answer_step(run)}: answer states {unsupported}, not found in the retrieved "
            f"documents {_ids(docs, id_key)} ({_where(run, tool)})"
            + (f"; stale documents ignored: {ignored}" if ignored else "")
        )
    return None


def _stale_why(
    d: dict[str, Any], superseded_key: str, date_key: str | None, not_before: str | None
) -> str | None:
    if d.get(superseded_key):
        return f"superseded by {d[superseded_key]}"
    if date_key and not_before and str(d.get(date_key, "")) < not_before:
        return f"{date_key}={d.get(date_key)!r} < {not_before}"
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
        why = _stale_why(d, superseded_key, date_key, not_before)
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
_WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|')*")
_SENT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_INTERJECTION = re.compile(r"^\s*(?:no|yes|nope|yeah|yep)\s*[,!.:;-]\s*", re.IGNORECASE)


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
    duplicates: str = "versions",
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
    ``min_overlap``) and wrong claims reusing document words can pass. ``duplicates``: as
    in :func:`answer_grounded`."""
    _select([], id_key, text_key, duplicates)
    if _abstained(answer):
        return None
    docs = retrieved(run, tool, id_key=id_key, text_key=text_key)
    if not docs:
        return None
    known = _ids(docs, id_key)
    all_cites = _resolve(cited_ids(answer, pattern), known)
    content_of: dict[str, tuple[set[str], bool]] = {}

    def doc_info(ds: str) -> tuple[set[str], bool]:
        if ds not in content_of:
            content_of[ds] = (set(_content(ds, stopwords)), _negated(ds))
        return content_of[ds]

    for sentence in _SENT.split(answer if isinstance(answer, str) else ""):
        cites = _resolve(cited_ids(sentence, pattern), known) or all_cites
        claim = _INTERJECTION.sub("", re.sub(pattern, " ", sentence)).strip()
        if _CJK.search(claim):  # no spaces between words: overlap would be meaningless
            continue
        words = _content(claim, stopwords)
        if len(words) < min_words:
            continue
        used = [d for _, d in docs if str(d.get(id_key)) in cites] or [d for _, d in docs]
        used, conflict = _select(used, id_key, text_key, duplicates)
        if conflict:
            return f"{_answer_step(run)}: {conflict} ({_where(run, tool)})"
        doc_sents = [ds for d in used for ds in _SENT.split(str(d.get(text_key, ""))) if ds]
        vocab: set[str] = set()
        for ds in doc_sents:
            vocab |= doc_info(ds)[0]
        where = f"[{'], ['.join(cites)}]" if cites else "the retrieved documents"
        missing = [surface for st, surface in words.items() if st not in vocab]
        if (len(words) - len(missing)) / len(words) < min_overlap:
            return (
                f"{_answer_step(run)}: claim {claim!r} is not supported by {where} "
                f"({_where(run, tool)}): words {missing} not found"
            )
        # negation: the best-matching document sentence must not have the opposite
        # polarity (ties favour the same polarity, so mixed documents do not flag)
        neg = _negated(claim)
        best: dict[bool, tuple[float, str]] = {}
        for ds in doc_sents:
            vocab_ds, ds_neg = doc_info(ds)
            score = len(set(words) & vocab_ds) / len(words)
            if score > best.get(ds_neg, (-1.0, ""))[0]:
                best[ds_neg] = (score, ds)
        flip_score, flip_ds = best.get(not neg, (0.0, ""))
        same_score = best.get(neg, (0.0, ""))[0]
        if flip_score >= min_overlap and flip_score > same_score:
            return (
                f"{_answer_step(run)}: claim {claim!r} contradicts {where} "
                f"({_where(run, tool)}): negation mismatch with {flip_ds.strip()!r}"
            )
    return None
