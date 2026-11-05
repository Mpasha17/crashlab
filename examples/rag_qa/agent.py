"""rag-qa: a minimal retrieval-augmented support bot with RAG outcome oracles.

The retriever is an in-memory keyword search over ``CORPUS`` (no vector database), wrapped
with ``@crashlab.tool`` so its results are recorded and replayed offline. Swap
``search_docs`` for your own retriever; the oracles only read the recorded results.

Faults (``faults=[...]``), recorded in the run input so replay reproduces them:

- ``wrong_retrieval``: the retriever returns off-topic documents
- ``missing_context``: the retriever returns nothing
- ``stale_docs``: the retriever returns the superseded policy version
- ``retrieval_timeout``: the retriever raises ``TimeoutError``
- ``retrieval_error``: the retriever returns an error object instead of documents

Run against a real model: ``python -m examples.rag_qa.agent "How many days ...?"``
(``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` / ``MODEL``).
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
from typing import Any

import crashlab
from crashlab import rag

FAULTS = (
    "wrong_retrieval",
    "missing_context",
    "stale_docs",
    "retrieval_timeout",
    "retrieval_error",
)

CORPUS: list[dict[str, Any]] = [
    {"id": "refund-policy-v2", "updated": "2026-09-01",
     "text": "Refunds are accepted within 30 days of delivery for unused items."},
    {"id": "refund-policy-v1", "updated": "2024-01-10", "superseded_by": "refund-policy-v2",
     "text": "Refunds are accepted within 14 days of delivery."},
    {"id": "shipping-eu", "updated": "2026-05-02",
     "text": "Standard shipping to the EU takes 5-7 business days."},
    {"id": "warranty", "updated": "2026-03-15",
     "text": "Electronics carry a 2-year warranty; returns of opened electronics are repairs."},
]  # fmt: skip

# question -> gold document ids (what a correct retriever must return)
GOLD = {"How many days do I have to return an item?": ["refund-policy-v2"]}

SYSTEM = (
    "You answer customer questions using only documents returned by search_docs. "
    "Call search_docs first. Cite every document you use as [doc-id]. "
    "If the documents do not contain the answer, say you don't know."
)

_faults: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "rag_qa_faults", default=frozenset()
)


@crashlab.tool()
def search_docs(query: str, k: int = 2) -> list[dict[str, Any]]:
    """Keyword search over CORPUS (top k, superseded documents excluded)."""
    faults = _faults.get()
    if "retrieval_timeout" in faults:
        raise TimeoutError("search_docs: vector store did not answer within 5s")
    if "retrieval_error" in faults:
        return {"error": "index unavailable", "status": 503}  # type: ignore[return-value]
    if "missing_context" in faults:
        return []
    if "wrong_retrieval" in faults:
        return [d for d in CORPUS if d["id"] in ("shipping-eu", "warranty")]
    words = {w for w in query.lower().split() if len(w) > 2}
    pool = [d for d in CORPUS if "stale_docs" in faults or not d.get("superseded_by")]
    if "stale_docs" in faults:
        pool = [d for d in pool if d["id"] != "refund-policy-v2"]

    def score(d: dict[str, Any]) -> int:
        text = (d["id"] + " " + d["text"]).lower().replace("-", " ")
        return sum(w.rstrip("s") in text for w in words)

    hits = sorted((d for d in pool if score(d)), key=score, reverse=True)
    return [dict(d) for d in hits[:k]]


TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the help-center documents.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    },
}]  # fmt: skip


@crashlab.agent(name="rag-qa", ignore_args=("client",))
def run_agent(
    question: str,
    *,
    client: Any = None,
    model: str | None = None,
    faults: list[str] | None = None,
    max_turns: int = 4,
) -> str:
    unknown = set(faults or ()) - set(FAULTS)
    if unknown:
        raise ValueError(f"unknown faults {sorted(unknown)}; choose from {FAULTS}")
    if client is None:
        from openai import OpenAI

        client = crashlab.wrap(OpenAI())
    model = model or os.environ.get("MODEL") or "gpt-4o-mini"
    token = _faults.set(frozenset(faults or ()))
    try:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        for _ in range(max_turns):
            msg = (
                client.chat.completions.create(
                    model=model, messages=messages, tools=TOOLS, temperature=0
                )
                .choices[0]
                .message
            )
            if not msg.tool_calls:
                return msg.content or ""
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]})  # fmt: skip
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                try:
                    result: Any = search_docs(str(args.get("query", "")))
                except TimeoutError as exc:  # tell the model; the checks judge the answer
                    result = {"error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result)})  # fmt: skip
        return "Sorry, I could not answer."
    finally:
        _faults.reset(token)


# --- outcome oracles (one per RAG failure mode) -------------------------------------


def _answer(run: crashlab.Run) -> str:
    return run.output if isinstance(run.output, str) else ""


@crashlab.check("rag-qa")
def retrieved_gold_doc(run: crashlab.Run) -> str | None:
    """Wrong retrieval: the gold documents for the question were retrieved."""
    gold = GOLD.get(str(run.input.get("question", "")))
    return rag.retrieved_ids(run, gold) if gold else None


@crashlab.check("rag-qa")
def abstains_without_context(run: crashlab.Run) -> str | None:
    """Missing context: no documents -> the answer says it doesn't know."""
    return rag.abstains_without_context(run, _answer(run))


@crashlab.check("rag-qa")
def no_stale_docs(run: crashlab.Run) -> str | None:
    """Stale documents: nothing superseded was retrieved."""
    return rag.no_stale_docs(run)


@crashlab.check("rag-qa")
def citations_were_retrieved(run: crashlab.Run) -> str | None:
    """Citation mismatch: every cited [doc-id] was retrieved; answers cite something."""
    return rag.citations_were_retrieved(run, _answer(run))


@crashlab.check("rag-qa")
def answer_grounded(run: crashlab.Run) -> str | None:
    """Unsupported answer: every number in the answer is in a cited retrieved document."""
    return rag.answer_grounded(run, _answer(run))


@crashlab.check("rag-qa")
def grounded_in_current_docs(run: crashlab.Run) -> str | None:
    """Stale grounding: numbers must come from non-superseded documents."""
    return rag.answer_grounded(run, _answer(run), exclude_stale=True)


@crashlab.check("rag-qa")
def claims_supported(run: crashlab.Run) -> str | None:
    """Unsupported claim: answer sentences share enough words (same negation) with the
    documents they cite."""
    return rag.claims_supported(run, _answer(run))


if __name__ == "__main__":  # pragma: no cover - live model
    print(run_agent(" ".join(sys.argv[1:]) or next(iter(GOLD))))
    print(crashlab.load("last").outcome_reason)
