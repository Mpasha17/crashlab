"""rag-lc: RAG checks over LangChain-style retriever documents (no vector database).

The retriever returns ``langchain_core.documents.Document`` objects when langchain-core
is installed, else a pydantic stand-in with the same fields (``page_content``,
``metadata``, ``id``). crashlab records them as dicts; ``crashlab.rag`` takes the id from
``metadata["source"]`` and the text from ``page_content``. The corpus holds two versions of
``refund.md`` (different ``metadata["updated"]``), so the checks pass
``duplicates="latest"``: answers must match the current version. That configuration lives
in the checks below, so ``crashlab check`` and exported tests use it too.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel

import crashlab
from crashlab import rag

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - depends on the environment

    class Document(BaseModel):  # type: ignore[no-redef]
        page_content: str
        metadata: dict[str, Any] = {}
        id: str | None = None


QUESTION = "How many days do I have to get a refund?"
CORPUS = [
    ("Refunds are accepted within 30 days of delivery.",
     {"source": "refund.md", "updated": "2026-09-01"}),
    ("Refunds are accepted within 14 days of delivery.",
     {"source": "refund.md", "updated": "2024-01-10"}),
    ("EU shipping takes 5-7 business days.", {"source": "shipping.md", "updated": "2026-05-02"}),
]  # fmt: skip
SYSTEM = (
    "Answer only from retrieve results. Cite sources as [source]. "
    "If the documents do not contain the answer, say you don't know."
)
TOOLS = [{"type": "function", "function": {
    "name": "retrieve", "description": "Search the help center.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}]  # fmt: skip


@crashlab.tool()
def retrieve(query: str) -> list[Document]:
    words = {w.rstrip("s") for w in query.lower().split() if len(w) > 2}
    return [Document(page_content=t, metadata=dict(m)) for t, m in CORPUS
            if any(w in t.lower() for w in words)]  # fmt: skip


@crashlab.agent(name="rag-lc", ignore_args=("client",))
def run_agent(question: str, *, client: Any = None, model: str | None = None) -> str:
    if client is None:
        from openai import OpenAI

        client = crashlab.wrap(OpenAI())
    model = model or os.environ.get("MODEL") or "gpt-4o-mini"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    for _ in range(4):
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
            docs = retrieve(str(json.loads(tc.function.arguments or "{}").get("query", "")))
            payload = [{"source": d.metadata.get("source"), "text": d.page_content} for d in docs]
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(payload)})  # fmt: skip
    return "Sorry, I could not answer."


def _answer(run: crashlab.Run) -> str:
    return run.output if isinstance(run.output, str) else ""


@crashlab.check("rag-lc")
def grounded_in_latest_version(run: crashlab.Run) -> str | None:
    """Numbers in the answer come from the newest version of each cited source."""
    return rag.answer_grounded(run, _answer(run), tool="retrieve", duplicates="latest")


@crashlab.check("rag-lc")
def claims_supported(run: crashlab.Run) -> str | None:
    """Answer sentences are supported by the newest version of the cited sources."""
    return rag.claims_supported(run, _answer(run), tool="retrieve", duplicates="latest")


@crashlab.check("rag-lc")
def citations_were_retrieved(run: crashlab.Run) -> str | None:
    """Every cited [source] was retrieved."""
    return rag.citations_were_retrieved(run, _answer(run), tool="retrieve")


if __name__ == "__main__":  # pragma: no cover - live model
    print(run_agent(QUESTION))
    print(crashlab.load("last").outcome_reason)
