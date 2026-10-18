"""crashlab: crash-test your AI agents.

Record every LLM and tool call from your existing agent loop into a local store
(``.crashlab/``), then inspect runs with ``crashlab ls`` / ``crashlab show``.

    import crashlab
    from openai import OpenAI

    client = crashlab.wrap(OpenAI())

    @crashlab.tool()
    def get_order(order_id: str) -> dict: ...

    @crashlab.agent(name="refund-bot")
    def run_agent(user_msg: str) -> str: ...

    run = crashlab.load("last")
"""

from __future__ import annotations

from ._version import __version__
from .capture import agent, record, tool, wrap
from .models import (
    Diagnosis,
    ErrorInfo,
    Evidence,
    Fix,
    Flag,
    Run,
    Step,
    TaxonomyLabel,
    Usage,
    Verification,
)
from .store import RunNotFoundError, Store, load
from .store import redact as redaction

__all__ = [
    "Diagnosis",
    "ErrorInfo",
    "Evidence",
    "Fix",
    "Flag",
    "Run",
    "RunNotFoundError",
    "Step",
    "Store",
    "TaxonomyLabel",
    "Usage",
    "Verification",
    "__version__",
    "agent",
    "load",
    "record",
    "redaction",
    "tool",
    "wrap",
]
