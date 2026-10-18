"""crashlab: crash-test your AI agents.

Record every LLM and tool call from your existing agent loop into a local store
(``.crashlab/``), inspect runs with ``crashlab ls`` / ``crashlab show``, replay them
offline with ``crashlab.replay(run)``, and judge them with ``@crashlab.check`` oracles.

    import crashlab
    from openai import OpenAI

    client = crashlab.wrap(OpenAI())          # or anthropic.Anthropic()

    @crashlab.tool()
    def get_order(order_id: str) -> dict: ...

    @crashlab.agent(name="refund-bot")
    def run_agent(user_msg: str) -> str: ...

    @crashlab.check("refund-bot")
    def refund_not_above_total(run): ...

    run = crashlab.load("last")
    crashlab.replay(run)                      # offline; DivergenceError if it differs
"""

from __future__ import annotations

from ._version import __version__
from .capture import agent, record, tool, wrap
from .context import (
    ContextThreadPoolExecutor,
    bind_context,
    copy_context_to_thread,
    is_replaying,
    run_in_thread,
)
from .models import (
    CheckResult,
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
from .replay import (
    DivergenceError,
    ReplayError,
    ReplayNetworkError,
    SideEffectBlockedError,
    replay,
)
from .store import RunNotFoundError, Store, load
from .store import redact as redaction
from .verify import check, run_checks

__all__ = [
    "CheckResult",
    "ContextThreadPoolExecutor",
    "Diagnosis",
    "DivergenceError",
    "ErrorInfo",
    "Evidence",
    "Fix",
    "Flag",
    "ReplayError",
    "ReplayNetworkError",
    "Run",
    "RunNotFoundError",
    "SideEffectBlockedError",
    "Step",
    "Store",
    "TaxonomyLabel",
    "Usage",
    "Verification",
    "__version__",
    "agent",
    "bind_context",
    "check",
    "copy_context_to_thread",
    "is_replaying",
    "load",
    "record",
    "redaction",
    "replay",
    "run_checks",
    "run_in_thread",
    "tool",
    "wrap",
]
