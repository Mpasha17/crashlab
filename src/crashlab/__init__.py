"""crashlab: crash-test your AI agents.

Record every LLM and tool call from your existing agent loop into a local store
(``.crashlab/``), inspect runs with ``crashlab ls`` / ``crashlab show``, replay them
offline with ``crashlab.replay(run)``, judge them with ``@crashlab.check`` oracles, and
ask "what if" with ``crashlab.fork(run, at_step=N, patch=...)``. Built-in runtime checks
(loops, tool error streaks, bad tool arguments, suspicious tool results, budgets, length
cut-offs) run while recording and store :class:`Finding` s on the run.

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
    branch = crashlab.fork(run, at_step=0,
                           patch=crashlab.patch.SystemPrompt(op="append", text="..."))
    crashlab.diff(run, branch).summary()
"""

from __future__ import annotations

from . import checks, patch, pricing
from ._version import __version__
from .capture import agent, record, tool, wrap
from .checks import ChecksConfig, CircuitBreakerError
from .compare import RunDiff, diff
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
    Finding,
    Fix,
    Flag,
    Run,
    Step,
    TaxonomyLabel,
    Usage,
    Verification,
)
from .patch import (
    InjectMessagePatch,
    MessagePatch,
    ModelParamPatch,
    PatchError,
    PatchSet,
    SystemPromptPatch,
    ToolDescriptionPatch,
    ToolResultPatch,
    ToolSchemaPatch,
)
from .replay import (
    DivergenceError,
    ForkError,
    PatchedToolError,
    ReplayError,
    ReplayNetworkError,
    SideEffectBlockedError,
    fork,
    replay,
)
from .store import RunNotFoundError, Store, load
from .store import redact as redaction
from .verify import check, run_checks

__all__ = [
    "CheckResult",
    "ChecksConfig",
    "CircuitBreakerError",
    "ContextThreadPoolExecutor",
    "Diagnosis",
    "DivergenceError",
    "ErrorInfo",
    "Evidence",
    "Finding",
    "Fix",
    "Flag",
    "ForkError",
    "InjectMessagePatch",
    "MessagePatch",
    "ModelParamPatch",
    "PatchError",
    "PatchSet",
    "PatchedToolError",
    "ReplayError",
    "ReplayNetworkError",
    "Run",
    "RunDiff",
    "RunNotFoundError",
    "SideEffectBlockedError",
    "Step",
    "Store",
    "SystemPromptPatch",
    "TaxonomyLabel",
    "ToolDescriptionPatch",
    "ToolResultPatch",
    "ToolSchemaPatch",
    "Usage",
    "Verification",
    "__version__",
    "agent",
    "bind_context",
    "check",
    "checks",
    "copy_context_to_thread",
    "diff",
    "fork",
    "is_replaying",
    "load",
    "patch",
    "pricing",
    "record",
    "redaction",
    "replay",
    "run_checks",
    "run_in_thread",
    "tool",
    "wrap",
]
