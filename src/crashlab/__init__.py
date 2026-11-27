"""crashlab: crash-test your AI agents.

Record every LLM and tool call from your existing agent loop into a local store
(``.crashlab/``), inspect runs with ``crashlab ls`` / ``crashlab show``, replay them
offline with ``crashlab.replay(run)``, judge them with ``@crashlab.check`` oracles, and
ask "what if" with ``crashlab.fork(run, at_step=N, patch=...)`` or inject faults with
``crashlab.faults(...)`` / ``crashlab stress``. Built-in runtime checks
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

from . import assertions, budgets, checks, flakes, judges, patch, pricing, rag, suite
from ._version import __version__
from .capture import agent, child_env, clear_fakes, fake_tool, record, tool, wrap
from .checks import ChecksConfig, CircuitBreakerError
from .compare import RunDiff, diff
from .context import (
    ContextThreadPoolExecutor,
    agent_scope,
    bind_context,
    copy_context_to_thread,
    is_replaying,
    run_in_thread,
)
from .dataset import build_dataset, inspect_dataset
from .diagnose import DiagnoseError, diagnose, propose_fixes, try_fix
from .faults import faults
from .fuzz import fuzz_agent, fuzz_run
from .guard import GuardViolation, guard
from .integrations.instrument import instrument, instrumented, uninstrument
from .judges import Judge, JudgeError, JudgeResult, judge_run, judge_runs
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
from .repair import RepairError, RepairResult, repair
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
from .rubrics import Rubric, RubricCriterion, RubricScore, score_run, score_runs
from .simulate import Scenario, ScenarioTurn, SimulationResult, simulate
from .store import RunNotFoundError, Store, load
from .store import redact as redaction
from .stress import stress_run
from .verify import VerifyError, check, run_checks, verify

__all__ = [
    "DiagnoseError",
    "diagnose",
    "propose_fixes",
    "try_fix",
    "verify",
    "VerifyError",
    "repair",
    "RepairError",
    "RepairResult",
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
    "Judge",
    "JudgeError",
    "JudgeResult",
    "ForkError",
    "GuardViolation",
    "InjectMessagePatch",
    "MessagePatch",
    "ModelParamPatch",
    "PatchError",
    "PatchSet",
    "PatchedToolError",
    "ReplayError",
    "ReplayNetworkError",
    "Run",
    "Rubric",
    "RubricCriterion",
    "RubricScore",
    "RunDiff",
    "RunNotFoundError",
    "Scenario",
    "ScenarioTurn",
    "SideEffectBlockedError",
    "SimulationResult",
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
    "agent_scope",
    "assertions",
    "bind_context",
    "budgets",
    "build_dataset",
    "check",
    "checks",
    "clear_fakes",
    "copy_context_to_thread",
    "diff",
    "fake_tool",
    "faults",
    "guard",
    "flakes",
    "fork",
    "fuzz_agent",
    "fuzz_run",
    "inspect_dataset",
    "instrument",
    "instrumented",
    "is_replaying",
    "judge_run",
    "judge_runs",
    "judges",
    "load",
    "patch",
    "pricing",
    "rag",
    "record",
    "child_env",
    "redaction",
    "replay",
    "run_checks",
    "run_in_thread",
    "score_run",
    "score_runs",
    "simulate",
    "stress_run",
    "suite",
    "tool",
    "uninstrument",
    "wrap",
]
