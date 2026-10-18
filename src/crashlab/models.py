"""Pydantic v2 data model for crashlab runs, steps and (later) diagnoses.

Step attribute keys follow the OpenTelemetry GenAI semantic conventions
(``gen_ai.operation.name``, ``gen_ai.request.model``, ``gen_ai.tool.name`` ...), so an
OTLP export can be a straight mapping. The conventions are still in Development status.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .patch import AnyPatch, PatchSet


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaxonomyLabel(str, Enum):
    """v0.1 failure labels (6 + UNKNOWN). The diagnoser must pick from this enum."""

    TOOL_ERROR_MISHANDLED = "TOOL_ERROR_MISHANDLED"
    BAD_TOOL_ARGUMENTS = "BAD_TOOL_ARGUMENTS"
    WRONG_TOOL_SELECTION = "WRONG_TOOL_SELECTION"
    STEP_REPETITION = "STEP_REPETITION"
    TASK_SPEC_VIOLATION = "TASK_SPEC_VIOLATION"
    EXTERNAL_FAILURE = "EXTERNAL_FAILURE"
    UNKNOWN = "UNKNOWN"


Operation = Literal["chat", "execute_tool", "invoke_agent", "custom"]
RunStatus = Literal["running", "ok", "error"]
Outcome = Literal["pass", "fail", "unknown"]
# live: really executed; cassette: served from a recording; patched: changed by a fork
# patch; blocked: a side-effect tool a replay / fork refused to run
StepSource = Literal["live", "cassette", "patched", "blocked"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Usage(_Model):
    """Summed resource usage for a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    errors: int = 0
    wall_time_ms: float | None = None


class ErrorInfo(_Model):
    type: str
    message: str
    is_tool_error: bool = False
    status_code: int | None = None
    traceback: str | None = None


Severity = Literal["info", "warn", "error", "critical"]
SEVERITY_RANK: dict[str, int] = {"info": 0, "warn": 1, "error": 2, "critical": 3}


class Evidence(_Model):
    """A verbatim quote from a recorded step (``field`` is a dotted path such as
    ``result``, ``args`` or ``response.text``)."""

    step_idx: int
    field: str
    quote: str


class Finding(_Model):
    """What a runtime check (built-in detector) found in a run: a loop, an error streak,
    a suspicious tool result, a blown budget ...

    Findings describe *how the run behaved*; they do not decide pass / fail. That is what
    ``@crashlab.check`` outcome oracles do (``run.checks`` / ``run.outcome``).
    """

    kind: str  # detector: step_repetition, tool_error_streak, bad_tool_args, ...
    severity: Severity = "warn"
    step: int | None = None  # first step where it fired
    steps: list[int] = Field(default_factory=list)  # every step involved
    message: str
    evidence: list[Evidence] = Field(default_factory=list)
    key: str | None = None  # identity within a run (repeats extend one finding)
    stopped_run: bool = False  # the circuit breaker stopped the run on this finding

    @model_validator(mode="before")
    @classmethod
    def _legacy(cls, data: Any) -> Any:
        """Day-1..4 flags: ``{detector, step_idx, severity, message}``."""
        if isinstance(data, dict) and ("detector" in data or "step_idx" in data):
            data = dict(data)
            data.setdefault("kind", data.pop("detector", "flag"))
            data.setdefault("step", data.pop("step_idx", None))
            data.pop("detector", None)
            data.pop("step_idx", None)
        return data

    @property
    def detector(self) -> str:  # older name
        return self.kind

    @property
    def step_idx(self) -> int | None:  # older name
        return self.step

    @property
    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)


Flag = Finding  # older name


# --- Patches -------------------------------------------------------------------------
# The typed patch models live in crashlab.patch (SystemPromptPatch, MessagePatch, ...).
Patch = AnyPatch


# --- Steps and runs ------------------------------------------------------------------


class CheckResult(_Model):
    """Result of one ``@crashlab.check`` outcome oracle on a run."""

    name: str
    status: Literal["pass", "fail", "error"]
    message: str | None = None


class Step(_Model):
    """One recorded interaction in a run: an LLM call, a tool execution or a custom span."""

    run_id: str
    idx: int
    span_id: str
    parent_span_id: str | None = None
    operation: Operation
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    # Normalized payloads.
    #   chat:         request={"system","messages","tools","params"}
    #                 response={"messages","tool_calls","finish_reason","model","usage"}
    #   execute_tool: request={"args"}, response={"result"}
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] | None = None
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    request_hash: str
    started_at: datetime = Field(default_factory=utcnow)
    duration_ms: float = 0.0
    source: StepSource = "live"

    @property
    def args(self) -> dict[str, Any]:
        return self.request.get("args", {})

    @property
    def result(self) -> Any:
        return (self.response or {}).get("result")


class Run(_Model):
    run_id: str
    name: str
    entrypoint: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any | None = None
    status: RunStatus = "running"
    outcome: Outcome = "unknown"
    outcome_reason: str | None = None
    error: ErrorInfo | None = None
    replay_of: str | None = None  # source run id when this run is a strict replay
    forked_from: str | None = None  # parent run id when this run is a fork
    fork_step: int | None = None  # steps < fork_step were served from the parent's cassette
    patch: PatchSet | None = None  # what the fork changed
    patch_description: str | None = None
    # runtime-check findings (detectors); run.checks holds the @crashlab.check oracles
    findings: list[Finding] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    check_config: dict[str, Any] | None = None  # effective runtime-check config
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    code_version: str | None = None
    env: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    span_id: str | None = None
    # Populated by crashlab.load(); not stored in the run row (steps live in the cassette).
    steps: list[Step] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _legacy_fields(cls, data: Any) -> Any:
        """Runs stored before forks existed used parent_run_id / fork_at / a stub patch;
        before runtime checks, findings were called flags."""
        if isinstance(data, dict) and "flags" in data:
            data = dict(data)
            flags = data.pop("flags") or []
            data["findings"] = list(data.get("findings") or []) + list(flags)
        if isinstance(data, dict) and ("parent_run_id" in data or "fork_at" in data):
            data = dict(data)
            parent, at = data.pop("parent_run_id", None), data.pop("fork_at", None)
            data.setdefault("forked_from", parent)
            data.setdefault("fork_step", at)
        if isinstance(data, dict) and isinstance(data.get("patch"), dict):
            patch = data["patch"]
            if patch.get("kind") in ("noop", "raw"):
                data = {**data, "patch": None}
            elif "kind" in patch:
                data = {**data, "patch": {"patches": [patch]}}
        return data

    @property
    def is_fork(self) -> bool:
        return self.forked_from is not None

    @property
    def flags(self) -> list[Finding]:  # older name
        return self.findings

    def findings_of(self, kind: str | None = None, *, min_severity: str = "info") -> list[Finding]:
        floor = SEVERITY_RANK.get(min_severity, 0)
        return [f for f in self.findings if (kind is None or f.kind == kind) and f.rank >= floor]

    def tool_steps(self, name: str | None = None) -> list[Step]:
        return [
            s
            for s in self.steps
            if s.operation == "execute_tool" and (name is None or s.name == name)
        ]

    def chat_steps(self) -> list[Step]:
        return [s for s in self.steps if s.operation == "chat"]

    def tool_calls(self, name: str | None = None, *, ok_only: bool = False) -> list[Step]:
        """Tool steps (optionally only one tool / only those that did not raise)."""
        return [s for s in self.tool_steps(name) if not (ok_only and s.error is not None)]

    def tool_results(self, name: str | None = None) -> list[Any]:
        """Results of the tool calls that succeeded, in order."""
        return [s.result for s in self.tool_calls(name, ok_only=True)]


# --- Diagnosis / fixes / verification (schemas only in v0.0.1) ------------------------


class Fix(_Model):
    fix_id: str
    kind: Literal[
        "system_prompt", "tool_description", "tool_result_note", "guard_config", "model_swap"
    ]
    target: str
    before: str | None = None
    after: str
    rationale: str
    as_patch: Patch | None = None


class Diagnosis(_Model):
    run_id: str
    failing_step: int | None
    step_window: tuple[int, int] | None = None
    label: TaxonomyLabel = TaxonomyLabel.UNKNOWN
    secondary: list[TaxonomyLabel] = Field(default_factory=list)
    confidence: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)
    explanation: str = ""
    fixes: list[Fix] = Field(default_factory=list)
    method: dict[str, Any] = Field(default_factory=dict)
    confirmed_by_counterfactual: bool | None = None


class RegressionResult(_Model):
    run_id: str
    k: int
    passes: int


class Verification(_Model):
    run_id: str
    fix_id: str
    k: int
    target_passes: int
    regression: list[RegressionResult] = Field(default_factory=list)
    pass_hat_k: float
    ci95: tuple[float, float]
    verdict: Literal["fixes_and_safe", "fixes_with_regressions", "does_not_fix", "inconclusive"]
    cost_usd: float | None = None
