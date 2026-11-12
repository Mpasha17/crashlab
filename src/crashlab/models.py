"""Pydantic v2 data model for crashlab runs, steps and (later) diagnoses.

Step attribute keys follow the OpenTelemetry GenAI semantic conventions
(``gen_ai.operation.name``, ``gen_ai.request.model``, ``gen_ai.tool.name`` ...), so an
OTLP export can be a straight mapping. The conventions are still in Development status.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

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
    # not a failure label: the diagnoser found nothing wrong with the agent's behaviour
    NO_FAILURE = "NO_FAILURE"


Operation = Literal["chat", "embeddings", "execute_tool", "invoke_agent", "custom"]
RunStatus = Literal["running", "ok", "error"]
Outcome = Literal["pass", "fail", "unknown"]
# live: really executed; cassette: served from a recording; patched: changed by a fork
# patch; blocked: a side-effect tool a replay / fork refused to run; fake: a side-effect
# tool a fork served from its registered fake (crashlab.fake_tool)
StepSource = Literal["live", "cassette", "patched", "blocked", "fake"]


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
    # an SDK status error's parsed response body (``exc.body``), when the message leaves
    # it out: Gemini's OpenAI endpoint returns a JSON list, so the OpenAI SDK's message is
    # just "Error code: 400"
    body: Any | None = None
    # what handlers read besides the message: small JSON ``args`` (``KeyError('b')``)
    # and public JSON attributes (``e.sku``, ``e.retry_after``); rebuilt on replay
    state: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    def _drop_empty_state(self, handler: Any) -> Any:
        # crashlab<=0.0.2 rejects unknown keys (extra="forbid"): only write it when set
        data = handler(self)
        if isinstance(data, dict) and data.get("state") is None:
            data.pop("state", None)
        return data

    @property
    def detail(self) -> str:
        """The message, plus the provider's error text from ``body`` when the message is
        only a status line ("Error code: 400")."""
        extra = _body_message(self.body)
        if extra and extra not in self.message:
            return f"{self.message} - {extra}"
        return self.message


def _body_message(body: Any) -> str:
    """The human part of a provider error body (``{"error": {"message": ...}}``, a list
    of those, or a plain string)."""
    if body is None:
        return ""
    if isinstance(body, list) and body:
        return _body_message(body[0])
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict):
            msg = inner.get("message") or inner.get("detail")
            if msg:
                return " ".join(str(msg).split())
        elif isinstance(inner, str):
            return " ".join(inner.split())
    if isinstance(body, str):
        return " ".join(body.split())[:500]
    return ""


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
    def agent(self) -> str | None:
        """The agent this step belongs to in a multi-agent run (``gen_ai.agent.name``,
        set by ``crashlab.wrap(client, agent=...)``, ``crashlab.agent_scope`` or a nested
        ``@crashlab.agent``); None when not labelled."""
        name = self.attributes.get("gen_ai.agent.name")
        return name if isinstance(name, str) else None

    @property
    def result(self) -> Any:
        return (self.response or {}).get("result")


def _msg_key(msg: Any) -> str:
    return json.dumps(msg, sort_keys=True, default=str)


def _answered_calls(request: dict[str, Any] | None) -> list[str]:
    """Tool call ids a chat request answers (tool results in its messages)."""
    out: list[str] = []
    for m in (request or {}).get("messages") or []:
        if not isinstance(m, dict):
            continue
        if m.get("tool_call_id"):
            out.append(str(m["tool_call_id"]))
        for p in m.get("parts") or []:
            if isinstance(p, dict) and p.get("type") == "tool_call_response" and p.get("id"):
                out.append(str(p["id"]))
    return out


class Handoff(_Model):
    """One agent handing the conversation to another (see :meth:`Run.handoffs`)."""

    from_agent: str
    to_agent: str
    #: first chat step of ``to_agent`` after the handoff
    step: int
    #: chat step of ``from_agent`` that made the handoff tool call
    via_step: int
    tool: str | None = None
    call_id: str


def _continued_thread(threads: list[dict[str, Any]], msgs: list[str]) -> int | None:
    """The conversation ``msgs`` continues: its last request is a strict prefix of
    ``msgs``; the one whose tool calls ``msgs`` answers first, else the longest."""
    best: int | None = None
    for i, t in enumerate(threads):
        prev = t["msgs"]
        if not prev or len(prev) >= len(msgs) or msgs[: len(prev)] != prev:
            continue
        tail = "".join(msgs[len(prev) :])
        if t["calls"] and any(c in tail for c in t["calls"]):
            return i
        if best is None or len(prev) >= len(threads[best]["msgs"]):
            best = i
    return best


def _interleaved(threads: list[dict[str, Any]]) -> bool:
    end = -1
    for start, stop in sorted((min(t["steps"]), max(t["steps"])) for t in threads if t["steps"]):
        if start < end:
            return True
        end = max(end, stop)
    return False


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
    diagnosis: Diagnosis | None = None  # the latest `crashlab diagnose` result
    fix_attempts: list[dict[str, Any]] = Field(default_factory=list)  # verify history
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    code_version: str | None = None
    env: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    span_id: str | None = None
    # Populated by crashlab.load(); not stored in the run row (steps live in the cassette).
    steps: list[Step] = Field(default_factory=list)

    @property
    def parent_run(self) -> str | None:
        """Run id of the run that started this one in another process or service
        (``crashlab.child_env()`` / ``record(parent=...)``); None when not linked."""
        link = (self.env or {}).get("link")
        return link.get("parent_run") if isinstance(link, dict) else None

    @property
    def trace_id(self) -> str:
        """Id shared by a run and every run it started, directly or not: the root run's
        id. An unlinked run is its own trace."""
        link = (self.env or {}).get("link")
        trace = link.get("trace") if isinstance(link, dict) else None
        return trace or self.run_id

    def describe_step(self, idx: int | None) -> str:
        """``"step 3 (tool place_order, called by chat step 2)"``: a tool step names the
        LLM call that decided to make it, since a diagnosis often points at the tool call
        (where the damage shows) while the decision was the model's."""
        if idx is None:
            return "step ?"
        step = next((s for s in self.steps if s.idx == idx), None)
        if step is None:
            return f"step {idx}"
        if step.operation != "execute_tool":
            return f"step {idx} ({step.operation} {step.name})"
        caller = self.caller_chat(idx)
        by = f", called by chat step {caller}" if caller is not None else ""
        return f"step {idx} (tool {step.name}{by})"

    def caller_chat(self, idx: int) -> int | None:
        """The chat step that requested tool step ``idx``: the latest earlier chat step
        whose response holds the tool step's ``gen_ai.tool.call.id`` (exact even when
        sub-agents run concurrently), else the nearest earlier chat step."""
        step = next((s for s in self.steps if s.idx == idx), None)
        chats = sorted(
            (s for s in self.steps if s.idx < idx and s.operation == "chat"),
            key=lambda s: -s.idx,
        )
        call_id = step.attributes.get("gen_ai.tool.call.id") if step is not None else None
        if call_id:
            for chat in chats:
                calls = (chat.response or {}).get("tool_calls") or []
                if any(isinstance(c, dict) and c.get("id") == call_id for c in calls):
                    return chat.idx
        return chats[0].idx if chats else None

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

    def _agent_match(self, agent: str | None) -> Callable[[Step], bool]:
        if agent is None:
            return lambda s: True
        if "#" not in agent:
            return lambda s: s.agent == agent
        labels = self.agent_instances()
        return lambda s: labels.get(s.idx) == agent

    def chat_steps(self, *, agent: str | None = None) -> list[Step]:
        """Chat steps (optionally only those of one agent, see :attr:`Step.agent`, or of
        one concurrent instance of it, ``"researcher#2"``, see :meth:`agent_instances`)."""
        match = self._agent_match(agent)
        return [s for s in self.steps if s.operation == "chat" and match(s)]

    def tool_calls(
        self, name: str | None = None, *, ok_only: bool = False, agent: str | None = None
    ) -> list[Step]:
        """Tool steps (optionally only one tool / only those that did not raise / only
        those of one agent or agent instance)."""
        match = self._agent_match(agent)
        return [
            s for s in self.tool_steps(name) if not (ok_only and s.error is not None) and match(s)
        ]

    def handoffs(self) -> list[Handoff]:
        """Handoffs between agents, in order: a chat step of agent B whose request answers a
        tool call made by a different agent A (OpenAI Agents SDK ``transfer_to_*``,
        LlamaIndex AgentWorkflow ``handoff``) continues A's conversation. When it answers
        several of A's calls, the latest one is the handoff. Agents called as tools start
        a fresh conversation and are not handoffs. Worked out from the recording, so it
        also works on older runs; needs agent labels."""
        calls: dict[str, tuple[int, str, str | None]] = {}
        used: set[str] = set()
        out: list[Handoff] = []
        for s in sorted(self.steps, key=lambda s: s.idx):
            if s.operation != "chat" or s.agent is None:
                continue
            fresh = [c for c in _answered_calls(s.request)
                     if c in calls and c not in used and calls[c][1] != s.agent]  # fmt: skip
            if fresh:
                cid = max(fresh, key=lambda c: calls[c][0])
                via, src, tool = calls[cid]
                out.append(Handoff(from_agent=src, to_agent=s.agent, step=s.idx, via_step=via,
                                   tool=tool, call_id=cid))  # fmt: skip
                used.update(fresh)
            for c in (s.response or {}).get("tool_calls") or []:
                if isinstance(c, dict) and c.get("id"):
                    name = c.get("name") or (c.get("function") or {}).get("name")
                    calls.setdefault(str(c["id"]), (s.idx, s.agent, name))
        return out

    @property
    def agent_labels(self) -> list[str]:
        """Agent labels in order of first appearance, with concurrent instances told
        apart (``["manager", "researcher#1", "researcher#2"]``); same as :attr:`agents`
        when no agent ran twice at once."""
        seen: dict[str, None] = {}
        labels = self.agent_instances()
        for idx in sorted(labels):
            seen.setdefault(labels[idx], None)
        return list(seen)

    def agent_instances(self) -> dict[int, str]:
        """Step index -> agent label, telling apart concurrent runs of the same agent.

        Chat steps of one agent are grouped into conversations: a request that extends an
        earlier request of that agent (its messages start with them) continues it, the one
        whose tool calls it answers first. When two conversations of the same agent
        interleave (the agent ran twice in parallel, e.g. ``as_tool`` called twice), their
        steps are labelled ``name#1``, ``name#2`` in order of first step; tool steps take the
        label of the chat that requested them. Otherwise the plain name is kept (an agent
        called again later is not split). Unlabelled steps are left out."""
        steps = sorted(self.steps, key=lambda s: s.idx)
        threads: dict[str, list[dict[str, Any]]] = {}
        thread_of: dict[int, tuple[str, int]] = {}
        for s in steps:
            if s.operation != "chat" or s.agent is None:
                continue
            msgs = [_msg_key(m) for m in (s.request or {}).get("messages") or []]
            own = threads.setdefault(s.agent, [])
            pick = _continued_thread(own, msgs) if msgs else (len(own) - 1 if own else None)
            if pick is None:
                own.append({"steps": [], "msgs": msgs, "calls": set()})
                pick = len(own) - 1
            t = own[pick]
            t["steps"].append(s.idx)
            if msgs:
                t["msgs"] = msgs
            t["calls"] = {c.get("id") for c in (s.response or {}).get("tool_calls") or []
                          if isinstance(c, dict) and c.get("id")}  # fmt: skip
            thread_of[s.idx] = (s.agent, pick)
        split = {name for name, ts in threads.items() if _interleaved(ts)}
        out: dict[int, str] = {}
        for s in steps:
            if s.agent is None:
                continue
            key = thread_of.get(s.idx)
            if key is None and s.operation == "execute_tool":
                caller = self.caller_chat(s.idx)
                key = thread_of.get(caller) if caller is not None else None
                if key is not None and key[0] != s.agent:
                    key = None
            if key is not None and key[0] in split:
                out[s.idx] = f"{key[0]}#{key[1] + 1}"
            else:
                out[s.idx] = s.agent
        return out

    @property
    def agents(self) -> list[str]:
        """Agent names labelled on this run's steps, in order of first appearance."""
        seen: dict[str, None] = {}
        for s in sorted(self.steps, key=lambda s: s.idx):
            if s.agent is not None:
                seen.setdefault(s.agent, None)
        return list(seen)

    def tool_results(self, name: str | None = None) -> list[Any]:
        """Results of the tool calls that succeeded, in order."""
        return [s.result for s in self.tool_calls(name, ok_only=True)]


# --- Diagnosis / fixes / verification (schemas only in v0.0.1) ------------------------


class Fix(_Model):
    """A proposed fix: one typed patch (see :mod:`crashlab.patch`) that ``crashlab fork
    --patch`` can apply, ranked, with why it should work. Produced by
    :func:`crashlab.propose_fixes` and validated before it is kept."""

    fix_id: str
    rank: int = 1
    kind: str  # the patch kind: system_prompt, tool_description, tool_schema ...
    target: str = ""  # "system prompt", a tool name, "step 3"
    before: str | None = None
    after: str = ""
    rationale: str = ""
    expected_effect: str = ""
    as_patch: Patch | None = None
    test_only: bool = False  # a counterfactual (tool_result): proves a cause, not a fix
    file: str | None = None  # the saved patch file (``crashlab fork --patch FILE``)
    notes: list[str] = Field(default_factory=list)
    tried: dict[str, Any] | None = None  # ``--try``: the fork run id, outcome and checks


class Diagnosis(_Model):
    """Which step went wrong, why (a taxonomy label), and the verbatim evidence.

    Produced by :func:`crashlab.diagnose`. ``evidence`` only holds quotes that were found
    verbatim in the cited step; if none survive, ``label`` is ``UNKNOWN``. ``notes`` says
    what validation changed (dropped quotes, a downgrade and why)."""

    run_id: str
    failing_step: int | None = None
    step_window: tuple[int, int] | None = None
    label: TaxonomyLabel = TaxonomyLabel.UNKNOWN
    secondary: list[TaxonomyLabel] = Field(default_factory=list)
    confidence: float = 0.0
    root_cause: str = ""  # one plain sentence
    evidence: list[Evidence] = Field(default_factory=list)
    explanation: str = ""
    notes: list[str] = Field(default_factory=list)
    model: str | None = None  # diagnoser model; None for the heuristic diagnosis
    created_at: datetime = Field(default_factory=utcnow)
    fixes: list[Fix] = Field(default_factory=list)
    fix_notes: list[str] = Field(default_factory=list)  # rejected proposals and why
    fix_method: dict[str, Any] = Field(default_factory=dict)
    method: dict[str, Any] = Field(default_factory=dict)
    confirmed_by_counterfactual: bool | None = None

    @field_validator("secondary", mode="before")
    @classmethod
    def _coerce_secondary(cls, v: Any) -> Any:
        """Drop unknown / non-label entries so old or hand-edited diagnosis JSON still loads.
        Valid TaxonomyLabel strings and members are kept; non-lists become []."""
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, (list, tuple)):
            return []
        valid = {m.value for m in TaxonomyLabel}
        out: list[Any] = []
        for item in v:
            if isinstance(item, TaxonomyLabel) or isinstance(item, str) and item in valid:
                out.append(item)
        return out

    @model_validator(mode="after")
    def _normalize_secondary(self) -> Diagnosis:
        """Dedupe secondary, drop primary / UNKNOWN / NO_FAILURE, cap at 3 (loop13).
        Empty default keeps old consumers that omit the field compatible."""
        seen: set[TaxonomyLabel] = set()
        out: list[TaxonomyLabel] = []
        for lbl in self.secondary:
            if lbl in (self.label, TaxonomyLabel.UNKNOWN, TaxonomyLabel.NO_FAILURE):
                continue
            if lbl in seen:
                continue
            seen.add(lbl)
            out.append(lbl)
            if len(out) >= 3:
                break
        if out != list(self.secondary):
            object.__setattr__(self, "secondary", out)
        return self

    @property
    def is_failure(self) -> bool:
        return self.label not in (TaxonomyLabel.NO_FAILURE, TaxonomyLabel.UNKNOWN)


class Trial(_Model):
    """One rerun inside a verification. ``status`` is the oracle verdict; ``errored``
    means the provider failed (429/5xx, connection) and the trial says nothing about the
    fix, so it is never counted as a fail."""

    role: Literal["patched", "control", "regression"]
    source_run: str  # the run that was forked
    run_id: str | None = None  # the child run (None if the fork never started)
    status: Literal["pass", "fail", "errored"]
    outcome: str | None = None
    failed_checks: list[str] = Field(default_factory=list)
    error: str | None = None
    cost_usd: float = 0.0
    tokens: int = 0


class ArmStats(_Model):
    """Pass counts for one arm (patched or control), errored trials excluded."""

    passes: int = 0
    fails: int = 0
    errored: int = 0
    rate: float | None = None  # passes / (passes + fails)
    ci95: tuple[float, float] | None = None  # Wilson interval on the per-trial pass rate
    pass_hat_k: float | None = None  # estimated P(all k trials pass), tau-bench style

    @property
    def n(self) -> int:
        return self.passes + self.fails


class RegressionResult(_Model):
    run_id: str  # the past good run
    status: Literal["pass", "fail", "errored", "skipped"]
    trial_run: str | None = None
    note: str | None = None


Verdict = Literal["VERIFIED", "LIKELY", "INCONCLUSIVE", "REJECTED"]


class Verification(_Model):
    """``crashlab verify``: k patched forks vs k unpatched controls, plus the patch rerun
    on past good runs. Child runs are tagged ``verify:<verification_id>``."""

    verification_id: str = ""
    run_id: str
    fix_id: str
    patch_description: str = ""
    # Applied-patch provenance (loop16c): what verify actually forked with.
    # Optional so older verification JSON / attempts without these keys still load.
    patch_kind: str | None = None
    patch_op: str | None = None
    patch_digest: str | None = None  # stable short digest of the applied patch payload
    fix_file: str | None = None
    k: int
    fork_step: int = 0
    patched: ArmStats = Field(default_factory=ArmStats)
    control: ArmStats | None = None
    regression: list[RegressionResult] = Field(default_factory=list)
    trials: list[Trial] = Field(default_factory=list)
    verdict: Verdict = "INCONCLUSIVE"
    reasons: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    tokens: int = 0
    max_cost_usd: float | None = None
    budget_exhausted: bool = False
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def regressions(self) -> list[RegressionResult]:
        return [r for r in self.regression if r.status == "fail"]


Run.model_rebuild()
