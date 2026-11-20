"""Chaos resilience: re-run a recorded agent under built-in fault scenarios."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .faults import (
    FaultRule,
    llm_bad_tool_call,
    llm_empty,
    llm_error,
    tool_error,
    tool_result,
)
from .models import Run
from .store import Store

SCENARIOS = (
    "tool_502",
    "tool_timeout",
    "tool_empty",
    "tool_schema_drift",
    "llm_rate_limit",
    "llm_empty_reply",
    "llm_bad_tool_args",
)

_TOOL_SCENARIOS = frozenset({"tool_502", "tool_timeout", "tool_empty", "tool_schema_drift"})
_LLM_SCENARIOS = frozenset({"llm_rate_limit", "llm_empty_reply", "llm_bad_tool_args"})


@dataclass
class StressResult:
    """One scenario applied to one target (a tool name, or the model)."""

    scenario: str
    target: str | None
    outcome: str  # survived, crashed, failed_checks, skipped, needs_live_agent
    survived: bool
    checks_passed: bool | None = None
    findings: list[str] = field(default_factory=list)
    run_id: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StressReport:
    """Results of :func:`stress_run` / ``crashlab stress``."""

    source_run_id: str
    results: list[StressResult] = field(default_factory=list)

    @property
    def survived(self) -> bool:
        """True when every executed scenario recovered (skipped ones are ignored)."""
        ran = [r for r in self.results if r.outcome not in ("skipped", "needs_live_agent")]
        return bool(ran) and all(r.survived for r in ran)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_run_id": self.source_run_id,
            "survived": self.survived,
            "results": [r.as_dict() for r in self.results],
        }


def _tools_used(run: Run, agent: str | None) -> list[str]:
    names: list[str] = []
    for step in run.steps:
        if step.operation != "execute_tool":
            continue
        if agent and not _agent_ok(step.agent, agent):
            continue
        if step.name not in names:
            names.append(step.name)
    return names


def _models_used(run: Run, agent: str | None) -> list[str]:
    names: list[str] = []
    for step in run.steps:
        if step.operation != "chat":
            continue
        if agent and not _agent_ok(step.agent, agent):
            continue
        model = step.attributes.get("gen_ai.request.model") or step.name
        if isinstance(model, str) and model not in names:
            names.append(model)
    return names


def _agent_ok(actual: str | None, wanted: str) -> bool:
    if not actual:
        return False
    return actual == wanted or (actual.startswith(wanted + "#") and "#" not in wanted)


def _rule_for(scenario: str, target: str | None, agent: str | None) -> FaultRule:
    kw: dict[str, Any] = {"on_calls": [1]}
    if agent is not None:
        kw["agent"] = agent
    if scenario == "tool_502":
        return tool_result(tool=target or "*", mode="html_502", **kw)
    if scenario == "tool_timeout":
        return tool_error(tool=target or "*", exc=TimeoutError, message="injected timeout", **kw)
    if scenario == "tool_empty":
        return tool_result(tool=target or "*", mode="empty", **kw)
    if scenario == "tool_schema_drift":
        return tool_result(tool=target or "*", mode="schema_drift", **kw)
    if scenario == "llm_rate_limit":
        return llm_error(status=429, message="injected rate limit", **kw)
    if scenario == "llm_empty_reply":
        return llm_empty(**kw)
    if scenario == "llm_bad_tool_args":
        return llm_bad_tool_call(arguments='{"order_id":', **kw)
    raise ValueError(f"unknown stress scenario {scenario!r}; choose from {SCENARIOS}")


def _planned(
    run: Run, scenarios: Sequence[str] | None, agent: str | None
) -> list[tuple[str, str | None]]:
    wanted = list(scenarios or SCENARIOS)
    unknown = [s for s in wanted if s not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown stress scenario(s) {unknown}; choose from {SCENARIOS}")
    tools = _tools_used(run, agent)
    models = _models_used(run, agent)
    out: list[tuple[str, str | None]] = []
    for name in wanted:
        if name in _TOOL_SCENARIOS:
            if not tools:
                out.append((name, None))  # later: skipped, no tool in the run
            else:
                out.extend((name, t) for t in tools)
        elif name in _LLM_SCENARIOS:
            out.append((name, models[0] if models else None))
    return out


def _skipped(scenario: str, target: str | None, outcome: str, detail: str) -> StressResult:
    return StressResult(
        scenario=scenario, target=target, outcome=outcome, survived=False,
        checks_passed=None, findings=[], detail=detail,
    )  # fmt: skip


def _entrypoint_available(
    source: Run, fn: Callable[..., Any] | None, store: Store
) -> tuple[Callable[..., Any] | None, str | None]:
    if fn is not None:
        return fn, None
    from .replay.engine import ReplayError, _ancestor_entrypoint, resolve_entrypoint

    entry = _ancestor_entrypoint(store, source) or source.entrypoint
    if not entry:
        return None, "the run has no recorded entry point"
    try:
        return resolve_entrypoint(entry), None
    except ReplayError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - a broken import is a skip, not a crash
        return None, f"{type(exc).__name__}: {exc}"


def _grade(branch: Run) -> tuple[str, bool, bool | None, list[str], str | None]:
    findings = [f"{f.kind}: {f.message}" for f in branch.findings]
    checks_passed = all(c.status == "pass" for c in branch.checks) if branch.checks else None
    if branch.status == "error":
        detail = None
        if branch.error is not None:
            detail = f"{branch.error.type}: {branch.error.message}"
        return "crashed", False, checks_passed, findings, detail
    if checks_passed is False:
        failed = [c.name for c in branch.checks if c.status != "pass"]
        return "failed_checks", False, False, findings, "failed: " + ", ".join(failed)
    return "survived", True, True if checks_passed else checks_passed, findings, None


def stress_run(
    run: Run | str = "last",
    *,
    scenarios: Sequence[str] | None = None,
    at_step: int = 0,
    agent: str | None = None,
    store: Store | str | None = None,
    model: str | None = None,
    fn: Callable[..., Any] | None = None,
    client: Any = None,
    overrides: dict[str, Any] | None = None,
    live: bool = True,
) -> StressReport:
    """Re-run ``run`` under built-in chaos scenarios and report what survived.

    Each scenario targets a tool or model the source run actually used (a 502
    HTML page from a tool, a tool timeout, an empty tool result, schema drift,
    an LLM rate limit, an empty reply, malformed tool-call arguments). When the
    recorded entry point (or ``fn``) can be called and a live client is
    available, the scenario is a :func:`~crashlab.fork` with an active
    :func:`~crashlab.faults` context. Otherwise the result is
    ``needs_live_agent`` or ``skipped``.

    ``live=False`` skips execution and reports ``needs_live_agent`` for every
    scenario (the default for a CLI invocation that would otherwise call a
    real model). Pass ``client=`` (or rely on the agent's own client, with
    ``live=True``) to actually run the trials.
    """
    from .faults import faults
    from .replay import ForkError, ReplayError, ReplayNetworkError, fork
    from .replay.divergence import DivergenceError
    from .store import Store as StoreCls

    st = store if isinstance(store, StoreCls) else StoreCls(store)
    source = st.load_run(run.run_id if isinstance(run, Run) else run)
    report = StressReport(source_run_id=source.run_id)
    planned = _planned(source, scenarios, agent)
    target_fn, entry_err = _entrypoint_available(source, fn, st)
    data = dict(overrides or {})
    if model is not None and "model" not in data:
        data["model"] = model

    can_run = live and target_fn is not None
    for scenario, target in planned:
        if scenario in _TOOL_SCENARIOS and target is None:
            report.results.append(
                _skipped(scenario, None, "skipped", "the run recorded no tool calls")
            )
            continue
        if scenario in _LLM_SCENARIOS and not _models_used(source, agent):
            report.results.append(
                _skipped(scenario, target, "skipped", "the run recorded no LLM calls")
            )
            continue
        if not can_run:
            why = entry_err or "re-running this scenario needs the agent and a live client"
            report.results.append(_skipped(scenario, target, "needs_live_agent", why))
            continue
        rule = _rule_for(scenario, target, agent)
        try:
            with faults(rule):
                branch = fork(
                    source,
                    at_step=at_step,
                    fn=target_fn,
                    client=client,
                    overrides=data or None,
                    tools="live",
                    store_dir=st.root,
                    raise_errors=False,
                )
        except DivergenceError as exc:
            report.results.append(_skipped(scenario, target, "skipped", f"fork diverged: {exc}"))
            continue
        except (ForkError, ReplayError, ReplayNetworkError) as exc:
            report.results.append(_skipped(scenario, target, "needs_live_agent", str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 - one scenario must not abort the report
            report.results.append(
                StressResult(
                    scenario=scenario,
                    target=target,
                    outcome="crashed",
                    survived=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )  # fmt: skip
            )
            continue
        outcome, survived, checks_passed, findings, detail = _grade(branch)
        report.results.append(
            StressResult(
                scenario=scenario,
                target=target,
                outcome=outcome,
                survived=survived,
                checks_passed=checks_passed,
                findings=findings,
                run_id=branch.run_id,
                detail=detail,
            )  # fmt: skip
        )
    return report


__all__ = ["SCENARIOS", "StressReport", "StressResult", "stress_run"]
