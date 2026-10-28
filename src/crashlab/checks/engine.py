"""Run the detectors over a run's steps, inline (one step at a time while recording) or
after the fact (``crashlab check``), and merge what they find into ``run.findings``."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..models import SEVERITY_RANK, Finding, Run, Step
from .config import ChecksConfig
from .detectors import KINDS, History, build, classify_result, shape_of

if TYPE_CHECKING:
    from ..store import Store

log = logging.getLogger("crashlab.checks")
logging.getLogger("crashlab").addHandler(logging.NullHandler())  # library: no stray output
MAX_EVIDENCE = 6


class CircuitBreakerError(BaseException):
    """Raised (opt-in, ``checks={"circuit_breaker": True}``) when a runtime check finds
    something critical: the agent is stopped at its next LLM / tool call.

    It derives from ``BaseException`` (like ``KeyboardInterrupt``) so the agent's own
    ``except Exception`` handlers do not swallow it; catch it explicitly to handle it.
    The run is stored with ``status="error"`` and the finding marked ``stopped_run``.
    """

    def __init__(self, finding: Finding) -> None:
        self.finding = finding
        at = f" at step {finding.step}" if finding.step is not None else ""
        super().__init__(f"crashlab stopped the run{at}: [{finding.kind}] {finding.message}")


def store_baseline(
    store: Store, run_name: str, exclude: str | None, limit: int
) -> Callable[[str], list[dict[str, str]]]:
    """tool name -> result shapes from the latest ``limit`` other finished runs of the
    same agent (loaded once, on first use)."""
    cache: dict[str, list[dict[str, str]]] | None = None

    def load() -> dict[str, list[dict[str, str]]]:
        out: dict[str, list[dict[str, str]]] = {}
        try:
            rows = store.list_runs(limit=limit * 4 + 10, name=run_name)
        except Exception:  # pragma: no cover - unreadable index
            return out
        picked = [r for r in rows if r["run_id"] != exclude and r["status"] == "ok"
                  and not r.get("replay_of")][:limit]  # fmt: skip
        for row in picked:
            try:
                steps = store.load_run(row["run_id"]).steps
            except Exception:
                continue
            for s in steps:
                if s.operation != "execute_tool" or s.error is not None:
                    continue
                shape = shape_of(s.result)
                if shape is not None and not classify_result(s.result):
                    out.setdefault(s.name, []).append(shape)
        return out

    def get(tool: str) -> list[dict[str, str]]:
        nonlocal cache
        if cache is None:
            cache = load()
        return cache.get(tool, [])

    return get


class CheckEngine:
    """Feeds steps to the enabled detectors and keeps one finding per (kind, key)."""

    def __init__(
        self,
        config: ChecksConfig,
        run: Run,
        *,
        baseline: Callable[[str], list[dict[str, str]]] | None = None,
    ) -> None:
        self.config = config
        self.detectors = build(config)
        self.h = History(run=run, baseline=baseline)
        self.by_key: dict[tuple[str, str], Finding] = {}
        self.findings: list[Finding] = []

    @property
    def active(self) -> bool:
        return bool(self.detectors)

    def _account(self, step: Step) -> None:
        from ..pricing import step_cost

        h = self.h
        h.steps.append(step)
        if step.operation == "chat":
            h.llm_calls += 1
            a = step.attributes
            inp = int(a.get("gen_ai.usage.input_tokens") or 0)
            out = int(a.get("gen_ai.usage.output_tokens") or 0)
            total = a.get("crashlab.usage.total_tokens")
            h.tokens += int(total) if total is not None else inp + out
            cost = a.get("crashlab.usage.cost_usd")
            if cost is None:
                cost = step_cost(step)
            if cost is None:
                h.cost_unknown = True
            else:
                h.cost_usd += float(cost)
        elif step.operation == "execute_tool":
            h.tool_calls += 1
        try:
            started = (step.started_at - h.run.started_at).total_seconds()
        except (TypeError, ValueError):  # naive vs aware datetimes in old data
            started = 0.0
        h.elapsed_s = max(h.elapsed_s, started + step.duration_ms / 1000)

    def _merge(self, found: list[Finding]) -> list[Finding]:
        """Add new findings / extend existing ones. Returns the ones that are new or got
        more severe."""
        changed = []
        for f in found:
            key = (f.kind, f.key or f"{f.step}:{f.message}")
            old = self.by_key.get(key)
            if old is None:
                if not f.steps and f.step is not None:
                    f.steps = [f.step]
                self.by_key[key] = f
                self.findings.append(f)
                changed.append(f)
                continue
            for i in f.steps:
                if i not in old.steps:
                    old.steps.append(i)
            seen = {(e.step_idx, e.field, e.quote) for e in old.evidence}
            for e in f.evidence:
                if len(old.evidence) < MAX_EVIDENCE and (e.step_idx, e.field, e.quote) not in seen:
                    old.evidence.append(e)
            if SEVERITY_RANK[f.severity] > SEVERITY_RANK[old.severity]:
                old.severity = f.severity
                changed.append(old)
            if len(f.steps) > 1:  # the detector described the whole picture again
                old.message = f.message
            elif len(old.steps) > 1:  # a per-step finding that repeats
                old.message = _with_steps(old)
        return changed

    def on_step(self, step: Step) -> list[Finding]:
        if not self.detectors:
            return []
        self._account(step)
        found: list[Finding] = []
        for d in self.detectors:
            try:
                found += d.on_step(step, self.h)
            except Exception:  # a detector bug must never break the agent
                log.debug("crashlab detector %s failed", d.kind, exc_info=True)
        changed = self._merge(found)
        for f in changed:
            level = logging.WARNING if f.rank >= SEVERITY_RANK["error"] else logging.INFO
            log.log(level, "crashlab %s (%s): %s", f.kind, f.severity, f.message)
        return changed

    def on_end(self, run: Run) -> list[Finding]:
        if not self.detectors:
            return []
        if len(self.h.steps) > len(run.steps):
            # a live run's steps go to the store as they happen; run.steps is empty here, so
            # end-of-run checks (no_llm_calls, failed_side_effect_claim ...) saw nothing
            run = run.model_copy(update={"steps": list(self.h.steps)})
        found: list[Finding] = []
        for d in self.detectors:
            try:
                found += d.on_end(run, self.h)
            except Exception:
                log.debug("crashlab detector %s failed", d.kind, exc_info=True)
        return self._merge(found)


def _with_steps(f: Finding) -> str:
    """Repeated single-step findings (an HTML page from 3 calls) mention every step."""
    base = f.message.split(" (steps ")[0]
    return f"{base} (steps {', '.join(str(i) for i in sorted(f.steps))})"


def evaluate(
    run: Run,
    config: ChecksConfig,
    *,
    store: Store | None = None,
) -> list[Finding]:
    """Run the detectors over a stored run's steps (does not modify the run)."""
    baseline = None
    if store is not None and config.suspicious_tool_result.baseline_runs:
        baseline = store_baseline(store, run.name, run.run_id,
                                  config.suspicious_tool_result.baseline_runs)  # fmt: skip
    probe = run.model_copy(update={"findings": [], "steps": []})
    engine = CheckEngine(config, probe, baseline=baseline)
    for step in sorted(run.steps, key=lambda s: s.idx):
        engine.on_step(step)
    engine.on_end(run)
    return engine.findings


def replace_findings(run: Run, findings: list[Finding]) -> None:
    """Swap the detector findings on ``run`` for ``findings``; other findings (fork /
    replay notices) are kept."""
    old = run.findings
    keep = [f for f in old if f.kind not in KINDS]
    new = [f.model_copy() for f in findings]
    # The finding that stopped the run is history, not a re-evaluable opinion: keep the
    # marker (strict replay re-trips on it) on the matching new finding, or keep the
    # original finding if the new config no longer produces it.
    for stopped in (f for f in old if f.stopped_run and f.kind in KINDS):
        match = next(
            (f for f in new if f.kind == stopped.kind and f.key == stopped.key), None
        ) or next(
            (f for f in new if f.kind == stopped.kind and stopped.step in (f.steps or [f.step])),
            None,
        )
        if match is not None:
            match.stopped_run = True
        else:
            new.append(stopped)
    run.findings = keep + new


def max_severity(findings: list[Finding]) -> str | None:
    if not findings:
        return None
    return max(findings, key=lambda f: f.rank).severity


__all__ = [
    "CheckEngine",
    "CircuitBreakerError",
    "evaluate",
    "max_severity",
    "replace_findings",
    "store_baseline",
]
