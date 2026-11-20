"""Weighted multi-criteria rubric scoring for recorded agent runs.

A :class:`Rubric` grades a run 0–100 across deterministic criteria (outcome,
safety findings, tool efficiency, cost, latency, assertion rules, or a custom
check) instead of a single pass/fail. ``crashlab score`` and
``crashlab eval --rubric`` use the same API.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .models import SEVERITY_RANK, CheckResult, Run

__all__ = [
    "DEFAULT_RUBRIC",
    "CriterionScore",
    "FleetRubricReport",
    "Rubric",
    "RubricCriterion",
    "RubricError",
    "RubricScore",
    "grade_for",
    "load_rubric",
    "parse_rubric",
    "score_run",
    "score_runs",
]

_GRADE_CUTOFFS: tuple[tuple[float, str], ...] = (
    (90.0, "A"),
    (80.0, "B"),
    (70.0, "C"),
    (60.0, "D"),
)
_GRADES = ("A", "B", "C", "D", "F")
_SEVERITIES = ("info", "warn", "error", "critical")

_KIND_ALIASES = {
    "outcome": "outcome_pass",
    "outcome_pass": "outcome_pass",
    "safety": "no_findings",
    "findings": "no_findings",
    "no_findings": "no_findings",
    "cost": "cost_budget",
    "cost_budget": "cost_budget",
    "latency": "latency_budget",
    "latency_budget": "latency_budget",
    "tools": "tool_efficiency",
    "tool_efficiency": "tool_efficiency",
    "rule": "rule",
    "rules": "rules",
    "check": "check",
}
_KINDS = frozenset(_KIND_ALIASES.values())


class RubricError(ValueError):
    """A rubric file could not be read, or a criterion failed schema validation."""


def grade_for(score: float) -> str:
    """Letter grade for a 0–100 score: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, else F."""
    value = max(0.0, min(100.0, float(score)))
    for cutoff, letter in _GRADE_CUTOFFS:
        if value >= cutoff:
            return letter
    return "F"


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class RubricCriterion:
    """One weighted scoring dimension on a rubric."""

    name: str
    weight: float = 1.0
    kind: str = "outcome_pass"
    target: Any = None
    max_value: Any = None
    rule: Any = None
    check: Callable[[Run], Any] | None = None
    severity: str | None = None
    partial_credit: float | bool | None = None
    expected_tools: Sequence[str] | None = None
    target_cost_usd: float | None = None
    max_cost_usd: float | None = None
    target_ms: float | None = None
    max_ms: float | None = None
    rules: Any = None

    def resolved_kind(self) -> str:
        kind = _KIND_ALIASES.get(str(self.kind).strip().lower().replace("-", "_"))
        if kind is None:
            raise RubricError(
                f"unknown criterion kind {self.kind!r} (expected one of "
                f"{', '.join(sorted(_KINDS))})"
            )
        return kind


@dataclass(frozen=True)
class Rubric:
    """A named set of weighted criteria and a passing ``min_score`` (0–100)."""

    name: str = "default"
    criteria: list[RubricCriterion] = field(default_factory=list)
    min_score: float = 70.0

    def with_min_score(self, min_score: float | None) -> Rubric:
        if min_score is None:
            return self
        return replace(self, min_score=float(min_score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_score": self.min_score,
            "criteria": [
                {
                    "name": c.name,
                    "weight": c.weight,
                    "kind": c.kind,
                    "target": c.target,
                    "max_value": c.max_value,
                }
                for c in self.criteria
            ],
        }


@dataclass
class CriterionScore:
    """One criterion's contribution to a :class:`RubricScore`."""

    name: str
    kind: str
    weight: float
    score: float
    contribution: float
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RubricScore:
    """A single run's 0–100 rubric grade."""

    score: float
    grade: str
    passed: bool
    breakdown: list[CriterionScore] = field(default_factory=list)
    run_id: str | None = None
    name: str | None = None
    min_score: float = 70.0
    rubric: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "score": self.score,
            "grade": self.grade,
            "passed": self.passed,
            "min_score": self.min_score,
            "rubric": self.rubric,
            "breakdown": [c.to_dict() for c in self.breakdown],
        }


@dataclass
class FleetRubricReport:
    """Aggregate rubric scores across a fleet of runs."""

    results: list[RubricScore] = field(default_factory=list)
    rubric: str = "default"
    min_score: float = 70.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.results]

    @property
    def mean(self) -> float:
        values = self.scores
        return sum(values) / len(values) if values else 0.0

    @property
    def median(self) -> float:
        values = self.scores
        return float(statistics.median(values)) if values else 0.0

    @property
    def min(self) -> float:
        values = self.scores
        return min(values) if values else 0.0

    @property
    def max(self) -> float:
        values = self.scores
        return max(values) if values else 0.0

    @property
    def grades(self) -> dict[str, int]:
        hist = {g: 0 for g in _GRADES}
        for result in self.results:
            hist[result.grade] = hist.get(result.grade, 0) + 1
        return hist

    @property
    def weakest(self) -> str | None:
        """Name of the criterion with the lowest mean raw score, if any."""
        if not self.results or not self.results[0].breakdown:
            return None
        names = [c.name for c in self.results[0].breakdown]
        means: list[tuple[float, int, str]] = []
        for i, name in enumerate(names):
            values = [
                next((c.score for c in r.breakdown if c.name == name), None) for r in self.results
            ]
            present = [v for v in values if v is not None]
            if not present:
                continue
            means.append((sum(present) / len(present), i, name))
        if not means:
            return None
        means.sort()
        return means[0][2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric": self.rubric,
            "min_score": self.min_score,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "mean": self.mean,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "grades": self.grades,
            "weakest": self.weakest,
            "results": [r.to_dict() for r in self.results],
        }


# --- loading -------------------------------------------------------------------------


def _as_float(value: Any, loc: str, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RubricError(f"{loc}: {what} must be a number, got {value!r}")
    return float(value)


def _optional_float(value: Any, loc: str, *, what: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, loc, what=what)


def _tools_of(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    return [str(value)]


def _parse_criterion(item: dict[str, Any], *, index: int, source: str) -> RubricCriterion:
    loc = f"{source} criterion {index}"
    name = item.get("name") or item.get("id") or f"criterion_{index}"
    if not isinstance(name, str) or not name.strip():
        raise RubricError(f"{loc}: name must be a non-empty string")
    kind = item.get("kind") or item.get("type") or "outcome_pass"
    if not isinstance(kind, str):
        raise RubricError(f"{loc}: kind must be a string")
    resolved = _KIND_ALIASES.get(kind.strip().lower().replace("-", "_"))
    if resolved is None:
        raise RubricError(
            f"{loc}: unknown kind {kind!r} (expected one of {', '.join(sorted(_KINDS))})"
        )
    weight = item.get("weight", 1.0)
    weight_f = _as_float(weight, loc, what="weight")
    if weight_f < 0:
        raise RubricError(f"{loc}: weight must be >= 0, got {weight_f}")
    partial = item.get("partial_credit", item.get("warn_partial"))
    if isinstance(partial, str):
        lowered = partial.strip().lower()
        if lowered in {"true", "yes", "on"}:
            partial = True
        elif lowered in {"false", "no", "off"}:
            partial = False
        else:
            partial = _as_float(partial, loc, what="partial_credit")
    expected = item.get("expected_tools", item.get("tools"))
    return RubricCriterion(
        name=name.strip(),
        weight=weight_f,
        kind=resolved,
        target=item.get("target"),
        max_value=item.get("max_value", item.get("max")),
        rule=item.get("rule"),
        severity=item.get("severity"),
        partial_credit=partial,
        expected_tools=_tools_of(expected) if expected is not None else None,
        target_cost_usd=_optional_float(item.get("target_cost_usd"), loc, what="target_cost_usd"),
        max_cost_usd=_optional_float(item.get("max_cost_usd"), loc, what="max_cost_usd"),
        target_ms=_optional_float(item.get("target_ms"), loc, what="target_ms"),
        max_ms=_optional_float(item.get("max_ms"), loc, what="max_ms"),
        rules=item.get("rules"),
    )


def parse_rubric(data: Any, *, source: str = "<memory>") -> Rubric:
    """Validate a parsed document (mapping, ``{rubric: {...}}``, or a criteria list)."""
    if isinstance(data, list):
        data = {"criteria": data}
    if not isinstance(data, dict):
        raise RubricError(f"{source}: expected a mapping, got {type(data).__name__}")
    if "criteria" not in data and isinstance(data.get("rubric"), dict):
        inner = dict(data["rubric"])
        if "name" not in inner and isinstance(data.get("name"), str):
            inner["name"] = data["name"]
        data = inner
    raw_criteria = data.get("criteria", data.get("criterion"))
    if raw_criteria is None:
        raise RubricError(f"{source}: missing criteria")
    if isinstance(raw_criteria, dict):
        raw_criteria = [raw_criteria]
    if not isinstance(raw_criteria, list):
        raise RubricError(f"{source}: criteria must be a list")
    criteria: list[RubricCriterion] = []
    for i, item in enumerate(raw_criteria, 1):
        if not isinstance(item, dict):
            raise RubricError(f"{source} criterion {i}: expected a mapping")
        criteria.append(_parse_criterion(item, index=i, source=source))
    if not criteria:
        raise RubricError(f"{source}: no criteria found")
    name = data.get("name") or data.get("id") or "default"
    if not isinstance(name, str) or not name.strip():
        raise RubricError(f"{source}: name must be a non-empty string")
    min_score = data.get("min_score", 70.0)
    min_f = _as_float(min_score, source, what="min_score")
    return Rubric(name=name.strip(), criteria=criteria, min_score=min_f)


def load_rubric(path: str | Path) -> Rubric:
    """Load a rubric from a JSON, TOML or YAML file."""
    from .assertions import RuleError, _loads, _read_text

    src = Path(str(path)).expanduser()
    if not src.is_file():
        raise RubricError(f"rubric file not found: {src}")
    try:
        raw = _loads(src, _read_text(src))
    except RuleError as exc:
        raise RubricError(str(exc)) from exc
    return parse_rubric(raw, source=str(src))


# --- scoring -------------------------------------------------------------------------


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _run_cost(run: Run) -> float | None:
    from .budgets import summarize_usage

    summary = summarize_usage(run)
    if summary.cost_usd is not None:
        return float(summary.cost_usd)
    if run.usage.cost_usd is not None:
        return float(run.usage.cost_usd)
    return None


def _run_latency_ms(run: Run) -> float | None:
    if run.usage.wall_time_ms is not None:
        return float(run.usage.wall_time_ms)
    if run.ended_at is not None:
        try:
            return (run.ended_at - run.started_at).total_seconds() * 1000.0
        except TypeError:
            return None
    return None


def _budget_score(
    value: float | None, target: float | None, maximum: float | None, *, unit: str
) -> tuple[float, str]:
    if target is None and maximum is None:
        if value is None:
            return 1.0, f"no {unit} budget"
        return 1.0, f"{value:g} {unit} (no budget)"
    if value is None:
        return 1.0, f"{unit} unknown"
    if target is not None and maximum is not None and maximum > target:
        if value <= target:
            score = 1.0
        elif value >= maximum:
            score = 0.0
        else:
            score = 1.0 - (value - target) / (maximum - target)
        return score, f"{value:g} {unit} (target {target:g}, max {maximum:g})"
    limit = target if target is not None else maximum
    assert limit is not None
    score = 1.0 if value <= limit else 0.0
    return score, f"{value:g} {unit} (threshold {limit:g})"


def _outcome_score(run: Run) -> tuple[float, str]:
    failing = [c for c in (run.checks or []) if c.status != "pass"]
    if run.status == "error":
        err = run.error
        detail = f"{err.type}: {err.message}" if err else "run status error"
        return 0.0, detail
    if run.outcome == "fail":
        return 0.0, run.outcome_reason or "stored outcome is fail"
    if failing:
        names = ", ".join(c.name for c in failing)
        return 0.0, f"failing checks: {names}"
    if run.outcome == "pass":
        return 1.0, "outcome pass"
    return 1.0, f"outcome {run.outcome}, no error"


def _no_findings_score(run: Run, criterion: RubricCriterion) -> tuple[float, str]:
    raw = _first(criterion.severity, criterion.target, "error")
    severity = str(raw).strip().lower()
    if severity not in SEVERITY_RANK:
        raise RubricError(
            f"no_findings severity must be one of {', '.join(_SEVERITIES)}, not {raw!r}"
        )
    floor = SEVERITY_RANK[severity]
    serious = [f for f in (run.findings or []) if f.rank >= floor]
    if serious:
        detail = ", ".join(f"[{f.severity}] {f.kind}" for f in serious)
        return 0.0, detail
    warn_floor = SEVERITY_RANK["warn"]
    warns = [f for f in (run.findings or []) if f.rank >= warn_floor]
    partial = criterion.partial_credit
    if warns and partial not in (None, False, 0, 0.0):
        credit = 0.5 if partial is True else _clamp01(float(partial))
        detail = ", ".join(f"[{f.severity}] {f.kind}" for f in warns)
        return credit, f"warn-only partial credit: {detail}"
    if warns:
        return 1.0, "no findings at or above " + severity
    return 1.0, "no findings"


def _rule_items(criterion: RubricCriterion) -> list[Any]:
    raw = _first(criterion.rule, criterion.rules, criterion.target)
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    return [raw]


def _rule_score(run: Run, criterion: RubricCriterion) -> tuple[float, str]:
    from .assertions import Rule, RuleError, evaluate_rule, parse_rules

    items = _rule_items(criterion)
    if not items:
        raise RubricError(f"criterion {criterion.name!r}: rule kind requires a rule mapping")
    rules: list[Any] = []
    try:
        for item in items:
            if isinstance(item, Rule):
                rules.append(item)
            else:
                rules.extend(parse_rules(item if isinstance(item, list) else [item]))
    except RuleError as exc:
        return 0.0, f"invalid rule: {exc}"
    results = [evaluate_rule(rule, run) for rule in rules]
    failed = [r for r in results if r.status != "pass"]
    if failed:
        detail = "; ".join(
            (r.message or r.name) + (f" [{r.status}]" if r.status == "error" else "")
            for r in failed
        )
        return 0.0, detail
    names = ", ".join(r.name for r in results) or "rules"
    return 1.0, f"passed {names}"


def _tool_efficiency_score(run: Run, criterion: RubricCriterion) -> tuple[float, str]:
    from collections import Counter

    expected = _tools_of(_first(criterion.expected_tools, criterion.target))
    calls = [s.name for s in run.tool_calls()]
    total = len(calls)
    counts = Counter(calls)
    if expected:
        wanted = list(dict.fromkeys(t for t in expected if t))
        wanted_set = set(wanted)
        hits = sum(1 for name in wanted if name in counts)
        extra_calls = sum(n for name, n in counts.items() if name not in wanted_set)
        redundant = sum(max(0, counts[name] - 1) for name in wanted_set)
        denom = len(wanted) + extra_calls + redundant
        score = hits / denom if denom else 1.0
        return score, (
            f"{hits}/{len(wanted)} expected tools, {extra_calls} extra, {redundant} redundant"
        )
    if total == 0:
        return 1.0, "no tool calls"
    unique = len(counts)
    redundant = total - unique
    return unique / total, f"{unique} unique / {total} calls ({redundant} redundant)"


def _check_score(run: Run, criterion: RubricCriterion) -> tuple[float, str]:
    fn = criterion.check
    if fn is None:
        raise RubricError(f"criterion {criterion.name!r}: check kind requires a callable")
    try:
        result = fn(run)
    except Exception as exc:
        return 0.0, f"{type(exc).__name__}: {exc}"
    if isinstance(result, CheckResult):
        if result.status == "pass":
            return 1.0, result.message or f"{result.name} passed"
        return 0.0, result.message or f"{result.name} {result.status}"
    if isinstance(result, bool):
        return (1.0 if result else 0.0), ("passed" if result else "failed")
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        value = float(result)
        if value > 1.0:
            value = value / 100.0
        return _clamp01(value), f"score {result:g}"
    return 0.0, f"check returned unsupported value {type(result).__name__}"


def _as_bound(value: Any) -> float | None:
    if value is None or isinstance(value, (list, dict, bool)):
        return None
    return float(value)


def _cost_bounds(criterion: RubricCriterion) -> tuple[float | None, float | None]:
    return (
        _as_bound(_first(criterion.target_cost_usd, criterion.target)),
        _as_bound(_first(criterion.max_cost_usd, criterion.max_value)),
    )


def _latency_bounds(criterion: RubricCriterion) -> tuple[float | None, float | None]:
    return (
        _as_bound(_first(criterion.target_ms, criterion.target)),
        _as_bound(_first(criterion.max_ms, criterion.max_value)),
    )


def _criterion_score(run: Run, criterion: RubricCriterion) -> tuple[float, str]:
    kind = criterion.resolved_kind()
    if kind == "outcome_pass":
        return _outcome_score(run)
    if kind == "no_findings":
        return _no_findings_score(run, criterion)
    if kind in {"rule", "rules"}:
        return _rule_score(run, criterion)
    if kind == "cost_budget":
        target, maximum = _cost_bounds(criterion)
        return _budget_score(_run_cost(run), target, maximum, unit="USD")
    if kind == "latency_budget":
        target, maximum = _latency_bounds(criterion)
        return _budget_score(_run_latency_ms(run), target, maximum, unit="ms")
    if kind == "tool_efficiency":
        return _tool_efficiency_score(run, criterion)
    if kind == "check":
        return _check_score(run, criterion)
    raise RubricError(f"unknown criterion kind {kind!r}")


def _coerce_checks(checks: Iterable[Any]) -> list[Any]:
    from .verify.oracles import CheckSpec

    specs: list[Any] = []
    for item in checks:
        if isinstance(item, CheckSpec):
            specs.append(item)
        elif callable(item):
            specs.append(CheckSpec(name=getattr(item, "__name__", "check"), agent=None, fn=item))
        else:
            raise RubricError(
                f"checks= items must be callables or CheckSpec, got {type(item).__name__}"
            )
    return specs


def _apply_checks(run: Run, checks: Iterable[Any] | None) -> Run:
    if not checks:
        return run
    from .verify.oracles import apply_outcome, run_checks

    working = run.model_copy()
    results = run_checks(working, _coerce_checks(checks))
    if results:
        apply_outcome(working, results)
    return working


def score_run(
    run: Run, rubric: Rubric | None = None, *, checks: Iterable[Any] | None = None
) -> RubricScore:
    """Grade ``run`` with ``rubric`` (default :data:`DEFAULT_RUBRIC`) on a 0–100 scale."""
    used = rubric if rubric is not None else DEFAULT_RUBRIC
    if not used.criteria:
        raise RubricError(f"rubric {used.name!r} has no criteria")
    working = _apply_checks(run, checks)
    total_weight = sum(c.weight for c in used.criteria)
    breakdown: list[CriterionScore] = []
    weighted_sum = 0.0
    for criterion in used.criteria:
        raw, message = _criterion_score(working, criterion)
        raw = _clamp01(float(raw))
        contribution = (100.0 * criterion.weight * raw / total_weight) if total_weight else 0.0
        weighted_sum += criterion.weight * raw
        breakdown.append(
            CriterionScore(
                name=criterion.name,
                kind=criterion.resolved_kind(),
                weight=criterion.weight,
                score=raw,
                contribution=contribution,
                message=message,
            )
        )
    score = 0.0 if not total_weight else 100.0 * weighted_sum / total_weight
    score = round(max(0.0, min(100.0, score)), 4)
    grade = grade_for(score)
    return RubricScore(
        score=score,
        grade=grade,
        passed=score >= used.min_score,
        breakdown=breakdown,
        run_id=working.run_id,
        name=working.name,
        min_score=used.min_score,
        rubric=used.name,
    )


def score_runs(
    runs: Sequence[Run], rubric: Rubric | None = None, *, checks: Iterable[Any] | None = None
) -> FleetRubricReport:
    """Grade many runs and return mean/median/min/max, a grade histogram and the
    weakest criterion."""
    used = rubric if rubric is not None else DEFAULT_RUBRIC
    results = [score_run(run, used, checks=checks) for run in runs]
    return FleetRubricReport(results=results, rubric=used.name, min_score=used.min_score)


DEFAULT_RUBRIC = Rubric(
    name="default",
    min_score=70.0,
    criteria=[
        RubricCriterion(name="outcome", kind="outcome_pass", weight=0.35),
        RubricCriterion(
            name="safety",
            kind="no_findings",
            weight=0.25,
            target="error",
            partial_credit=0.5,
        ),
        RubricCriterion(name="tool_efficiency", kind="tool_efficiency", weight=0.15),
        RubricCriterion(name="cost", kind="cost_budget", weight=0.15),
        RubricCriterion(name="latency", kind="latency_budget", weight=0.10),
    ],
)
