"""Fleet analytics over recorded runs: ``crashlab stats`` / :func:`compute_stats`."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from ._runscan import ScanStats, iter_runs, parse_since
from .models import Run, Step
from .pricing import run_cost, step_cost
from .store import Store

__all__ = [
    "AgentBreakdown",
    "CheckStat",
    "FindingStat",
    "FleetStats",
    "ModelBreakdown",
    "RunCost",
    "ToolStat",
    "compute_stats",
]

_TOP_N = 10


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pct(values: list[float], p: float) -> float | None:
    """Linear interpolation between closest ranks. ``None`` when there are no values."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _median(values: list[float]) -> float | None:
    return _pct(values, 50)


def _round(value: float | None, ndigits: int = 6) -> float | None:
    return None if value is None else round(value, ndigits)


def _tokens(run: Run) -> int:
    if run.usage.total_tokens:
        return int(run.usage.total_tokens)
    total = 0
    for step in run.steps:
        if step.operation != "chat":
            continue
        a = step.attributes or {}
        raw = a.get("crashlab.usage.total_tokens")
        if raw is not None and not isinstance(raw, bool):
            try:
                total += int(raw)
                continue
            except (TypeError, ValueError):
                pass
        inp, out = a.get("gen_ai.usage.input_tokens"), a.get("gen_ai.usage.output_tokens")
        try:
            total += int(inp or 0) + int(out or 0)
        except (TypeError, ValueError):
            continue
    return total


def _duration_s(run: Run) -> float | None:
    if run.usage.wall_time_ms is not None:
        return max(run.usage.wall_time_ms, 0.0) / 1000.0
    if run.ended_at is not None and run.started_at is not None:
        return max((run.ended_at - run.started_at).total_seconds(), 0.0)
    return None


def _cost(run: Run) -> float | None:
    try:
        return run_cost(run)
    except Exception:
        return None


def _step_usd(step: Step) -> float | None:
    raw = (step.attributes or {}).get("crashlab.usage.cost_usd")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if step.operation != "chat":
        return None
    try:
        return step_cost(step)
    except Exception:
        return None


def _step_model(step: Step) -> str | None:
    a = step.attributes or {}
    for key in ("gen_ai.request.model", "gen_ai.response.model"):
        value = a.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if step.operation == "chat" and step.name:
        return step.name
    return None


def _parse_since(since: datetime | str | None) -> datetime | None:
    if since is None or isinstance(since, datetime):
        return since
    return parse_since(str(since))


@dataclass
class AgentBreakdown:
    name: str
    runs: int
    pass_n: int = 0
    fail_n: int = 0
    error_n: int = 0
    unknown_n: int = 0
    total_cost_usd: float | None = 0.0
    total_tokens: int = 0
    mean_duration_s: float | None = None


@dataclass
class ModelBreakdown:
    model: str
    calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float | None = 0.0
    error_n: int = 0


@dataclass
class ToolStat:
    name: str
    calls: int
    error_n: int = 0
    error_rate: float = 0.0
    p95_duration_s: float | None = None


@dataclass
class FindingStat:
    kind: str
    severity: str
    count: int


@dataclass
class CheckStat:
    name: str
    fail_n: int
    error_n: int = 0


@dataclass
class RunCost:
    run_id: str
    name: str
    outcome: str
    duration_s: float | None = None
    cost_usd: float | None = None
    tokens: int = 0


@dataclass
class FleetStats:
    """Aggregate counts and cost / latency / tool stats over a set of recorded runs."""

    n: int = 0
    n_pass: int = 0
    n_fail: int = 0
    n_error: int = 0  # status == "error"
    n_unknown: int = 0
    pass_rate: float = 0.0
    fail_rate: float = 0.0
    error_rate: float = 0.0
    unknown_rate: float = 0.0
    total_cost_usd: float | None = 0.0
    cost_known: int = 0
    cost_unknown: int = 0
    mean_cost_usd: float | None = None
    median_cost_usd: float | None = None
    p95_cost_usd: float | None = None
    total_tokens: int = 0
    mean_duration_s: float | None = None
    p95_duration_s: float | None = None
    duration_known: int = 0
    by_agent: list[AgentBreakdown] = field(default_factory=list)
    by_model: list[ModelBreakdown] = field(default_factory=list)
    tools: list[ToolStat] = field(default_factory=list)
    findings: list[FindingStat] = field(default_factory=list)
    failing_checks: list[CheckStat] = field(default_factory=list)
    slowest: list[RunCost] = field(default_factory=list)
    most_expensive: list[RunCost] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bucket(run: Run) -> str:
    """Mutually exclusive outcome bucket for rates that sum to 1."""
    if run.status == "error":
        return "error"
    if run.outcome == "pass":
        return "pass"
    if run.outcome == "fail":
        return "fail"
    return "unknown"


def compute_stats(
    store: Store,
    *,
    since: datetime | str | None = None,
    name: str | None = None,
    limit: int | None = None,
) -> FleetStats:
    """Scan stored runs (newest first) and summarise fleet health, cost, and tools.

    ``since`` is a relative window (``\"7d\"``, ``\"2h\"``) or an ISO date / datetime,
    the same grammar as ``crashlab search --since``. ``name`` keeps one agent.
    ``limit`` is a cap on how many matching runs are read (``None`` = all).
    Runs whose files cannot be read are listed on :attr:`FleetStats.unreadable`
    and are not counted.
    """
    cutoff = _parse_since(since)
    scan = ScanStats()
    runs: list[Run] = []
    for run in iter_runs(store, since=cutoff, name=name, stats=scan):
        runs.append(run)
        if limit is not None and len(runs) >= limit:
            break

    stats = FleetStats(n=len(runs), unreadable=list(scan.unreadable))
    if not runs:
        return stats

    costs: list[float] = []
    durs: list[float] = []
    agent_runs: dict[str, list[Run]] = defaultdict(list)
    agent_costs: dict[str, list[float]] = defaultdict(list)
    agent_tokens: Counter[str] = Counter()
    agent_durs: dict[str, list[float]] = defaultdict(list)
    model_calls: Counter[str] = Counter()
    model_in: Counter[str] = Counter()
    model_out: Counter[str] = Counter()
    model_cost: dict[str, float] = defaultdict(float)
    model_cost_unknown: set[str] = set()
    model_err: Counter[str] = Counter()
    tool_calls: Counter[str] = Counter()
    tool_err: Counter[str] = Counter()
    tool_durs: dict[str, list[float]] = defaultdict(list)
    finding_n: Counter[tuple[str, str]] = Counter()
    check_fail: Counter[str] = Counter()
    check_err: Counter[str] = Counter()
    run_rows: list[RunCost] = []
    total_cost = 0.0
    any_cost = False

    for run in runs:
        bucket = _bucket(run)
        if bucket == "pass":
            stats.n_pass += 1
        elif bucket == "fail":
            stats.n_fail += 1
        elif bucket == "error":
            stats.n_error += 1
        else:
            stats.n_unknown += 1
        cost = _cost(run)
        tokens = _tokens(run)
        dur = _duration_s(run)
        stats.total_tokens += tokens
        if cost is None:
            stats.cost_unknown += 1
        else:
            stats.cost_known += 1
            costs.append(cost)
            total_cost += cost
            any_cost = True
        if dur is not None:
            stats.duration_known += 1
            durs.append(dur)
        agent_runs[run.name].append(run)
        if cost is not None:
            agent_costs[run.name].append(cost)
        agent_tokens[run.name] += tokens
        if dur is not None:
            agent_durs[run.name].append(dur)
        run_rows.append(
            RunCost(
                run_id=run.run_id,
                name=run.name,
                outcome=run.outcome,
                duration_s=_round(dur, 4),
                cost_usd=_round(cost, 6),
                tokens=tokens,
            )
        )
        for f in run.findings:
            finding_n[(f.kind, f.severity)] += 1
        for c in run.checks:
            if c.status == "fail":
                check_fail[c.name] += 1
            elif c.status == "error":
                check_err[c.name] += 1
        for step in run.steps:
            if step.operation == "chat":
                model = _step_model(step) or "unknown"
                model_calls[model] += 1
                a = step.attributes or {}
                model_in[model] += _as_int(a.get("gen_ai.usage.input_tokens"))
                model_out[model] += _as_int(a.get("gen_ai.usage.output_tokens"))
                usd = _step_usd(step)
                if usd is None:
                    model_cost_unknown.add(model)
                else:
                    model_cost[model] += usd
                if step.error is not None:
                    model_err[model] += 1
            elif step.operation == "execute_tool":
                tool_calls[step.name] += 1
                if step.error is not None:
                    tool_err[step.name] += 1
                tool_durs[step.name].append(max(step.duration_ms, 0.0) / 1000.0)

    n = stats.n
    stats.pass_rate = stats.n_pass / n
    stats.fail_rate = stats.n_fail / n
    stats.error_rate = stats.n_error / n
    stats.unknown_rate = stats.n_unknown / n
    stats.total_cost_usd = (
        round(total_cost, 6) if any_cost else (None if stats.cost_unknown else 0.0)
    )
    # if we saw no priced calls at all (and no unknown either): total is 0
    if not any_cost and stats.cost_unknown == 0:
        stats.total_cost_usd = 0.0
    stats.mean_cost_usd = _round(_mean(costs))
    stats.median_cost_usd = _round(_median(costs))
    stats.p95_cost_usd = _round(_pct(costs, 95))
    stats.mean_duration_s = _round(_mean(durs), 4)
    stats.p95_duration_s = _round(_pct(durs, 95), 4)

    for agent, items in sorted(agent_runs.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        counts = Counter(_bucket(r) for r in items)
        ac = agent_costs.get(agent) or []
        total = sum(ac) if ac else None
        stats.by_agent.append(
            AgentBreakdown(
                name=agent,
                runs=len(items),
                pass_n=counts.get("pass", 0),
                fail_n=counts.get("fail", 0),
                error_n=counts.get("error", 0),
                unknown_n=counts.get("unknown", 0),
                total_cost_usd=_round(total),
                total_tokens=int(agent_tokens[agent]),
                mean_duration_s=_round(_mean(agent_durs.get(agent) or []), 4),
            )
        )

    for model, calls in sorted(model_calls.items(), key=lambda kv: (-kv[1], kv[0])):
        known = model not in model_cost_unknown
        # still report the known portion of cost when some calls were unpriced
        raw_cost = model_cost.get(model)
        if model in model_cost_unknown and raw_cost is None:
            usd = None
        else:
            usd = _round(raw_cost or 0.0)
            if model in model_cost_unknown:
                # some calls unpriced: keep the known sum (not a complete total)
                usd = _round(raw_cost or 0.0)
        stats.by_model.append(
            ModelBreakdown(
                model=model,
                calls=calls,
                input_tokens=int(model_in[model]),
                output_tokens=int(model_out[model]),
                total_cost_usd=usd if known or raw_cost else None,
                error_n=int(model_err[model]),
            )
        )

    for tool, calls in sorted(tool_calls.items(), key=lambda kv: (-kv[1], kv[0])):
        err = int(tool_err[tool])
        stats.tools.append(
            ToolStat(
                name=tool,
                calls=calls,
                error_n=err,
                error_rate=(err / calls) if calls else 0.0,
                p95_duration_s=_round(_pct(tool_durs.get(tool) or [], 95), 4),
            )
        )
    # leaderboard already sorted by calls; also expose high-error tools near the top
    # (stable: calls desc, then name). Fine.

    stats.findings = [
        FindingStat(kind=k, severity=sev, count=c)
        for (k, sev), c in sorted(finding_n.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    ]
    check_names = list(dict.fromkeys([*check_fail, *check_err]))
    stats.failing_checks = sorted(
        [
            CheckStat(name=n_, fail_n=int(check_fail[n_]), error_n=int(check_err[n_]))
            for n_ in check_names
        ],
        key=lambda c: (-(c.fail_n + c.error_n), c.name),
    )

    stats.slowest = sorted(
        [r for r in run_rows if r.duration_s is not None],
        key=lambda r: (-(r.duration_s or 0.0), r.run_id),
    )[:_TOP_N]
    stats.most_expensive = sorted(
        [r for r in run_rows if r.cost_usd is not None],
        key=lambda r: (-(r.cost_usd or 0.0), r.run_id),
    )[:_TOP_N]
    return stats
