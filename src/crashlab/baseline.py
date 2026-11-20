"""CI regression baselines: snapshot fleet metrics and compare later runs against them.

``crashlab baseline save`` writes pass rate, error rate, p95 cost and latency,
per-agent pass rates and the tool distribution to a JSON file (default
``.crashlab/baseline.json``). ``crashlab baseline check`` compares the current
store against that snapshot and reports regressions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._version import __version__
from .stats import FleetStats, compute_stats
from .store import Store

__all__ = [
    "BaselineError",
    "BaselineRegression",
    "BaselineResult",
    "compare_baseline",
    "save_baseline",
]

_SCHEMA_VERSION = 1
_DEFAULT_NAME = "baseline.json"


class BaselineError(ValueError):
    """The baseline file is missing, unreadable, or not a crashlab baseline."""


@dataclass
class BaselineRegression:
    """One metric that moved past its allowed threshold."""

    metric: str
    message: str
    baseline: float | None
    current: float | None
    delta: float | None = None
    ratio: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineResult:
    """Structured result of :func:`compare_baseline`."""

    ok: bool
    baseline_path: str
    n_baseline: int
    n_current: int
    regressions: list[BaselineRegression] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    baseline: dict[str, Any] = field(default_factory=dict)
    current: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if self.ok:
            return (
                f"no regressions vs baseline "
                f"({self.n_current} run(s) vs {self.n_baseline} in baseline)"
            )
        n = len(self.regressions)
        noun = "regression" if n == 1 else "regressions"
        return f"{n} {noun} vs baseline ({self.n_current} run(s) vs {self.n_baseline} in baseline)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "baseline_path": self.baseline_path,
            "n_baseline": self.n_baseline,
            "n_current": self.n_current,
            "summary": self.summary(),
            "regressions": [r.to_dict() for r in self.regressions],
            "comparisons": list(self.comparisons),
            "baseline": self.baseline,
            "current": self.current,
        }


def _limit(limit: int | None) -> int | None:
    if limit is None or limit == 0:
        return None
    return limit


def _agent_pass_rate(agent: dict[str, Any]) -> float:
    if "pass_rate" in agent and isinstance(agent["pass_rate"], (int, float)):
        return float(agent["pass_rate"])
    runs = int(agent.get("runs") or 0)
    return (int(agent.get("pass_n") or 0) / runs) if runs else 0.0


def metrics_from_stats(stats: FleetStats) -> dict[str, Any]:
    """Serializable fleet metrics stored in a baseline snapshot."""
    total_tool_calls = sum(t.calls for t in stats.tools)
    return {
        "n": stats.n,
        "n_pass": stats.n_pass,
        "n_fail": stats.n_fail,
        "n_error": stats.n_error,
        "n_unknown": stats.n_unknown,
        "pass_rate": stats.pass_rate,
        "fail_rate": stats.fail_rate,
        "error_rate": stats.error_rate,
        "unknown_rate": stats.unknown_rate,
        "total_cost_usd": stats.total_cost_usd,
        "p95_cost_usd": stats.p95_cost_usd,
        "mean_duration_s": stats.mean_duration_s,
        "p50_duration_s": getattr(stats, "p50_duration_s", None),
        "p95_duration_s": stats.p95_duration_s,
        "total_tokens": stats.total_tokens,
        "by_agent": [
            {
                "name": a.name,
                "runs": a.runs,
                "pass_n": a.pass_n,
                "fail_n": a.fail_n,
                "error_n": a.error_n,
                "unknown_n": a.unknown_n,
                "pass_rate": (a.pass_n / a.runs) if a.runs else 0.0,
                "total_cost_usd": a.total_cost_usd,
            }
            for a in stats.by_agent
        ],
        "tools": [
            {
                "name": t.name,
                "calls": t.calls,
                "error_n": t.error_n,
                "error_rate": t.error_rate,
                "share": (t.calls / total_tool_calls) if total_tool_calls else 0.0,
            }
            for t in stats.tools
        ],
    }


def _snapshot(
    stats: FleetStats,
    *,
    store_dir: str,
    since: datetime | str | None,
    name: str | None,
    limit: int,
) -> dict[str, Any]:
    payload = metrics_from_stats(stats)
    payload.update(
        {
            "version": _SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "crashlab_version": __version__,
            "store_dir": store_dir,
            "filters": {
                "since": None if since is None else str(since),
                "name": name,
                "limit": limit,
            },
        }
    )
    return payload


def save_baseline(
    store_dir: str | Path,
    out_path: str | Path,
    *,
    since: datetime | str | None = None,
    name: str | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    """Write a fleet-metrics snapshot for ``store_dir`` to ``out_path`` and return it."""
    root = Path(store_dir).expanduser()
    store = Store(root)
    stats = compute_stats(store, since=since, name=name, limit=_limit(limit))
    payload = _snapshot(stats, store_dir=str(root.resolve()), since=since, name=name, limit=limit)
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", "utf-8")
    return payload


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BaselineError(f"baseline file not found: {path}")
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot read baseline {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BaselineError(f"baseline {path} is not a JSON object")
    if "pass_rate" not in data:
        raise BaselineError(f"baseline {path} is missing pass_rate")
    return data


def _record(
    comparisons: list[dict[str, Any]],
    *,
    metric: str,
    baseline: float | None,
    current: float | None,
    delta: float | None = None,
    ratio: float | None = None,
    threshold: float | None = None,
    ok: bool,
    skipped: bool = False,
) -> None:
    comparisons.append(
        {
            "metric": metric,
            "baseline": baseline,
            "current": current,
            "delta": delta,
            "ratio": ratio,
            "threshold": threshold,
            "ok": ok,
            "skipped": skipped,
        }
    )


def _pass_regression(
    metric: str,
    baseline: float,
    current: float,
    min_delta: float,
) -> tuple[dict[str, Any], BaselineRegression | None]:
    delta = current - baseline
    ok = delta >= min_delta
    row = {
        "metric": metric,
        "baseline": baseline,
        "current": current,
        "delta": delta,
        "ratio": None,
        "threshold": min_delta,
        "ok": ok,
        "skipped": False,
    }
    if ok:
        return row, None
    drop = -delta
    return row, BaselineRegression(
        metric=metric,
        message=(
            f"{metric} dropped by {drop:.1%} "
            f"(from {baseline:.1%} to {current:.1%}; allowed drop {-min_delta:.1%})"
        ),
        baseline=baseline,
        current=current,
        delta=delta,
        threshold=min_delta,
    )


def _ratio_regression(
    metric: str,
    baseline: float | None,
    current: float | None,
    max_ratio: float,
    *,
    unit: str,
) -> tuple[dict[str, Any], BaselineRegression | None]:
    if baseline is None or current is None:
        row = {
            "metric": metric,
            "baseline": baseline,
            "current": current,
            "delta": None,
            "ratio": None,
            "threshold": max_ratio,
            "ok": True,
            "skipped": True,
        }
        return row, None
    if baseline == 0:
        if current <= 0:
            row = {
                "metric": metric,
                "baseline": baseline,
                "current": current,
                "delta": current - baseline,
                "ratio": None,
                "threshold": max_ratio,
                "ok": True,
                "skipped": False,
            }
            return row, None
        row = {
            "metric": metric,
            "baseline": baseline,
            "current": current,
            "delta": current - baseline,
            "ratio": None,
            "threshold": max_ratio,
            "ok": False,
            "skipped": False,
        }
        return row, BaselineRegression(
            metric=metric,
            message=f"{metric} rose from 0{unit} to {current:g}{unit} (no non-zero baseline)",
            baseline=baseline,
            current=current,
            delta=current - baseline,
            threshold=max_ratio,
        )
    ratio = current / baseline
    ok = ratio <= max_ratio
    row = {
        "metric": metric,
        "baseline": baseline,
        "current": current,
        "delta": current - baseline,
        "ratio": ratio,
        "threshold": max_ratio,
        "ok": ok,
        "skipped": False,
    }
    if ok:
        return row, None
    return row, BaselineRegression(
        metric=metric,
        message=(
            f"{metric} rose by {ratio:.2f}× "
            f"(from {baseline:g}{unit} to {current:g}{unit}; allowed {max_ratio:g}×)"
        ),
        baseline=baseline,
        current=current,
        delta=current - baseline,
        ratio=ratio,
        threshold=max_ratio,
    )


def compare_baseline(
    baseline_path: str | Path,
    store_dir: str | Path,
    *,
    min_pass_rate_delta: float = -0.05,
    max_cost_ratio: float = 1.25,
    max_latency_ratio: float = 1.5,
    since: datetime | str | None = None,
    name: str | None = None,
    limit: int = 0,
) -> BaselineResult:
    """Compare current fleet metrics in ``store_dir`` against a saved baseline.

    A regression is reported when:

    - the pass rate drops by more than ``-min_pass_rate_delta`` (default 5 points)
    - p95 cost grows by more than ``max_cost_ratio`` (default 1.25×)
    - p95 latency grows by more than ``max_latency_ratio`` (default 1.5×)
    - an agent's pass rate drops by more than the same pass-rate delta

    Missing cost or latency on either side is skipped, not treated as zero.
    """
    path = Path(baseline_path).expanduser()
    baseline = _load_baseline(path)
    root = Path(store_dir).expanduser()
    stats = compute_stats(Store(root), since=since, name=name, limit=_limit(limit))
    current = metrics_from_stats(stats)

    comparisons: list[dict[str, Any]] = []
    regressions: list[BaselineRegression] = []

    row, reg = _pass_regression(
        "pass_rate",
        float(baseline.get("pass_rate") or 0.0),
        float(current["pass_rate"]),
        min_pass_rate_delta,
    )
    comparisons.append(row)
    if reg:
        regressions.append(reg)

    row, reg = _ratio_regression(
        "p95_cost_usd",
        baseline.get("p95_cost_usd"),
        current.get("p95_cost_usd"),
        max_cost_ratio,
        unit="",
    )
    # format messages with $ when we have a regression on cost
    if reg is not None:
        b, c = reg.baseline, reg.current
        if reg.ratio is None:
            reg.message = f"p95 cost rose from ${b:g} to ${c:g} (no non-zero baseline)"
        else:
            reg.message = (
                f"p95 cost rose by {reg.ratio:.2f}× "
                f"(from ${b:g} to ${c:g}; allowed {max_cost_ratio:g}×)"
            )
    comparisons.append(row)
    if reg:
        regressions.append(reg)

    row, reg = _ratio_regression(
        "p95_duration_s",
        baseline.get("p95_duration_s"),
        current.get("p95_duration_s"),
        max_latency_ratio,
        unit="s",
    )
    comparisons.append(row)
    if reg:
        regressions.append(reg)

    base_agents = {
        a["name"]: a
        for a in baseline.get("by_agent") or []
        if isinstance(a, dict) and a.get("name")
    }
    cur_agents = {a["name"]: a for a in current.get("by_agent") or []}
    for agent_name, base in sorted(base_agents.items()):
        cur = cur_agents.get(agent_name)
        if cur is None:
            _record(
                comparisons,
                metric=f"pass_rate:{agent_name}",
                baseline=_agent_pass_rate(base),
                current=None,
                threshold=min_pass_rate_delta,
                ok=True,
                skipped=True,
            )
            continue
        row, reg = _pass_regression(
            f"pass_rate:{agent_name}",
            _agent_pass_rate(base),
            _agent_pass_rate(cur),
            min_pass_rate_delta,
        )
        comparisons.append(row)
        if reg:
            regressions.append(reg)

    return BaselineResult(
        ok=not regressions,
        baseline_path=str(path),
        n_baseline=int(baseline.get("n") or 0),
        n_current=int(current["n"]),
        regressions=regressions,
        comparisons=comparisons,
        baseline=baseline,
        current=current,
    )


def default_baseline_path(store_dir: str | Path | None = None) -> Path:
    """``<store>/baseline.json`` (the store defaults to ``$CRASHLAB_DIR`` or ``./.crashlab``)."""
    root = Path(store_dir).expanduser() if store_dir is not None else Store().root
    return Path(root) / _DEFAULT_NAME
