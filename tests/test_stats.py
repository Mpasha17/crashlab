"""Fleet analytics. Offline; no provider SDKs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.models import CheckResult, ErrorInfo, Finding
from crashlab.stats import compute_stats
from crashlab.store import Store
from tests.debug_fixtures import T0, chat, make_run, tool

runner = CliRunner()


def _tune(store: Store, run, *, cost=None, tokens=None, wall_ms=None, started=None,
          outcome=None, status=None, findings=None, checks=None, error=None):  # fmt: skip
    run = store.load_run(run.run_id)
    if cost is not None:
        run.usage.cost_usd = cost
    if tokens is not None:
        inp, out = tokens if isinstance(tokens, tuple) else (tokens, 0)
        run.usage.input_tokens = inp
        run.usage.output_tokens = out
        run.usage.total_tokens = inp + out
    if wall_ms is not None:
        run.usage.wall_time_ms = wall_ms
        if started is None:
            started = run.started_at
        run.ended_at = started + timedelta(milliseconds=wall_ms)
    if started is not None:
        run.started_at = started
        if wall_ms is not None:
            run.ended_at = started + timedelta(milliseconds=wall_ms)
    if outcome is not None:
        run.outcome = outcome
    if status is not None:
        run.status = status
    if findings is not None:
        run.findings = findings
    if checks is not None:
        run.checks = checks
    if error is not None:
        run.error = error
    store.update_run(run)
    return store.load_run(run.run_id)


def _fleet(store: Store):
    a = make_run(store, 1, name="refund-bot", outcome="pass")
    b = make_run(
        store,
        2,
        name="refund-bot",
        outcome="fail",
        checks=[CheckResult(name="refund_within_policy", status="fail", message="over")],
        findings=[Finding(kind="step_repetition", severity="warn", step=0, message="loop")],
    )
    c = make_run(
        store,
        3,
        name="researcher",
        status="error",
        outcome="fail",
        error=ErrorInfo(type="Timeout", message="deadline"),
        checks=[CheckResult(name="grounded", status="error", message="check crashed")],
    )
    # a third refund-bot with no known cost
    d = make_run(
        store,
        4,
        name="refund-bot",
        outcome="unknown",
        steps=[
            chat(
                "x",
                0,
                "hello",
                "ok",
                cost=None,
                tokens=(50, 5),
                duration_ms=40,
                model="acme/unpriced-1",
            ),
            tool("x", 1, "search", {"q": "x"}, "hits", duration_ms=200),
            tool(
                "x",
                2,
                "search",
                {"q": "y"},
                error=ErrorInfo(type="HTTPError", message="502"),
                duration_ms=15,
            ),
        ],
    )
    _tune(store, a, cost=0.010, tokens=(800, 80), wall_ms=1000, started=T0)
    _tune(store, b, cost=0.030, tokens=(2000, 200), wall_ms=3000, started=T0 + timedelta(minutes=1))
    _tune(store, c, cost=0.005, tokens=(100, 10), wall_ms=500, started=T0 + timedelta(minutes=2))
    _tune(store, d, cost=None, tokens=(50, 5), wall_ms=800, started=T0 + timedelta(minutes=3))
    # drop the stored cost on d's first step too (already cost=None) and wipe run cost
    d = store.load_run(d.run_id)
    d.usage.cost_usd = None
    store.update_run(d)
    return a, b, c, d


def test_compute_stats_empty(crashlab_dir):
    stats = compute_stats(Store())
    assert stats.n == 0 and stats.pass_rate == 0 and stats.total_tokens == 0
    assert stats.mean_cost_usd is None and stats.p95_duration_s is None
    assert stats.to_dict()["n"] == 0


def test_compute_stats_counts_cost_tools_and_filters(crashlab_dir):
    store = Store()
    a, b, c, d = _fleet(store)
    stats = compute_stats(store)
    assert stats.n == 4
    assert (stats.n_pass, stats.n_fail, stats.n_error, stats.n_unknown) == (1, 1, 1, 1)
    assert abs(stats.pass_rate + stats.fail_rate + stats.error_rate + stats.unknown_rate - 1) < 1e-9
    # costs: 0.01, 0.03, 0.005 — d is unknown
    assert stats.cost_known == 3 and stats.cost_unknown == 1
    assert stats.total_cost_usd == 0.045
    assert stats.mean_cost_usd == 0.015
    assert stats.median_cost_usd == 0.01
    assert stats.p95_cost_usd is not None and stats.p95_cost_usd >= stats.median_cost_usd
    assert stats.total_tokens == 800 + 80 + 2000 + 200 + 100 + 10 + 50 + 5
    assert stats.mean_duration_s is not None and stats.p95_duration_s is not None
    # 0.5, 0.8, 1.0, 3.0 seconds — mean 1.325, p95 near 3
    assert abs(stats.mean_duration_s - 1.325) < 0.001
    assert stats.p95_duration_s >= 2.0

    agents = {x.name: x for x in stats.by_agent}
    assert agents["refund-bot"].runs == 3
    assert agents["researcher"].error_n == 1
    models = {x.model: x for x in stats.by_model}
    assert "gpt-4o-mini" in models and "acme/unpriced-1" in models
    tools = {x.name: x for x in stats.tools}
    assert tools["get_order"].calls >= 1
    assert tools["search"].calls == 2 and tools["search"].error_n == 1
    assert tools["search"].error_rate == 0.5
    assert tools["search"].p95_duration_s is not None

    kinds = {(f.kind, f.severity): f.count for f in stats.findings}
    assert kinds[("step_repetition", "warn")] == 1
    checks = {c.name: c for c in stats.failing_checks}
    assert checks["refund_within_policy"].fail_n == 1
    assert checks["grounded"].error_n == 1

    assert stats.slowest[0].run_id == b.run_id  # 3s
    assert stats.most_expensive[0].run_id == b.run_id  # $0.03

    only = compute_stats(store, name="researcher")
    assert only.n == 1 and only.n_error == 1
    limited = compute_stats(store, limit=2)
    assert limited.n == 2  # newest first
    # since after everything
    future = compute_stats(store, since=datetime.now(timezone.utc) + timedelta(days=1))
    assert future.n == 0
    month = compute_stats(store, since="30d")
    assert month.n == 4
    # a date before T0 (2026-11-10) is still all of them; a date after the last run is none
    after = compute_stats(store, since="2026-12-01")
    assert after.n == 0


def test_cli_stats(crashlab_dir):
    _fleet(Store())
    res = runner.invoke(app, ["stats"])
    assert res.exit_code == 0, res.output
    assert "4 runs" in res.output and "pass" in res.output
    assert "refund-bot" in res.output and "search" in res.output
    res = runner.invoke(app, ["stats", "--name", "researcher", "--json"])
    assert res.exit_code == 0
    data = __import__("json").loads(res.output)
    assert data["n"] == 1 and data["n_error"] == 1
    res = runner.invoke(app, ["stats", "--since", "nope"])
    assert res.exit_code == 2
    empty = runner.invoke(app, ["--dir", str(crashlab_dir / "empty"), "stats"])
    # empty dir: no runs (store created on Store())
    assert empty.exit_code == 0
    assert "no runs" in empty.output
