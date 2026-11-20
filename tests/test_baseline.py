"""Fleet HTML reports, trajectory diffs, and CI regression baselines. Offline; no SDKs."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from crashlab.baseline import BaselineError, compare_baseline, save_baseline
from crashlab.cli import app
from crashlab.compare import diff
from crashlab.stats import (
    AgentBreakdown,
    CheckStat,
    FindingStat,
    FleetStats,
    ModelBreakdown,
    RunCost,
    ToolStat,
    compute_stats,
    render_fleet_html,
)
from crashlab.store import Store
from tests.debug_fixtures import chat, make_run, tool
from tests.test_stats import _fleet, _tune

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def test_render_fleet_html_escapes_and_is_self_contained(crashlab_dir):
    stats = FleetStats(
        n=2,
        n_pass=1,
        n_fail=0,
        n_error=1,
        pass_rate=0.5,
        error_rate=0.5,
        total_cost_usd=1.25,
        p50_duration_s=0.4,
        p95_duration_s=1.5,
        by_agent=[
            AgentBreakdown(name="<script>alert(1)</script>", runs=2, pass_n=1, error_n=1),
        ],
        by_model=[ModelBreakdown(model='<img src=x onerror="alert(1)">', calls=3, error_n=1)],
        tools=[ToolStat(name="<script>steal()</script>", calls=4, error_n=1, error_rate=0.25)],
        findings=[FindingStat(kind="<script>x</script>", severity="error", count=2)],
        failing_checks=[CheckStat(name="<b>check</b>", fail_n=1, error_n=1)],
        slowest=[
            RunCost(
                run_id="abc123",
                name="<script>run</script>",
                outcome="fail",
                duration_s=1.5,
                cost_usd=1.25,
            )
        ],
        most_expensive=[
            RunCost(
                run_id="abc123",
                name="<script>run</script>",
                outcome="fail",
                duration_s=1.5,
                cost_usd=1.25,
            )
        ],
    )
    html = render_fleet_html(stats)
    assert html.startswith("<!doctype html>")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<img src=x" not in html and "&lt;img" in html
    assert "<b>check</b>" not in html and "&lt;b&gt;check&lt;/b&gt;" in html
    # no external assets or executable tags (ignore escaped text such as "&lt;img src=x")
    assert not re.search(r"<(?:script|link|iframe|img)\b", html, re.I)
    assert "src='" not in html and 'src="' not in html
    assert "href=" not in html and "@import" not in html and "url(" not in html
    for needle in (
        "Pass rate",
        "Total cost",
        "p50 latency",
        "p95 latency",
        "Errors",
        "By agent",
        "By model",
        "Runtime findings",
        "Failing checks",
        "Slowest",
        "Most expensive",
        "50.0%",
        "$1.2500",
        "400ms",
        "1.50s",
    ):
        assert needle in html, needle

    empty = render_fleet_html(FleetStats())
    assert empty.startswith("<!doctype html>") and "No runs" in empty
    assert "<script>" not in empty


def test_render_fleet_html_escapes_optional_runs_table(crashlab_dir):
    store = Store()
    run = make_run(store, 1, name="<script>agent</script>", user="<img src=x>")
    html = render_fleet_html(compute_stats(store), runs=[store.load_run(run.run_id)])
    assert "<script>agent</script>" not in html
    assert "&lt;script&gt;agent&lt;/script&gt;" in html
    assert "<h2>Runs</h2>" in html


def test_cli_stats_html(crashlab_dir, tmp_path):
    _fleet(Store())
    out = tmp_path / "nested" / "fleet.html"
    res = runner.invoke(app, ["stats", "--html", str(out)])
    assert res.exit_code == 0, res.output
    assert out.is_file()
    html = out.read_text("utf-8")
    assert html.startswith("<!doctype html>")
    assert "refund-bot" in html and "Pass rate" in html
    assert "wrote" in _plain(res.output) and "4 runs" in res.output


def test_tool_sequence_edit_distance_and_to_dict(crashlab_dir):
    store = Store()
    a = make_run(
        store,
        1,
        steps=[
            chat("x", 0, "hi", "ok"),
            tool("x", 1, "get_order", {"id": "A"}, {"ok": True}),
            tool("x", 2, "search", {"q": "x"}, {"hits": []}),
        ],
    )
    b = make_run(
        store,
        2,
        steps=[
            chat("x", 0, "hi", "ok"),
            tool("x", 1, "get_order", {"id": "A"}, {"ok": True}),
            tool("x", 2, "refund", {"id": "A"}, {"ok": True}),
        ],
    )
    a, b = store.load_run(a.run_id), store.load_run(b.run_id)
    d = diff(a, b)
    assert d.tool_sequence_a == ["get_order", "search"]
    assert d.tool_sequence_b == ["get_order", "refund"]
    assert d.tool_edit_distance == 1
    assert abs(d.trajectory_similarity - 0.5) < 1e-9
    assert {op["op"] for op in d.tool_alignment} >= {"equal", "replace"}
    # existing summary() contract is unchanged
    assert "first divergence at step" in d.summary() or "steps are the same" in d.summary()
    payload = d.to_dict()
    assert payload["a"] == a.run_id and payload["b"] == b.run_id
    assert payload["tool_edit_distance"] == 1
    assert payload["tool_sequence_a"] == ["get_order", "search"]
    assert payload["summary"] == d.summary()
    assert "first_divergence" in payload and "steps" in payload

    same = diff(a, store.load_run(a.run_id))
    assert same.tool_edit_distance == 0 and same.trajectory_similarity == 1.0
    assert same.summary()  # still a string

    empty_a = store.load_run(make_run(store, 3, steps=[chat("x", 0, "hi", "ok")]).run_id)
    empty_b = store.load_run(make_run(store, 4, steps=[chat("x", 0, "hi", "ok")]).run_id)
    both = diff(empty_a, empty_b)
    assert both.tool_sequence_a == both.tool_sequence_b == []
    assert both.tool_edit_distance == 0 and both.trajectory_similarity == 1.0

    extra = store.load_run(
        make_run(
            store,
            5,
            steps=[
                chat("x", 0, "hi", "ok"),
                tool("x", 1, "get_order", {"id": "A"}, {"ok": True}),
                tool("x", 2, "search", {"q": "x"}, {"hits": []}),
                tool("x", 3, "refund", {"id": "A"}, {"ok": True}),
            ],
        ).run_id
    )
    inserted = diff(a, extra)
    assert inserted.tool_edit_distance == 1
    assert abs(inserted.trajectory_similarity - (1 - 1 / 3)) < 1e-9


def test_cli_diff_json_includes_trajectory(crashlab_dir):
    store = Store()
    a = make_run(store, 1)
    b = make_run(
        store,
        2,
        steps=[
            chat("x", 0, "hi", tool_call=("search", {"q": "x"})),
            tool("x", 1, "search", {"q": "x"}, {"hits": []}),
            chat("x", 2, None, "done"),
        ],
    )
    res = runner.invoke(app, ["diff", a.run_id, b.run_id, "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["a"] == a.run_id and data["b"] == b.run_id
    assert "first_divergence" in data and "summary" in data
    assert data["tool_sequence_a"] == ["get_order"]
    assert data["tool_sequence_b"] == ["search"]
    assert data["tool_edit_distance"] == 1
    assert data["trajectory_similarity"] == 0.0
    assert isinstance(data["tool_alignment"], list)


def test_save_and_check_baseline_pass(crashlab_dir, tmp_path):
    store = Store()
    _fleet(store)
    path = tmp_path / "baseline.json"
    payload = save_baseline(crashlab_dir, path)
    assert path.is_file()
    assert payload["n"] == 4
    assert payload["pass_rate"] == 0.25
    assert "p95_cost_usd" in payload and "p95_duration_s" in payload
    assert {a["name"] for a in payload["by_agent"]} >= {"refund-bot", "researcher"}
    assert any(t["name"] == "search" for t in payload["tools"])
    stats = compute_stats(store)
    assert stats.p50_duration_s is not None
    assert abs(stats.p50_duration_s - 0.9) < 0.001

    result = compare_baseline(path, crashlab_dir)
    assert result.ok
    assert result.n_current == 4
    assert result.regressions == []
    assert "no regressions" in result.summary()
    dumped = result.to_dict()
    assert dumped["ok"] is True and dumped["summary"] == result.summary()


def test_compare_baseline_detects_pass_cost_and_latency_regressions(crashlab_dir, tmp_path):
    store = Store()
    a, b, c, d = _fleet(store)
    path = tmp_path / "baseline.json"
    save_baseline(crashlab_dir, path)

    # turn the only passing run into a failure: pass rate 0.25 -> 0
    _tune(store, a, outcome="fail")
    dropped = compare_baseline(path, crashlab_dir)
    assert not dropped.ok
    assert any(r.metric == "pass_rate" for r in dropped.regressions)
    # restore and blow the cost budget (p95 was ~0.03)
    _tune(store, a, outcome="pass")
    _tune(store, b, cost=1.0)
    costly = compare_baseline(path, crashlab_dir)
    assert not costly.ok
    assert any(r.metric == "p95_cost_usd" for r in costly.regressions)
    # restore cost, blow latency (p95 was ~3s)
    _tune(store, b, cost=0.030, wall_ms=20_000)
    slow = compare_baseline(path, crashlab_dir)
    assert not slow.ok
    assert any(r.metric == "p95_duration_s" for r in slow.regressions)

    # a 4-point drop stays within the default -0.05 budget
    _tune(store, b, cost=0.030, wall_ms=3000)
    # 1 pass / 4 runs = 0.25; adding one unknown keeps pass rate at 1/5=0.20 (delta -0.05, allowed)
    make_run(store, 99, name="other-bot", outcome="unknown")
    tight = compare_baseline(path, crashlab_dir)
    assert tight.ok, [r.message for r in tight.regressions]


def test_compare_baseline_missing_or_invalid(crashlab_dir, tmp_path):
    missing = tmp_path / "nope.json"
    try:
        compare_baseline(missing, crashlab_dir)
        raise AssertionError("expected BaselineError")
    except BaselineError as exc:
        assert "not found" in str(exc)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", "utf-8")
    try:
        compare_baseline(bad, crashlab_dir)
        raise AssertionError("expected BaselineError")
    except BaselineError:
        pass
    empty = tmp_path / "empty.json"
    empty.write_text("[]", "utf-8")
    try:
        compare_baseline(empty, crashlab_dir)
        raise AssertionError("expected BaselineError")
    except BaselineError:
        pass


def test_cli_baseline_save_check_json_and_fail_on_regression(crashlab_dir, tmp_path):
    store = Store()
    a, b, c, d = _fleet(store)
    path = tmp_path / "base.json"
    saved = runner.invoke(app, ["baseline", "save", "--out", str(path)])
    assert saved.exit_code == 0, saved.output
    assert "wrote" in _plain(saved.output) and path.is_file()

    ok = runner.invoke(app, ["baseline", "check", str(path), "--json"])
    assert ok.exit_code == 0, ok.output
    data = json.loads(ok.output)
    assert data["ok"] is True and data["n_current"] == 4

    _tune(store, a, outcome="fail")
    shown = runner.invoke(app, ["baseline", "check", str(path)])
    assert shown.exit_code == 0, shown.output
    assert "regression" in _plain(shown.output).lower()

    failed = runner.invoke(app, ["baseline", "check", str(path), "--fail-on-regression"])
    assert failed.exit_code == 1, failed.output

    missing = runner.invoke(app, ["baseline", "check", str(tmp_path / "missing.json")])
    assert missing.exit_code == 2
    bad_since = runner.invoke(app, ["baseline", "save", "--since", "nope"])
    assert bad_since.exit_code == 2


def test_baseline_and_stats_help_strips_ansi():
    for args in (
        ["baseline", "--help"],
        ["baseline", "save", "--help"],
        ["baseline", "check", "--help"],
        ["stats", "--help"],
        ["diff", "--help"],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, (args, res.output)
        out = _plain(res.output)
        assert "\x1b[" not in out
        if args[0] == "baseline" and args[-1] == "--help" and len(args) == 2:
            assert "save" in out and "check" in out
        if args[:2] == ["baseline", "save"]:
            assert "--out" in out and "--since" in out and "--name" in out
        if args[:2] == ["baseline", "check"]:
            for flag in (
                "--fail-on-regression",
                "--json",
                "--min-pass-rate-delta",
                "--max-cost-ratio",
                "--max-latency-ratio",
            ):
                assert flag in out
        if args[0] == "stats":
            assert "--html" in out
        if args[0] == "diff":
            assert "--json" in out
