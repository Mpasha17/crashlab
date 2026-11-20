"""Candidate patch sweeps (crashlab.suite.sweep_runs / crashlab sweep). Offline only."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab import patch as P
from crashlab.cli import app
from crashlab.models import Run, Step, Usage
from crashlab.store import Store
from crashlab.suite import SweepReport, SweepResult, patch_from_flags, sweep_runs

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
ENV = {"COLUMNS": "220", "TERM": "xterm-256color", "NO_COLOR": "1"}
runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_oracles():
    """Restore the process-wide oracle registry: unregister() would drop the
    refund-bot checks imported by other tests in this session."""
    from crashlab.verify.oracles import _registry

    before = dict(_registry)
    yield
    _registry.clear()
    _registry.update(before)


def chat(run_id: str, idx: int, *, tokens: int = 50, cost: float = 0.001) -> Step:
    return Step(
        run_id=run_id,
        idx=idx,
        span_id=f"s{idx}",
        operation="chat",
        name="fake-model",
        attributes={
            "gen_ai.request.model": "fake-model",
            "gen_ai.usage.input_tokens": tokens,
            "gen_ai.usage.output_tokens": 10,
            "crashlab.usage.cost_usd": cost,
        },
        request={"messages": [{"role": "user", "content": "hi"}]},
        response={"messages": [{"role": "assistant", "content": "ok"}]},
        request_hash=f"h{idx}",
        started_at=T0 + timedelta(seconds=idx),
        duration_ms=100.0,
    )


def make_run(
    rid: str,
    *,
    name: str = "bot",
    outcome: str = "pass",
    status: str = "ok",
    tokens: int = 50,
    cost: float = 0.001,
    n_steps: int = 2,
    suffix: str = "",
) -> Run:
    # run ids must be unique and path-safe; suffix lets forks differ
    run_id = rid if not suffix else rid[:20] + suffix
    steps = [chat(run_id, i, tokens=tokens, cost=cost) for i in range(n_steps)]
    return Run(
        run_id=run_id,
        name=name,
        status=status,
        outcome=outcome,
        started_at=T0,
        ended_at=T0 + timedelta(seconds=2),
        steps=steps,
        usage=Usage(
            input_tokens=tokens * n_steps,
            output_tokens=10 * n_steps,
            total_tokens=(tokens + 10) * n_steps,
            cost_usd=cost * n_steps,
            steps=n_steps,
            llm_calls=n_steps,
            wall_time_ms=2000.0,
        ),
    )


def put(store: Store, run: Run) -> Run:
    store.begin_run(run)
    for step in run.steps:
        store.add_step(step)
    store.end_run(run)
    return store.load_run(run.run_id)


def _scripted_forks(monkeypatch, mapping, errors=None):
    errors = errors or {}

    def fake_fork(run, **_kw):
        if run.run_id in errors:
            raise errors[run.run_id]
        branch = mapping[run.run_id]
        return branch

    monkeypatch.setattr("crashlab.suite.fork", fake_fork)


def test_sweep_classifies_fixed_regressed_unchanged_and_errored(monkeypatch):
    store = Store()
    failing = put(store, make_run("SWP00000000000000000000001", outcome="fail"))
    passing = put(store, make_run("SWP00000000000000000000002", outcome="pass"))
    still_bad = put(store, make_run("SWP00000000000000000000003", outcome="fail"))
    still_good = put(store, make_run("SWP00000000000000000000004", outcome="pass"))
    broken = put(store, make_run("SWP00000000000000000000005", outcome="fail"))

    @crashlab.check("bot")
    def cap(run):
        # fail the original failing runs; the "fixed" fork will have outcome=pass
        # and a different run id, but the same name, so this check still applies.
        return True if run.outcome == "pass" else "still bad"

    # The check above uses stored outcome, which is what we set on the scripted forks.
    fixed = make_run("SWP00000000000000000000001", outcome="pass", suffix="FX", cost=0.002)
    regressed = make_run("SWP00000000000000000000002", outcome="fail", suffix="RG", cost=0.002)
    still_failing = make_run("SWP00000000000000000000003", outcome="fail", suffix="SF")
    still_passing = make_run("SWP00000000000000000000004", outcome="pass", suffix="SP")
    _scripted_forks(
        monkeypatch,
        {
            failing.run_id: fixed,
            passing.run_id: regressed,
            still_bad.run_id: still_failing,
            still_good.run_id: still_passing,
        },
        errors={broken.run_id: RuntimeError("fork exploded")},
    )
    report = sweep_runs(
        [failing, passing, still_bad, still_good, broken],
        P.SystemPromptPatch(op="append", text="Be careful."),
        store=store,
    )
    by = {r.run_id: r.verdict for r in report.results}
    assert by[failing.run_id] == "fixed"
    assert by[passing.run_id] == "regressed"
    assert by[still_bad.run_id] == "still_failing"
    assert by[still_good.run_id] == "still_passing"
    assert by[broken.run_id] == "errored"
    assert report.fixes == 1 and report.regressions == 1
    assert report.unchanged == 2 and report.errored == 1
    assert report.total == 5
    # cost rose on the fixed and regressed forks (0.002*2 vs 0.001*2)
    assert report.cost_delta_usd != 0
    assert isinstance(report.token_delta, int)
    data = report.to_dict()
    assert data["fixes"] == 1 and data["regressions"] == 1
    assert data["patch"].startswith("system prompt")


def test_sweep_markdown_table():
    report = SweepReport(
        patch_description="system prompt += never over-refund",
        results=[
            SweepResult(
                run_id="A", name="bot", verdict="fixed", baseline_status="fail",
                patched_status="pass", cost_delta_usd=-0.001, token_delta=-10,
            ),
            SweepResult(
                run_id="B", name="bot", verdict="regressed", baseline_status="pass",
                patched_status="fail", reason="check cap failed", cost_delta_usd=0.002,
                token_delta=20,
            ),
        ],
    )  # fmt: skip
    md = report.to_markdown()
    assert md.startswith("# crashlab sweep")
    assert "| run | agent | verdict | baseline | patched | cost Δ | tokens Δ |" in md
    assert "| A | bot | fixed | fail | pass |" in md
    assert "| B | bot | regressed | pass | fail |" in md
    assert "1 fixed" in md and "1 regressed" in md
    assert "## Notable runs" in md
    assert "`B`" in md and "regressed" in md
    assert "never over-refund" in md


def test_sweep_accepts_json_patch_file(tmp_path, monkeypatch):
    store = Store()
    run = put(store, make_run("SWPFILE000000000000000001", outcome="fail"))
    path = tmp_path / "fix.json"
    path.write_text(
        '{"kind": "system_prompt", "op": "append", "text": "Stay in policy."}\n',
        encoding="utf-8",
    )
    fork_called = {}

    def fake_fork(src, *, patch, **_kw):
        fork_called["patch"] = patch
        return make_run(src.run_id, outcome="pass", suffix="FX")

    monkeypatch.setattr("crashlab.suite.fork", fake_fork)

    @crashlab.check("bot")
    def cap(run):
        return True if run.outcome == "pass" else "bad"

    report = sweep_runs([run], path, store=store)
    assert report.fixes == 1
    assert "Stay in policy" in fork_called["patch"].describe()


def test_sweep_model_and_agent_are_applied(monkeypatch):
    store = Store()
    run = put(store, make_run("SWPSCOPE00000000000000001", outcome="pass"))
    seen = {}

    def fake_fork(src, *, patch, at_step, **_kw):
        seen["patch"] = patch
        seen["at_step"] = at_step
        return src

    monkeypatch.setattr("crashlab.suite.fork", fake_fork)
    sweep_runs(
        [run],
        P.SystemPromptPatch(op="append", text="Be kind."),
        store=store,
        model="gpt-4o-mini",
        agent="researcher",
        at_step=2,
    )
    assert seen["at_step"] == 2
    kinds = [p.kind for p in seen["patch"].patches]
    assert "system_prompt" in kinds and "model_params" in kinds
    assert all(getattr(p, "agent", None) == "researcher" for p in seen["patch"].patches)


def test_patch_from_flags_combines_file_and_system_prompt(tmp_path):
    path = tmp_path / "p.json"
    path.write_text('{"kind": "model_params", "model": "gpt-4o-mini"}\n', encoding="utf-8")
    patches = patch_from_flags(
        patch_file=path, system_prompt="Be careful.", model=None, agent="refund-bot"
    )
    assert any(p.kind == "system_prompt" for p in patches.patches)
    assert any(p.kind == "model_params" for p in patches.patches)
    assert all(p.agent == "refund-bot" for p in patches.patches)


def test_sweep_budget_can_turn_a_pass_into_a_regression(monkeypatch):
    store = Store()
    cheap = put(store, make_run("SWPCOST000000000000000001", outcome="pass", cost=0.001))
    expensive = make_run("SWPCOST000000000000000001", outcome="pass", suffix="EX", cost=1.0)

    def fake_fork(src, **_kw):
        return expensive

    monkeypatch.setattr("crashlab.suite.fork", fake_fork)
    report = sweep_runs([cheap], P.SystemPromptPatch(op="append", text="x"), store=store,
                        max_cost_usd=0.01)  # fmt: skip
    assert report.regressions == 1
    assert "max_cost_usd" in (report.results[0].reason or "")


def test_sweep_cli_json_markdown_and_fail_on_regression(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store()
    good = put(store, make_run("SWPCLI00000000000000000001", outcome="pass"))
    put(store, make_run("SWPCLI00000000000000000002", outcome="fail"))

    def fake_fork(src, **_kw):
        # regress the passing run, fix the failing one
        return make_run(src.run_id, outcome="fail" if src.outcome == "pass" else "pass",
                        suffix="FK")  # fmt: skip

    monkeypatch.setattr("crashlab.suite.fork", fake_fork)

    @crashlab.check("bot")
    def cap(run):
        return True if run.outcome == "pass" else "bad"

    patch_path = tmp_path / "fix.json"
    patch_path.write_text(
        '{"kind": "system_prompt", "op": "append", "text": "Stay in policy."}\n',
        encoding="utf-8",
    )
    md = tmp_path / "sweep.md"
    result = runner.invoke(
        app,
        ["sweep", "--patch", str(patch_path), "--markdown", str(md), "--json"],
        env=ENV,
    )
    assert result.exit_code == 0, result.output
    assert '"regressions": 1' in result.output
    assert '"fixes": 1' in result.output
    assert md.is_file() and "# crashlab sweep" in md.read_text("utf-8")

    result = runner.invoke(
        app,
        ["sweep", "--patch", str(patch_path), "--fail-on-regression"],
        env=ENV,
    )
    assert result.exit_code == 1, result.output
    assert "regressed" in result.output

    result = runner.invoke(app, ["sweep", good.run_id, "--system-prompt", "Be careful."], env=ENV)
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["sweep", "--name", "bot"], env=ENV)
    assert result.exit_code == 2  # no patch / system-prompt / model
    assert "--patch" in result.output

    result = runner.invoke(app, ["sweep", "NOPE", "--system-prompt", "x"], env=ENV)
    assert result.exit_code == 2
    result = runner.invoke(app, ["sweep", "--patch", "missing.json"], env=ENV)
    assert result.exit_code == 2
    # empty store of another name
    result = runner.invoke(app, ["sweep", "--system-prompt", "x", "--name", "nope"], env=ENV)
    assert result.exit_code == 2


def test_sweep_cli_help_lists_flags():
    result = runner.invoke(app, ["sweep", "--help"], env=ENV)
    assert result.exit_code == 0
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for flag in ("--patch", "--system-prompt", "--at-step", "--agent", "--model",
                 "--fail-on-regression", "--markdown", "--json"):  # fmt: skip
        assert flag in out
