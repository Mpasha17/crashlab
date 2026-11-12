"""Flake detection (crashlab.flakes, `crashlab flakes`): same agent + input + code
version, outcomes that differ between runs. Offline: tools only, no SDK needed."""

from __future__ import annotations

import contextlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.flakes import find_flakes, input_key, render_text
from crashlab.models import CheckResult, ErrorInfo, Finding, Run
from crashlab.verify.oracles import unregister

runner = CliRunner()
ENV = {"COLUMNS": "200", "NO_COLOR": "1"}
T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
OUTCOMES: list[str] = []


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setenv("CRASHLAB_NO_PYPROJECT", "1")
    OUTCOMES.clear()
    yield
    unregister("flaky-bot")


class Timeout(Exception):
    pass


@crashlab.tool
def fetch(q: str) -> dict:
    what = OUTCOMES.pop(0) if OUTCOMES else "ok"
    if what == "raise":
        raise Timeout("upstream timed out")
    return {"q": q, "value": 1 if what == "ok" else 2}


@crashlab.agent(name="flaky-bot")
def agent(q: str, n: int = 1) -> str:
    return f"value={fetch(q)['value']}"


def _record(*outcomes: str, q: str = "a", n: int = 1) -> None:
    for o in outcomes:
        OUTCOMES.append(o)
        with contextlib.suppress(Timeout):
            agent(q, n=n)


def test_intermittent_exception_and_check_are_reported():
    @crashlab.check("flaky-bot")
    def value_is_one(run):
        assert run.output == "value=1", f"got {run.output}"

    _record("ok", "raise", "ok", "wrong", "ok")
    _record("ok", "ok", q="b")  # a different input: consistent, not flaky
    (g,) = find_flakes()
    assert g.name == "flaky-bot" and g.input == {"q": "a", "n": 1} and len(g.runs) == 5
    sig = {s.name: s for s in g.signals}
    # the tool exception ending the run also shows up as the unhandled_tool_error finding
    assert set(sig) == {"raised", "outcome", "check:value_is_one",
                        "finding:unhandled_tool_error"}  # fmt: skip
    assert [s.name for s in g.signals][:2] == ["raised", "outcome"]
    assert (sig["raised"].bad, sig["raised"].n, sig["raised"].detail) == (1, 5,
            "tests.test_flakes.Timeout")  # fmt: skip
    # the check runs on every run (raised ones too): 2 bad of 5
    assert (sig["check:value_is_one"].bad, sig["check:value_is_one"].n) == (2, 5)
    lo, hi = sig["raised"].ci95
    assert 0 < lo < 0.2 < hi < 1  # Wilson for 1/5: about 0.036-0.62
    assert lo == pytest.approx(0.0362, abs=1e-3) and hi == pytest.approx(0.6245, abs=1e-3)
    assert g.distinct_outputs == 2  # value=1 and value=2 among the ok runs
    groups = find_flakes(include_stable=True)
    assert [x.flaky for x in groups] == [True, False]


def test_same_input_is_canonical_json_and_name_scoped():
    a = Run(run_id="A", name="x", input={"a": 1, "b": [1, 2]})
    b = Run(run_id="B", name="x", input={"b": [1, 2], "a": 1})
    c = Run(run_id="C", name="x", input={"a": 1, "b": [2, 1]})
    assert input_key(a) == input_key(b) != input_key(c)


def _run(rid: str, *, status="ok", version="abc", name="bot", inp=None, **kw) -> Run:
    err = ErrorInfo(type="ValueError", message="x") if status == "error" else None
    return Run(run_id=rid, name=name, input=inp or {"q": 1}, status=status, error=err,
               code_version=version, started_at=T0 + timedelta(seconds=int(rid[1:])),
               **kw)  # fmt: skip


def test_code_versions_are_separate_unless_asked():
    runs = [_run("r1", version="v1"), _run("r2", version="v1"),
            _run("r3", version="v2", status="error"), _run("r4", version="v2")]  # fmt: skip
    groups = find_flakes(runs=runs, include_stable=True)
    assert sorted((g.code_version, g.flaky) for g in groups) == [("v1", False), ("v2", True)]
    # a change between commits is not a flake: v1 always ok, v2 always error
    v2_err = {"version": "v2", "status": "error"}
    runs = [_run("r1", version="v1"), _run("r2", version="v1"),
            _run("r3", **v2_err), _run("r4", **v2_err)]  # fmt: skip
    assert find_flakes(runs=runs) == []
    (g,) = find_flakes(runs=runs, across_versions=True)
    assert g.code_version is None and "several code versions" in g.notes[0]


def test_replays_forks_running_runs_and_singletons_are_ignored():
    runs = [_run("r1"), _run("r2", status="error", replay_of="r1"),
            _run("r3", status="error", forked_from="r1", fork_step=0),
            _run("r4", status="running"), _run("r5", inp={"q": 2}, status="error")]  # fmt: skip
    assert find_flakes(runs=runs, include_stable=True) == []
    with pytest.raises(ValueError):
        find_flakes(runs=runs, min_runs=1)
    with pytest.raises(ValueError):
        find_flakes(runs=runs, min_severity="loud")


def test_findings_count_only_where_checks_ran_and_by_severity():
    cfg = crashlab.ChecksConfig().model_dump(mode="json")
    loop = Finding(kind="step_repetition", severity="warn", message="m")
    note = Finding(kind="budget", severity="info", message="m")
    runs = [_run("r1", check_config=cfg, findings=[loop, note]),
            _run("r2", check_config=cfg), _run("r3")]  # r3: checks did not run  # fmt: skip
    (g,) = find_flakes(runs=runs)
    (s,) = g.signals  # info-level budget note is below the default warn floor
    assert (s.name, s.bad, s.n) == ("finding:step_repetition", 1, 2)
    (g,) = find_flakes(runs=runs, min_severity="info")
    assert {x.name for x in g.signals} == {"finding:step_repetition", "finding:budget"}
    assert find_flakes(runs=runs, min_severity="error") == []
    # a disabled detector is not observed
    off = dict(cfg, step_repetition={**cfg["step_repetition"], "enabled": False})
    runs = [_run("r1", check_config=off, findings=[loop]), _run("r2", check_config=off)]
    assert find_flakes(runs=runs) == []


def test_check_error_counts_as_bad_and_only_runs_with_the_check():
    runs = [_run("r1", checks=[CheckResult(name="c", status="error", message="boom")]),
            _run("r2", checks=[CheckResult(name="c", status="pass")]),
            _run("r3")]  # fmt: skip
    (g,) = find_flakes(runs=runs)
    (s,) = g.signals
    assert (s.name, s.bad, s.n, s.detail) == ("check:c", 1, 2, "error")


def test_dirty_version_note_and_json_shape():
    runs = [_run("r1", version="abc-dirty"), _run("r2", version="abc-dirty", status="error")]
    (g,) = find_flakes(runs=runs)
    assert any("-dirty" in n for n in g.notes)
    d = json.loads(json.dumps(g.to_dict()))
    assert d["flaky"] is True and d["signals"][0]["rate"] == 0.5
    assert d["signals"][0]["bad_runs"] == ["r2"] and d["signals"][0]["good_runs"] == ["r1"]


def test_cli_flakes_text_json_and_exit_codes():
    res = runner.invoke(app, ["flakes"], env=ENV)
    assert res.exit_code == 0 and "nothing to compare" in res.output
    _record("ok", "raise", "ok")
    res = runner.invoke(app, ["flakes"], env=ENV)
    assert res.exit_code == 0, res.output
    assert "FLAKY  flaky-bot" in res.output and "raised: 1/3 runs (33%" in res.output
    assert "crashlab diff " in res.output and "1 flaky of 1 repeated input(s)" in res.output
    assert runner.invoke(app, ["flakes", "--fail-on-flaky"], env=ENV).exit_code == 1
    data = json.loads(runner.invoke(app, ["flakes", "--json"], env=ENV).output)
    assert data[0]["name"] == "flaky-bot" and data[0]["signals"][0]["name"] == "raised"
    assert runner.invoke(app, ["flakes", "--min-runs", "1"], env=ENV).exit_code == 2
    assert runner.invoke(app, ["flakes", "--name", "other", "--fail-on-flaky"],
                         env=ENV).exit_code == 0  # fmt: skip


def test_cli_stable_only_with_all():
    _record("ok", "ok")
    out = runner.invoke(app, ["flakes"], env=ENV).output
    assert "FLAKY" not in out and "0 flaky of 1 repeated input(s)" in out
    assert "stable  flaky-bot" in runner.invoke(app, ["flakes", "--all"], env=ENV).output


def test_render_text_empty():
    assert "nothing to compare" in render_text([])[0]
