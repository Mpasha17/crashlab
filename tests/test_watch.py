"""`crashlab watch` / crashlab.watch.watch_runs: deterministic, with a fake sleep that
records runs between polls."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.models import CheckResult, ErrorInfo, Run
from crashlab.store import Store
from crashlab.watch import watch_runs
from tests.debug_fixtures import make_run, run_id, tool

runner = CliRunner()
ENV = {"COLUMNS": "200", "TERM": "xterm-256color", "NO_COLOR": "1"}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _script(store, actions):
    """A sleep() that runs the next action each time watch_runs waits."""
    queue = list(actions)

    def sleep(_seconds):
        if queue:
            queue.pop(0)(store)

    return sleep


def _failing(n):
    return lambda s: make_run(
        s, n, status="error", outcome="fail",
        error=ErrorInfo(type="TimeoutError", message="slow"),
        checks=[CheckResult(name="policy", status="fail")],
        steps=[tool("x", 0, "get_order", {}, error=ErrorInfo(type="TimeoutError", message="slow"))],
    )  # fmt: skip


def _begin(n):
    def act(s):
        s.begin_run(Run(run_id=run_id(n), name="refund-bot", status="running"))

    return act


def _finish(n):
    def act(s):
        run = Run(run_id=run_id(n), name="refund-bot", status="ok", outcome="pass")
        s.end_run(run)

    return act


def test_new_runs_only_by_default_then_started_and_finished(crashlab_dir):
    s = Store()
    make_run(s, 1)  # existing: not reported
    events = list(watch_runs(
        s, polls=4, sleep=_script(s, [_begin(2), _finish(2), _failing(3)])
    ))  # fmt: skip
    assert [(e.kind, e.run_id) for e in events] == [
        ("started", run_id(2)),
        ("finished", run_id(2)),
        ("finished", run_id(3)),
    ]
    bad = events[-1]
    assert bad.failed and bad.status == "error" and bad.steps == 1
    assert bad.failure == [
        "first error: step 0 (tool get_order)",
        "raised TimeoutError",
        "checks failed: policy",
    ]
    assert not events[1].failed and events[1].failure == []


def test_existing_and_failed_only_and_name(crashlab_dir):
    s = Store()
    make_run(s, 1)
    _failing(2)(s)
    make_run(s, 3, name="other")
    assert [e.run_id for e in watch_runs(s, polls=1, include_existing=True)] == [
        run_id(1), run_id(2), run_id(3)
    ]  # fmt: skip
    got = watch_runs(s, polls=1, include_existing=True, failed_only=True)
    assert [e.run_id for e in got] == [run_id(2)]
    assert [e.run_id for e in watch_runs(s, polls=1, include_existing=True, name="other")] == [
        run_id(3)
    ]


def test_a_run_running_when_watch_starts_is_reported_when_it_finishes(crashlab_dir):
    s = Store()
    _begin(1)(s)
    events = list(watch_runs(s, polls=2, sleep=_script(s, [_finish(1)])))
    assert [(e.kind, e.run_id) for e in events] == [("finished", run_id(1))]


def test_each_run_reported_once_and_sleep_interval(crashlab_dir):
    s = Store()
    waits = []

    def sleep(sec):
        waits.append(sec)
        if len(waits) == 1:
            make_run(s, 1)

    events = list(watch_runs(s, polls=4, interval=0.25, sleep=sleep))
    assert [e.run_id for e in events] == [run_id(1)] and waits == [0.25, 0.25, 0.25]


def test_store_created_after_watch_starts(crashlab_dir):
    s = Store()
    assert not s.index.exists()
    events = list(watch_runs(s, polls=2, sleep=_script(s, [lambda st: make_run(st, 1)])))
    assert [e.run_id for e in events] == [run_id(1)]


def test_cli_json_and_text(crashlab_dir):
    s = Store()
    make_run(s, 1)
    _failing(2)(s)
    res = runner.invoke(app, ["watch", "--existing", "--timeout", "0.01", "--interval",
                              "0.01", "--json"], env=ENV)  # fmt: skip
    assert res.exit_code == 0, res.output
    lines = [json.loads(ln) for ln in res.output.splitlines()]
    assert [(d["kind"], d["run_id"]) for d in lines] == [
        ("finished", run_id(1)), ("finished", run_id(2))
    ]  # fmt: skip
    res = runner.invoke(app, ["watch", "--existing", "--failed", "--timeout", "0.01",
                              "--interval", "0.01"], env=ENV)  # fmt: skip
    out = _ANSI.sub("", res.output)
    assert res.exit_code == 0 and "watching" in out
    assert run_id(2) in out and run_id(1) not in out
    assert "→ first error: step 0 (tool get_order) · raised TimeoutError" in out


def test_cli_rejects_bad_interval(crashlab_dir):
    res = runner.invoke(app, ["watch", "--interval", "0"], env=ENV)
    assert res.exit_code == 2


@pytest.mark.parametrize("flag", [[], ["--json"]])
def test_cli_ctrl_c_exits_cleanly(crashlab_dir, monkeypatch, flag):
    import time

    def boom(_s):
        raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", boom)
    res = runner.invoke(app, ["watch", *flag], env=ENV)
    assert res.exit_code == 0
