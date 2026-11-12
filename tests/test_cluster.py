"""`crashlab cluster` / crashlab.cluster over hand-built fixture runs."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.cluster import (
    cluster_runs,
    failure_point,
    failure_signature,
    is_failed,
    normalize_message,
)
from crashlab.models import CheckResult, ErrorInfo, Finding
from crashlab.store import Store
from tests.debug_fixtures import chat, make_run, run_id, tool

runner = CliRunner()
ENV = {"COLUMNS": "200", "TERM": "xterm-256color", "NO_COLOR": "1"}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def invoke(args, env=None):
    res = runner.invoke(app, args, env=env or ENV)
    return res.exit_code, _ANSI.sub("", res.output)


def _timeout_run(s, n, order, secs, at_step=1, name="refund-bot"):
    """The same bug in different runs: different order ids, timings and step index."""
    steps = [chat("x", i, "hi" if i == 0 else None, "thinking") for i in range(at_step)]
    steps.append(
        tool("x", at_step, "get_order", {"order_id": order},
             error=ErrorInfo(type="TimeoutError",
                             message=f"order {order} lookup req-{n}9f8e7d6c5b timed out "
                                     f"after {secs}.5s",
                             is_tool_error=True))
    )  # fmt: skip
    return make_run(
        s, n, name=name, status="error", outcome="fail",
        error=ErrorInfo(type="TimeoutError",
                        message=f"order {order} lookup timed out after {secs}.5s"),
        steps=steps,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Order 123456 not found", "Order 998877 not found"),
        ("at 2026-11-12T10:00:00Z", "at 2026-11-13T23:59:59.123+05:30"),
        ("req 8f2a9c0d11e2 failed", "req 00ff9c0d11e9 failed"),
        ("run 01M78HD4AJPTCYMW711GZ8MSMT", "run 01M78HD5AJPTCYMW711GZ8MSMX"),
        ("bob@x.com denied", "alice@corp.example.org denied"),
        ("took 1.25s", "took 30.5s"),
        ("ORD-1017 missing", "ORD-99210 missing"),
    ],
)
def test_normalize_merges_volatile_parts(a, b):
    assert normalize_message(a) == normalize_message(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("HTTP 429 Too Many Requests", "HTTP 500 Internal Server Error"),
        ("KeyError: 'sku'", "KeyError: 'price'"),
        ("retry 3 of 5", "retry 4 of 5"),
        ("invalid api key", "insufficient quota"),
    ],
)
def test_normalize_keeps_meaningful_differences(a, b):
    assert normalize_message(a) != normalize_message(b)


def test_same_bug_clusters_across_ids_timings_and_step_index(crashlab_dir):
    s = Store()
    _timeout_run(s, 1, "ORD-1017", 30)
    _timeout_run(s, 2, "ORD-99210", 12, at_step=3)
    _timeout_run(s, 3, "ORD-3003", 7)
    make_run(s, 4)  # passing run: not clustered
    r = cluster_runs(store=s)
    assert r.scanned == 4 and r.failed == 3 and len(r.clusters) == 1
    c = r.clusters[0]
    assert c.size == 3 and c.representative == run_id(3)
    assert c.run_ids == [run_id(3), run_id(2), run_id(1)]
    assert c.signature["step"].startswith("execute_tool get_order → TimeoutError: order ")
    assert "TimeoutError" in c.summary


def test_short_ids_are_kept_so_they_split(crashlab_dir):
    # "A17" vs "C3" could be meaningful words: conservative normalization keeps them
    s = Store()
    _timeout_run(s, 1, "A17", 30)
    _timeout_run(s, 2, "C3", 30)
    assert [c.size for c in cluster_runs(store=s).clusters] == [1, 1]
    loose = cluster_runs(store=s, by=("name", "checks"))
    assert [c.size for c in loose.clusters] == [2]
    assert loose.clusters[0].summary == "failed; nothing recorded for checks"
    assert cluster_runs(store=s, by=("name",)).clusters[0].summary == "failed"


def test_different_failures_are_not_merged(crashlab_dir):
    s = Store()
    _timeout_run(s, 1, "A1", 30)
    _timeout_run(s, 2, "A2", 30, name="other-agent")  # different agent
    make_run(s, 3, status="error", outcome="fail",  # different exception message
             error=ErrorInfo(type="TimeoutError", message="model call timed out"))  # fmt: skip
    make_run(s, 4, outcome="fail",  # check failure, no exception
             checks=[CheckResult(name="refund_within_policy", status="fail")])  # fmt: skip
    make_run(s, 5, outcome="fail",  # same as 4 plus another failing check
             checks=[CheckResult(name="refund_within_policy", status="fail"),
                     CheckResult(name="no_double_refund", status="fail")])  # fmt: skip
    make_run(s, 6, outcome="fail", checks=[CheckResult(name="refund_within_policy",
             status="fail")], diagnosis_label="BAD_TOOL_ARGUMENTS")  # fmt: skip
    r = cluster_runs(store=s)
    assert r.failed == 6 and len(r.clusters) == 6 and all(c.size == 1 for c in r.clusters)


def test_looser_by_merges_and_validates(crashlab_dir):
    s = Store()
    make_run(s, 1, outcome="fail", checks=[CheckResult(name="policy", status="fail")])
    make_run(s, 2, outcome="fail", checks=[CheckResult(name="policy", status="fail")],
             diagnosis_label="BAD_TOOL_ARGUMENTS")  # fmt: skip
    assert len(cluster_runs(store=s).clusters) == 2
    r = cluster_runs(store=s, by=("name", "checks"))
    assert [c.size for c in r.clusters] == [2] and r.by == ["name", "checks"]
    with pytest.raises(ValueError, match="unknown signature component"):
        cluster_runs(store=s, by=("name", "colour"))


def test_flagged_includes_serious_findings_only(crashlab_dir):
    s = Store()
    loop = Finding(kind="step_repetition", severity="error", step=1, steps=[1], message="loop")
    warn = Finding(kind="slow_step", severity="warn", step=1, message="slow")
    make_run(s, 1, findings=[loop])
    make_run(s, 2, findings=[warn])
    assert cluster_runs(store=s).failed == 0
    r = cluster_runs(store=s, flagged=True)
    assert r.failed == 1 and r.clusters[0].signature["findings"] == ["step_repetition"]
    assert r.clusters[0].signature["step"] == "execute_tool get_order ⚑ step_repetition"


def test_sizes_order_min_size_limit_and_filters(crashlab_dir):
    s = Store()
    for n in (1, 2, 3):
        _timeout_run(s, n, f"ORD-{1000 + n}", 30)
    make_run(s, 4, outcome="fail", checks=[CheckResult(name="policy", status="fail")])
    r = cluster_runs(store=s)
    assert [c.size for c in r.clusters] == [3, 1]
    assert [c.size for c in cluster_runs(store=s, min_size=2).clusters] == [3]
    r = cluster_runs(store=s, limit=2)  # newest two runs only
    assert r.scanned == 2 and sorted(c.size for c in r.clusters) == [1, 1]
    assert cluster_runs(store=s, run_prefix=run_id(4)).failed == 1
    assert cluster_runs(store=s, name="nobody").failed == 0


def test_cluster_ids_are_stable(crashlab_dir):
    s = Store()
    _timeout_run(s, 1, "ORD-1001", 30)
    first = cluster_runs(store=s).clusters[0].cluster_id
    _timeout_run(s, 2, "ORD-9999", 3)
    assert cluster_runs(store=s).clusters[0].cluster_id == first


def test_failure_point_and_is_failed(crashlab_dir):
    s = Store()
    run = _timeout_run(s, 1, "A1", 30, at_step=2)
    run = s.load_run(run.run_id)
    fp = failure_point(run)
    assert fp.step == 2 and fp.step_kind == "error"
    assert fp.step_description == "step 2 (tool get_order, called by chat step 1)"
    assert fp.run_error.startswith("TimeoutError")
    ok = s.load_run(make_run(s, 2).run_id)
    assert not is_failed(ok) and failure_point(ok).step is None
    assert failure_signature(ok, ("name",)) == {"name": "refund-bot"}


def test_old_store_runs_without_steps_or_checks(crashlab_dir):
    s = Store()
    make_run(s, 1, status="error", outcome="unknown", steps=[],
             error=ErrorInfo(type="ValueError", message="bad"))  # fmt: skip
    r = cluster_runs(store=s)
    assert r.clusters[0].signature["step"] is None
    assert r.clusters[0].summary == "ValueError: bad"


def test_cli_table_and_json(crashlab_dir):
    s = Store()
    for n in (1, 2, 3):
        _timeout_run(s, n, f"ORD-{1000 + n}", 30)
    make_run(s, 4, outcome="fail", checks=[CheckResult(name="policy", status="fail")])
    code, out = invoke(["cluster"])
    assert code == 0, out
    assert "4 failed run(s) in 2 cluster(s)" in out and "checks failed: policy" in out
    assert "TimeoutError" in out and run_id(3)[:10] in out
    assert "crashlab show" in out and "--failing" in out
    code, out = invoke(["cluster", "--json", "--by", "name,checks"])
    data = json.loads(out)
    assert data["by"] == ["name", "checks"] and [c["size"] for c in data["clusters"]] == [3, 1]
    code, out = invoke(["cluster", "--by", "nope"])
    assert code == 2 and "unknown signature component" in out
    code, out = invoke(["cluster"], env={**ENV, "COLUMNS": "80"})
    assert max(len(line) for line in out.splitlines()) <= 80


def test_cli_empty(crashlab_dir):
    code, out = invoke(["cluster"])
    assert code == 0 and "No failed runs to cluster (0 scanned)" in out
