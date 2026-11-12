"""Runs in other processes link to the run that started them: ``crashlab.child_env()``
(env vars for a subprocess) or ``crashlab.record(..., parent=, trace=)``; ``show`` lists
child runs, ``ls`` / ``search`` filter by trace. The link lives in
``run.env["link"]``, so the Run schema is unchanged and 0.0.3 reads the runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.store import Store

ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


@crashlab.tool
def lookup(q: str) -> str:
    return q.upper()


def test_explicit_parent_and_trace(crashlab_dir):
    with crashlab.record("orchestrator") as root:
        pass
    with crashlab.record("worker", parent=root) as child:
        pass
    with crashlab.record("sub-worker", parent=child) as grand:
        pass
    with crashlab.record("http-worker", parent=child.run_id, trace=root.run_id) as http:
        pass
    assert root.parent_run is None and root.trace_id == root.run_id
    assert "link" not in root.env  # an unlinked run is stored exactly as before
    assert (child.parent_run, child.trace_id) == (root.run_id, root.run_id)
    assert (grand.parent_run, grand.trace_id) == (child.run_id, root.run_id)
    assert (http.parent_run, http.trace_id) == (child.run_id, root.run_id)
    store = Store()
    assert [r["run_id"] for r in store.child_runs(root.run_id)] == [child.run_id]
    assert {r["run_id"] for r in store.child_runs(child.run_id)} == {grand.run_id, http.run_id}
    trace = {r["run_id"] for r in store.list_runs(limit=None, trace=root.run_id)}
    assert trace == {root.run_id, child.run_id, grand.run_id, http.run_id}
    loaded = crashlab.load(grand.run_id)
    assert (loaded.parent_run, loaded.trace_id) == (child.run_id, root.run_id)


def test_child_env_outside_and_inside_a_run(crashlab_dir):
    assert crashlab.child_env() == {}
    with crashlab.record("orchestrator") as run:
        env = crashlab.child_env()
    assert env["CRASHLAB_PARENT_RUN"] == run.run_id
    assert env["CRASHLAB_TRACE_ID"] == run.run_id
    assert os.path.samefile(env["CRASHLAB_DIR"], Store().root)


def test_real_subprocess_links_to_parent(crashlab_dir, tmp_path):
    code = textwrap.dedent("""
        import crashlab

        @crashlab.tool
        def work(n: int) -> int:
            return n * 2

        with crashlab.record("worker", input={"n": 21}) as run:
            run.output = work(21)
            run.outcome = "pass"
    """)
    src = os.path.dirname(os.path.dirname(crashlab.__file__))
    with crashlab.record("orchestrator") as parent:
        env = {**os.environ, **crashlab.child_env(),
               "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}  # fmt: skip
        env.pop("CRASHLAB_DIR_UNUSED", None)
        out = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                             capture_output=True, text=True, timeout=120)  # fmt: skip
        assert out.returncode == 0, out.stderr
        parent.output = "done"
    kids = Store().child_runs(parent.run_id)
    assert [k["name"] for k in kids] == ["worker"]
    worker = crashlab.load(kids[0]["run_id"])
    assert worker.output == 42 and worker.parent_run == parent.run_id
    assert worker.trace_id == parent.run_id
    res = CliRunner().invoke(app, ["show", parent.run_id], env=ENV)
    assert res.exit_code == 0, res.output
    assert "children" in res.output and worker.run_id in res.output and "worker" in res.output
    res = CliRunner().invoke(app, ["show", worker.run_id], env=ENV)
    assert f"parent       {parent.run_id}" in res.output or parent.run_id in res.output
    assert "trace" in res.output


def test_ls_and_search_filter_by_trace(crashlab_dir):
    with crashlab.record("orchestrator") as root:
        pass
    with crashlab.record("worker", parent=root) as child:
        pass
    with crashlab.record("unrelated"):
        pass
    res = CliRunner().invoke(app, ["ls", "--trace", child.run_id], env=ENV)  # any run of it
    assert res.exit_code == 0, res.output
    assert "orchestrator" in res.output and "worker" in res.output
    assert "unrelated" not in res.output
    res = CliRunner().invoke(app, ["search", "--trace", root.run_id[:20], "--json"], env=ENV)
    ids = {m["run_id"] for m in json.loads(res.output)["matches"]}
    assert ids == {root.run_id, child.run_id}
    res = CliRunner().invoke(app, ["ls", "--trace", "NOPE"], env=ENV)
    assert res.exit_code != 0


def test_env_link_bad_values_and_replay(crashlab_dir, monkeypatch):
    monkeypatch.setenv("CRASHLAB_PARENT_RUN", "x; rm -rf /")
    with pytest.warns(RuntimeWarning, match="CRASHLAB_PARENT_RUN"):
        with crashlab.record("w") as run:
            pass
    assert run.parent_run is None

    monkeypatch.setenv("CRASHLAB_PARENT_RUN", "01PARENT0000000000000000AA")
    monkeypatch.delenv("CRASHLAB_TRACE_ID", raising=False)

    def main():
        with crashlab.record("w") as r:
            r.output = lookup("a")

    main()
    src = crashlab.load("last")
    assert src.parent_run == "01PARENT0000000000000000AA"
    assert src.trace_id == "01PARENT0000000000000000AA"
    rep = crashlab.replay(src, fn=main)
    assert rep.output == "A" and rep.parent_run is None  # a replay links via replay_of


def test_parent_validation(crashlab_dir):
    with pytest.raises(ValueError, match="parent"):
        crashlab.record("w", parent="")
    with pytest.raises(ValueError, match="trace"):
        crashlab.record("w", trace="a b")
    with pytest.raises(ValueError, match="trace= needs parent="):
        crashlab.record("w", trace="01ROOT")
