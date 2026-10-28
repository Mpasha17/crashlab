"""A process forked inside a run inherits the run context. Its tool calls used to be
stored into the parent's run with duplicate step indexes (both processes counted from
the same number), and a replay of the parent diverged in the child, which then never
answered: the replay hung. Calls in a forked child are now outside the parent's run
(as with spawn / forkserver), with one warning; an agent called in the child records a
run of its own."""

from __future__ import annotations

import multiprocessing as mp
import sys

import pytest

import crashlab

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or "fork" not in mp.get_all_start_methods(),
    reason="needs the fork start method",
)


@crashlab.tool
def double(x: int) -> int:
    return x * 2


def _child_tool(q):
    q.put(double(5))


@crashlab.agent(name="child-agent")
def child_agent() -> int:
    return double(7)


def _child_agent(q):
    q.put(child_agent())


def _in_child(target) -> int:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q,))
    p.start()
    try:
        out = q.get(timeout=20)
    finally:
        p.join(timeout=20)
    assert p.exitcode == 0
    return out


@crashlab.agent(name="forks-tool")
def forks_tool() -> int:
    return _in_child(_child_tool) + double(1)


@crashlab.agent(name="forks-agent")
def forks_agent() -> int:
    return _in_child(_child_agent) + double(1)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork with threads (3.12+)
@pytest.mark.parametrize("agent", [forks_tool, forks_agent], ids=["tool", "agent"])
def test_forked_child_does_not_write_into_the_parent_run(crashlab_dir, agent):
    assert agent() in (12, 16)
    runs = crashlab.Store().list_runs(limit=None)
    parent = next(r for r in runs if r["name"] == agent.__name__.replace("_", "-"))
    run = crashlab.load(parent["run_id"])
    assert [(s.idx, s.name) for s in run.steps] == [(0, "double")]
    assert run.steps[0].response == {"result": 2}
    replayed = crashlab.replay(run)
    assert replayed.status == "ok"
    runs = crashlab.Store().list_runs(limit=None)
    if agent is forks_agent:  # the child's agent is a run of its own, in both processes
        assert [r["name"] for r in runs].count("child-agent") == 2
        child = crashlab.load(next(r["run_id"] for r in runs if r["name"] == "child-agent"))
        assert [s.name for s in child.steps] == ["double"]


def test_warns_once_in_the_child(crashlab_dir):
    def target(q):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            double(1)
            double(2)
        q.put([str(x.message) for x in w if "child process" in str(x.message)])

    @crashlab.agent(name="warns")
    def warns():
        return _in_child(target)

    msgs = warns()
    assert len(msgs) == 1 and "not recorded in run" in msgs[0]
