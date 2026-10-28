"""Tools called from a worker thread that did not copy the context (smolagents CodeAgent
runs each code block in a fresh ThreadPoolExecutor) still land in the only live run."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import crashlab
from crashlab import context

INHERITS_CONTEXT = bool(getattr(sys.flags, "thread_inherit_context", False))


@crashlab.tool()
def add_one(x: int) -> int:
    return x + 1


def test_a_tool_in_a_plain_worker_thread_is_recorded(crashlab_dir):
    @crashlab.agent(name="threads")
    def agent() -> int:
        with ThreadPoolExecutor(1) as ex:
            return ex.submit(add_one, 1).result()

    assert agent() == 2
    run = crashlab.load("last")
    assert [(s.operation, s.name) for s in run.steps] == [("execute_tool", "add_one")]
    assert context._live == {}


def test_no_fallback_with_two_live_runs_or_when_disabled(crashlab_dir, monkeypatch):
    seen: list = []
    # timeouts so a run that fails to start breaks the barrier instead of hanging CI
    gate, done = threading.Barrier(3, timeout=10), threading.Event()

    @crashlab.agent(name="busy")
    def busy() -> None:
        gate.wait()
        done.wait(5)

    threads = [threading.Thread(target=busy) for _ in range(2)]
    for t in threads:
        t.start()
    gate.wait()
    seen.append(context.current())  # main thread: no run of its own, two live runs
    done.set()
    for t in threads:
        t.join()
    assert seen == [None]

    monkeypatch.setenv("CRASHLAB_THREAD_FALLBACK", "0")

    @crashlab.agent(name="off")
    def off() -> bool:
        with ThreadPoolExecutor(1) as ex:
            return ex.submit(context.current).result() is not None

    # free-threaded builds (sys.flags.thread_inherit_context) give threads the caller's
    # context, so the run is there without any fallback
    assert off() is INHERITS_CONTEXT


def test_concurrent_runs_on_a_fresh_store_both_record(crashlab_dir):
    """Two runs starting at once must not race the index schema setup
    ("database is locked" from PRAGMA journal_mode=WAL)."""
    errors: list[BaseException] = []
    start = threading.Barrier(4, timeout=10)

    @crashlab.agent(name="fresh")
    def fresh(x: int) -> int:
        return add_one(x)

    def go(x: int) -> None:
        try:
            start.wait()
            fresh(x)
        except BaseException as exc:  # pragma: no cover - only on regression
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    from crashlab.store import Store

    assert sorted(r["name"] for r in Store().index.list_runs()) == ["fresh"] * 4
