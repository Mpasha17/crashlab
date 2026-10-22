"""Tools called from a worker thread that did not copy the context (smolagents CodeAgent
runs each code block in a fresh ThreadPoolExecutor) still land in the only live run."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import crashlab
from crashlab import context


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
    gate, done = threading.Barrier(3), threading.Event()

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
    def off() -> object:
        with ThreadPoolExecutor(1) as ex:
            return ex.submit(context.current).result()

    assert off() is None
