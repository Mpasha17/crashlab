"""Two top-level agent calls in different threads are two runs, never one nested in the
other through the worker-thread fallback (a threaded web server serving two requests)."""

from __future__ import annotations

import threading

import crashlab
from crashlab.store import Store


@crashlab.tool()
def echo(x: int) -> int:
    return x


A_STARTED = threading.Event()
B_DONE = threading.Event()


@crashlab.agent(name="request")
def handle_request(x: int) -> int:
    if x == 1:
        A_STARTED.set()
        assert B_DONE.wait(10)
    return echo(x)


def test_two_threads_two_runs():
    A_STARTED.clear()
    B_DONE.clear()
    errors: list[BaseException] = []

    def a():
        try:
            handle_request(1)
        except BaseException as exc:  # pragma: no cover - only on regression
            errors.append(exc)

    ta = threading.Thread(target=a)
    ta.start()
    assert A_STARTED.wait(10)

    def b():
        try:
            handle_request(2)  # while run A is live, in a thread with no bound run
        except BaseException as exc:  # pragma: no cover - only on regression
            errors.append(exc)
        finally:
            B_DONE.set()

    tb = threading.Thread(target=b)
    tb.start()
    tb.join(10)
    ta.join(10)
    assert errors == []
    runs = [Store().load_run(r["run_id"]) for r in Store().list_runs()]
    assert sorted(r.input["x"] for r in runs) == [1, 2]
    assert all([s.operation for s in r.steps] == ["execute_tool"] for r in runs)
