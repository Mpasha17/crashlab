from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor

import crashlab


@crashlab.tool()
def square(x: int) -> int:
    return x * x


def test_plain_thread_falls_back_to_the_only_live_run(monkeypatch):
    with crashlab.record("no-bind"):
        t = threading.Thread(target=square, args=(3,))
        t.start()
        t.join()
    assert [s.args["x"] for s in crashlab.load("last").steps] == [3]
    monkeypatch.setenv("CRASHLAB_THREAD_FALLBACK", "0")
    with crashlab.record("no-bind-off"):
        t = threading.Thread(target=square, args=(3,))
        t.start()
        t.join()
    assert crashlab.load("last").steps == []


def test_bind_context_thread():
    with crashlab.record("bind") as run:
        threads = [
            threading.Thread(target=crashlab.bind_context(square), args=(i,)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    loaded = crashlab.load(run.run_id)
    assert sorted(s.args["x"] for s in loaded.steps) == [0, 1, 2, 3, 4]
    assert sorted(s.idx for s in loaded.steps) == [0, 1, 2, 3, 4]
    assert all(s.parent_span_id == run.span_id for s in loaded.steps)


def test_same_bound_callable_concurrently():
    barrier = threading.Barrier(4)

    @crashlab.tool()
    def wait_and_square(x: int) -> int:
        barrier.wait(timeout=5)
        return x * x

    with crashlab.record("concurrent"):
        bound = crashlab.copy_context_to_thread(wait_and_square)
        with ThreadPoolExecutor(4) as pool:
            assert list(pool.map(bound, range(4))) == [0, 1, 4, 9]
    assert len(crashlab.load("last").steps) == 4


def test_context_thread_pool_executor():
    with crashlab.record("pool"), crashlab.ContextThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(square, range(6)))
        fut = pool.submit(square, 10)
        assert fut.result() == 100
    assert results == [0, 1, 4, 9, 16, 25]
    run = crashlab.load("last")
    assert run.usage.tool_calls == 7
    assert sorted(s.result for s in run.steps) == sorted([*results, 100])


def test_run_in_thread_and_exceptions():
    @crashlab.tool()
    def fails() -> None:
        raise ValueError("nope")

    with crashlab.record("rit"):
        assert crashlab.run_in_thread(square, 4) == 16
        with contextlib.suppress(ValueError):
            crashlab.run_in_thread(fails)
    steps = crashlab.load("last").steps
    assert [s.name for s in steps] == ["square", "fails"]
    assert steps[1].error.type == "ValueError"


def test_nested_spans_inside_threads():
    @crashlab.tool()
    def outer(n: int) -> int:
        return square(n) + 1

    with crashlab.record("nested-thread"), crashlab.ContextThreadPoolExecutor(2) as pool:
        list(pool.map(outer, [1, 2]))
    steps = crashlab.load("last").steps
    outers = {s.span_id: s for s in steps if s.name == "outer"}
    inners = [s for s in steps if s.name == "square"]
    assert len(outers) == 2 and len(inners) == 2
    assert all(s.parent_span_id in outers for s in inners)
    assert {outers[s.parent_span_id].args["n"] for s in inners} == {1, 2}


async def test_asyncio_to_thread_inherits_context():
    import asyncio

    async with crashlab.record("to-thread"):
        assert await asyncio.to_thread(square, 5) == 25
    assert crashlab.load("last").steps[0].result == 25
