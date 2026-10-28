"""Async agents under an event loop other than asyncio (trio, curio, a hand-rolled
driver). An async generator agent raised "RuntimeError: no running event loop" on its
first item (each step ran as an asyncio task), and crashlab.replay() called from trio
code started an asyncio loop in trio's thread: TrioInternalError."""

from __future__ import annotations

import types

import pytest

import crashlab


@types.coroutine
def _suspend():
    yield "tick"  # what a foreign event loop sees: an opaque yielded value


@crashlab.tool
async def fetch(i: int) -> int:
    await _suspend()
    return i * 10


@crashlab.agent(name="foreign-gen")
async def gen():
    for i in range(3):
        yield await fetch(i)


async def _consume():
    return [x async for x in gen()]


def _drive(coro):
    """A minimal event loop that is not asyncio: resume the coroutine until it returns."""
    ticks = 0
    try:
        while True:
            assert coro.send(None) == "tick"
            ticks += 1
    except StopIteration as stop:
        return stop.value, ticks


def test_async_generator_agent_without_asyncio_loop(crashlab_dir):
    out, ticks = _drive(_consume())
    assert out == [0, 10, 20] and ticks == 3
    run = crashlab.load("last")
    assert run.status == "ok" and [s.name for s in run.steps] == ["fetch"] * 3
    assert crashlab.replay(run).output == [0, 10, 20]


def test_cancellation_reaches_the_generator(crashlab_dir):
    coro = _consume()
    assert coro.send(None) == "tick"
    with pytest.raises(KeyError):
        coro.throw(KeyError("cancelled by the loop"))
    run = crashlab.load("last")
    assert run.status == "error" and run.error.type.endswith("KeyError")


def test_trio_generator_agent_and_replay_inside_trio(crashlab_dir):
    trio = pytest.importorskip("trio")

    @crashlab.tool
    async def tick(i: int) -> int:
        await trio.sleep(0)
        return i

    @crashlab.agent(name="trio-gen")
    async def tgen():
        for i in range(3):
            yield await tick(i)

    async def main():
        return [x async for x in tgen()]

    assert trio.run(main) == [0, 1, 2]
    run = crashlab.load("last")

    async def replay_in_trio():
        return crashlab.replay(run, fn=tgen)

    replayed = trio.run(replay_in_trio)
    assert replayed.status == "ok" and replayed.output == [0, 1, 2]
