"""Generator agents driven with send() / throw() replay the same way: the consumer's
sent values and thrown-in exceptions are recorded per item and re-applied."""

from __future__ import annotations

import asyncio

import pytest

import crashlab


@crashlab.tool()
def step(n: int) -> int:
    return n * 10


@crashlab.agent(name="gen-send")
def accumulate():
    total = 0
    while total < 10:
        got = yield step(total)
        total += got or 1


@crashlab.agent(name="gen-throw")
def recovering():
    i = 0
    while i < 3:
        try:
            yield step(i)
        except KeyError:
            yield "recovered"
        i += 1


@crashlab.agent(name="agen-send")
async def aaccumulate():
    total = 0
    while total < 10:
        got = yield step(total)
        total += got or 1
        await asyncio.sleep(0)


def _drive_send(gen):
    out = [next(gen)]
    try:
        while True:
            out.append(gen.send(4))
    except StopIteration:
        return out


async def _adrive_send(gen):
    out = [await gen.__anext__()]
    try:
        while True:
            out.append(await gen.asend(4))
    except StopAsyncIteration:
        return out


def _replayed_output():
    replayed = crashlab.replay(crashlab.load("last"))
    assert replayed.status == "ok"
    return replayed.output


def test_sent_values_are_replayed(crashlab_dir):
    live = _drive_send(accumulate())
    assert live == [0, 40, 80]
    assert crashlab.load("last").env["generator"]["actions"][0][:2] == [1, "send"]
    assert _replayed_output() == live


def test_async_sent_values_are_replayed(crashlab_dir):
    live = asyncio.run(_adrive_send(aaccumulate()))
    assert live == [0, 40, 80]
    assert _replayed_output() == live


def test_exception_thrown_in_and_handled_is_replayed(crashlab_dir):
    gen = recovering()
    live = [next(gen), gen.throw(KeyError("stop")), *gen]
    assert live == [0, "recovered", 10, 20]
    assert _replayed_output() == live


def test_exception_thrown_in_and_unhandled_ends_the_replay_the_same(crashlab_dir):
    @crashlab.agent(name="gen-unhandled")
    def unhandled():
        for i in range(3):
            yield step(i)

    gen = unhandled()
    next(gen)
    with pytest.raises(ValueError):
        gen.throw(ValueError("consumer gave up"))
    run = crashlab.load("last")
    assert run.status == "error" and run.output == [0]
    replayed = crashlab.replay(run, fn=unhandled)
    assert replayed.status == "error" and replayed.output == [0]
    assert len(replayed.steps) == 1
