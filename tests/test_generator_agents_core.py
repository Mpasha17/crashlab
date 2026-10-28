"""Generator agents without provider SDKs (runs on every CI job, 3.10's task-context path)."""

from __future__ import annotations

import asyncio

import crashlab


@crashlab.tool
def double(x: int) -> int:
    return 2 * x


@crashlab.agent(name="tool-gen")
def tool_gen(n: int):
    for i in range(n):
        yield double(i)


@crashlab.agent(name="tool-agen")
async def tool_agen(n: int):
    for i in range(n):
        await asyncio.sleep(0)
        yield double(i)


@crashlab.agent(name="outer")
def outer() -> int:
    return sum(tool_gen(3))  # nested generator agent: steps go to the outer run


def test_sync_generator_with_tools(crashlab_dir):
    assert list(tool_gen(3)) == [0, 2, 4]
    run = crashlab.load("last")
    assert run.output == [0, 2, 4] and [s.name for s in run.steps] == ["double"] * 3
    assert crashlab.replay(run).output == [0, 2, 4]


def test_async_generator_with_tools(crashlab_dir):
    async def consume():
        return [x async for x in tool_agen(3)]

    assert asyncio.run(consume()) == [0, 2, 4]
    run = crashlab.load("last")
    assert run.output == [0, 2, 4] and len(run.steps) == 3
    assert crashlab.replay(run).output == [0, 2, 4]


def test_generator_agent_nested_in_an_agent(crashlab_dir):
    assert outer() == 6
    run = crashlab.load("last")
    assert run.name == "outer" and [s.name for s in run.steps] == ["double"] * 3
    assert crashlab.replay(run).output == 6


def test_send_and_return_value_pass_through(crashlab_dir):
    @crashlab.agent(name="echo")
    def echo():
        got = yield "ready"
        yield f"got {got}"
        return "done"

    g = echo()
    assert next(g) == "ready"
    assert g.send("x") == "got x"
    try:
        next(g)
    except StopIteration as stop:
        assert stop.value == "done"
    assert crashlab.load("last").output == ["ready", "got x"]
