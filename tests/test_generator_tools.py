"""Tools that yield (search result pages, file lines, progress): the step recorded
``'<generator object ...>'`` and replay handed the agent that string."""

from __future__ import annotations

import asyncio

import crashlab


@crashlab.tool()
def pages(n: int):
    for i in range(n):
        yield {"page": i}


@crashlab.tool()
async def apages(n: int):
    for i in range(n):
        await asyncio.sleep(0)
        yield i * 10


@crashlab.tool()
def failing_pages():
    yield 1
    raise RuntimeError("page 2 failed")


@crashlab.agent(name="gen-tools")
def gen_agent() -> str:
    got = [p["page"] for p in pages(3)]
    first = next(iter(pages(5)))  # only one item taken: generator closed early
    try:
        partial = list(failing_pages())
    except RuntimeError as e:
        partial = [str(e)]
    return f"{got} {first} {partial}"


def test_generator_tool_records_items_and_replays():
    out = gen_agent()
    assert out == "[0, 1, 2] {'page': 0} ['page 2 failed']"
    run = crashlab.load("last")
    assert [s.response for s in run.steps[:2]] == [
        {"result": [{"page": 0}, {"page": 1}, {"page": 2}]},
        {"result": [{"page": 0}]},
    ]
    assert run.steps[2].error.message == "page 2 failed"
    assert crashlab.replay(run).output == out


@crashlab.agent(name="agen-tools")
async def agen_agent() -> list:
    return [x async for x in apages(3)]


def test_async_generator_tool_records_items_and_replays():
    assert asyncio.run(agen_agent()) == [0, 10, 20]
    run = crashlab.load("last")
    assert run.steps[0].response == {"result": [0, 10, 20]}
    assert crashlab.replay(run).output == [0, 10, 20]


_kept: list = []


@crashlab.agent(name="gen-kept-open")
def kept_open() -> str:
    g = pages(3)
    next(g)
    _kept.append(g)  # still open when the run ends
    return "x"


def test_generator_open_at_run_end_is_stored_with_items_so_far():
    kept_open()
    run = crashlab.load("last")
    assert [s.response for s in run.steps] == [{"result": [{"page": 0}]}]
    _kept.clear()
    assert crashlab.replay(run).output == "x"
