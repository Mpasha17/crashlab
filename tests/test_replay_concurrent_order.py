"""Concurrent calls (threads) may start in another order on replay than when recorded."""

from __future__ import annotations

import pytest

import crashlab
from tests.fakes import FakeOpenAI, text_response


@crashlab.tool()
def square(x: int) -> int:
    return x * x


@crashlab.agent(name="order-agent", ignore_args=["order"])
def order_agent(xs: list, order: list | None = None) -> int:
    return sum(square(x) for x in (order or xs))


def test_identical_calls_in_another_order_still_replay():
    assert order_agent([1, 2, 3]) == 14
    run = crashlab.load("last")
    # the same three calls, started in a different order (as threads would)
    replayed = crashlab.replay(run, overrides={"order": [3, 1, 2]})
    assert replayed.status == "ok" and replayed.output == 14
    assert [s.attributes["crashlab.replay.source_step"] for s in replayed.steps] == [2, 0, 1]


def test_a_different_call_is_still_a_divergence():
    order_agent([1, 2, 3])
    run = crashlab.load("last")
    with pytest.raises(crashlab.DivergenceError, match="args"):
        crashlab.replay(run, overrides={"order": [1, 2, 4]})


def test_a_dropped_call_is_a_missing_step():
    order_agent([1, 2, 3])
    run = crashlab.load("last")
    with pytest.raises(crashlab.DivergenceError):
        crashlab.replay(run, overrides={"order": [2, 3]})


def test_threaded_llm_calls_replay():
    client = crashlab.wrap(FakeOpenAI([text_response(i, f"r{i}") for i in range(8)]))

    @crashlab.agent(name="threads-llm")
    def fan_out(n: int) -> list:
        def one(i):
            msgs = [{"role": "user", "content": f"q{i}"}]
            return (
                client.chat.completions.create(model="m", messages=msgs).choices[0].message.content
            )

        with crashlab.ContextThreadPoolExecutor(max_workers=8) as ex:
            return sorted(ex.map(one, range(n)))

    fan_out(8)
    run = crashlab.load("last")
    replayed = crashlab.replay(run, fn=fan_out)
    assert replayed.status == "ok" and len(replayed.steps) == 8
