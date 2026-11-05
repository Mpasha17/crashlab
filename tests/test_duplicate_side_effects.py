"""A model that emits the same side-effect tool call twice in one message (a known
parallel-tool-call failure): LangGraph's ToolNode ran both, two orders were placed, and
no check said anything (step_repetition needed 3 identical calls)."""

from __future__ import annotations

import crashlab
from tests.fakes import FakeOpenAI, text_response


@crashlab.tool(side_effects=True)
def place_order(item: str, quantity: int) -> dict:
    return {"ok": True, "item": item}


@crashlab.tool()
def get_price(item: str) -> float:
    return 2.5


def _findings(kind="step_repetition"):
    return [f for f in crashlab.load("last").findings if f.kind == kind]


def test_two_identical_side_effect_calls_in_one_turn_are_an_error():
    client = crashlab.wrap(FakeOpenAI([text_response(1, "ordering"), text_response(2, "done")]))
    with crashlab.record("dup-turn"):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        place_order("widget", 2)
        place_order("widget", 2)  # same model message: a duplicate
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "y"}])
    (f,) = _findings()
    assert f.severity == "error" and f.steps == [1, 2]
    assert "succeeded 2 times" in f.message


def test_identical_side_effect_calls_in_different_turns_warn():
    client = crashlab.wrap(FakeOpenAI([text_response(i, f"t{i}") for i in range(3)]))
    with crashlab.record("dup-turns"):
        for i in range(2):
            client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": str(i)}]
            )
            place_order("widget", 2)
    (f,) = _findings()
    assert f.severity == "warn"


def test_two_identical_reads_or_different_orders_are_fine():
    with crashlab.record("reads"):
        get_price("widget")
        get_price("widget")
        place_order("widget", 2)
        place_order("widget", 3)
    assert _findings() == []
