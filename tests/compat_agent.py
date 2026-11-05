"""Agent used to record the crashlab 0.0.2 compatibility fixtures
(tests/fixtures/v0_0_2/); keep its behaviour unchanged or re-record them."""

from __future__ import annotations

import contextlib

import crashlab
from tests.fakes import FakeOpenAI, text_response, tool_call_response


class OutOfStock(Exception):
    pass


@crashlab.tool()
def get_stock(sku: str) -> dict:
    if sku == "B-2":
        raise OutOfStock(f"{sku} is out of stock")
    return {"sku": sku, "left": 4}


@crashlab.tool(side_effects=True)
def place_order(sku: str, quantity: int) -> dict:
    return {"order_id": "o1", "sku": sku, "quantity": quantity}


def client():
    return crashlab.wrap(
        FakeOpenAI(
            [
                tool_call_response(1, "get_stock", {"sku": "A-1"}),
                text_response(2, "ordered 2 A-1; B-2 is out of stock"),
            ]
        )
    )


@crashlab.agent(name="compat-shop")
def agent(msg: str) -> str:
    c = client()
    msgs = [{"role": "system", "content": "shop"}, {"role": "user", "content": msg}]
    c.chat.completions.create(model="fake-model", messages=msgs)
    get_stock("A-1")
    with contextlib.suppress(OutOfStock):
        get_stock("B-2")
    place_order("A-1", 2)
    r = c.chat.completions.create(
        model="fake-model", messages=[*msgs, {"role": "user", "content": "?"}]
    )
    return r.choices[0].message.content
