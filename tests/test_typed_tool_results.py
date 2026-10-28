"""Tools that return pydantic models / dataclasses get the same types back on replay."""

from __future__ import annotations

import dataclasses

from pydantic import BaseModel

import crashlab


class Order(BaseModel):
    id: str
    total: float


@dataclasses.dataclass
class Item:
    sku: str
    qty: int


@crashlab.tool()
def get_order(oid: str):  # no annotation: the recorded type is used
    return Order(id=oid, total=49.0)


@crashlab.tool()
def get_item(sku: str) -> Item:
    return Item(sku=sku, qty=3)


@crashlab.tool()
def list_orders() -> list[Order]:
    return [Order(id="A1", total=1.0), Order(id="A2", total=2.5)]


@crashlab.tool()
def plain() -> dict:
    return {"a": 1}


@crashlab.agent(name="typed-results")
def typed_agent(oid: str) -> str:
    o = get_order(oid)
    it = get_item("S1")
    total = sum(x.total for x in list_orders())
    return f"{o.id} {o.total} {it.qty} {total} {plain()['a']}"


def test_replay_returns_models_and_dataclasses():
    assert typed_agent("A17") == "A17 49.0 3 3.5 1"
    run = crashlab.load("last")
    assert run.steps[0].attributes["crashlab.result.type"].endswith(":Order")
    replayed = crashlab.replay(run)
    assert replayed.status == "ok" and replayed.output == "A17 49.0 3 3.5 1"
