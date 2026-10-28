"""Tool results JSON does not round-trip (tuples, sets, datetimes, Decimals, UUIDs, paths,
enums) come back with their type on replay, by the returned value's type or, for nested
values, the return annotation."""

from __future__ import annotations

import datetime as dt
import decimal
import enum
import pathlib
import uuid

from pydantic import BaseModel

import crashlab


class Color(enum.Enum):
    RED = "red"


class Order(BaseModel):
    id: str


VALUES = {
    "set": {1, 2},
    "frozenset": frozenset({"a"}),
    "tuple": (1, "a"),
    "datetime": dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),
    "date": dt.date(2026, 1, 1),
    "time": dt.time(9, 30),
    "timedelta": dt.timedelta(seconds=90),
    "decimal": decimal.Decimal("1.50"),
    "uuid": uuid.UUID(int=5),
    "path": pathlib.Path("/tmp/x"),
    "enum": Color.RED,
}
SEEN: list[list] = []


@crashlab.tool()
def get(kind: str):  # no annotation: the recorded type is used
    return VALUES[kind]


@crashlab.tool()
def dates() -> dict[str, dt.date]:
    return {"d": dt.date(2026, 1, 1)}


@crashlab.tool()
def orders() -> tuple[Order, ...]:
    return (Order(id="A1"), Order(id="A2"))


@crashlab.agent(name="scalar-results")
def scalar_agent() -> int:
    got = [get(k) for k in VALUES] + [dates(), orders()]
    SEEN.append(got)
    return len(got)


def test_non_json_tool_results_keep_their_type_on_replay(crashlab_dir):
    SEEN.clear()
    scalar_agent()
    replayed = crashlab.replay(crashlab.load("last"))
    assert replayed.status == "ok"
    live, again = SEEN
    for a, b in zip(live, again, strict=True):
        assert type(a) is type(b) and a == b, (a, b)
    assert again[3].tzinfo is dt.timezone.utc
    assert all(type(o) is Order for o in again[-1])
