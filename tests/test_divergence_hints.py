"""Divergences caused by run-time uuids / timestamps say so, and the advice works."""

from __future__ import annotations

import datetime
import uuid

import pytest

import crashlab
from crashlab.replay.divergence import nondeterminism_hint


@crashlab.tool()
def open_ticket(ticket_id: str, at: str) -> str:
    return ticket_id


@crashlab.agent(name="fresh-ids")
def fresh_ids(q: str) -> str:
    open_ticket(ticket_id=str(uuid.uuid4()), at=datetime.datetime.now().isoformat())
    return "ok"


@crashlab.tool()
def new_ticket_id() -> str:
    return str(uuid.uuid4())


@crashlab.agent(name="ids-from-a-tool")
def ids_from_a_tool(q: str) -> str:
    open_ticket(ticket_id=new_ticket_id(), at="2026-01-01")
    return "ok"


def test_uuid_and_timestamp_divergence_gets_a_hint():
    fresh_ids("x")
    with pytest.raises(crashlab.DivergenceError) as info:
        crashlab.replay(crashlab.load("last"))
    assert "args.ticket_id" in str(info.value)
    assert "look like UUIDs / timestamps" in str(info.value)


def test_ids_made_in_a_tool_replay():
    ids_from_a_tool("x")
    assert crashlab.replay(crashlab.load("last")).status == "ok"


def test_no_hint_for_real_changes():
    diff = "--- a\n+++ b\n@@ -1 +1 @@\n-refund 2026-01-01 amount 10\n+refund 2026-01-02 amount 99"
    assert nondeterminism_hint(diff) is None
    diff = "--- a\n+++ b\n@@ -1 +1 @@\n-id 1761234567\n+id 1761234999"
    assert nondeterminism_hint(diff)
