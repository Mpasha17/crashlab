"""A fork whose agent catches SideEffectBlockedError and carries on (openai-agents turns
tool exceptions into 'An error occurred while running the tool', LangGraph's ToolNode
returns them to the model) looked like an ordinary run: 6 blocked place_order calls,
then a fail for an unrelated reason, and nothing said the result was not trustworthy."""

from __future__ import annotations

import crashlab

KNOB = {"qty": 2}
CALLS = {"n": 0}


@crashlab.tool(side_effects=True)
def place_order(item: str, quantity: int) -> dict:
    CALLS["n"] += 1
    return {"ok": True}


@crashlab.agent(name="swallows-blocked")
def agent() -> str:
    out = []
    for _ in range(2):
        try:
            out.append(str(place_order("widget", KNOB["qty"])))
        except Exception as e:  # frameworks report tool errors to the model
            out.append(f"tool error: {type(e).__name__}")
    return " | ".join(out)


def test_fork_notes_side_effect_calls_the_agent_swallowed():
    agent()
    src = crashlab.load("last")
    KNOB["qty"] = 5  # the fork's agent now asks for a different order
    try:
        br = crashlab.fork(src, at_step=0)
    finally:
        KNOB["qty"] = 2
    assert CALLS["n"] == 2  # only the recording ran it
    assert br.status == "ok"  # the agent swallowed the error
    notes = [f.message for f in br.flags if f.kind == "fork"]
    assert any(
        "2 side-effect call(s) were blocked" in m and "place_order (steps 0, 1)" in m
        and "fake_tool" in m
        for m in notes
    ), notes  # fmt: skip


def test_no_note_when_side_effect_calls_match_the_recording():
    agent()
    br = crashlab.fork(crashlab.load("last"), at_step=0)
    assert not [f for f in br.flags if "blocked" in f.message]


@crashlab.agent(name="raises-blocked")
def raising_agent() -> str:
    return str(place_order("widget", KNOB["qty"]))


def test_no_extra_note_when_the_single_blocked_call_ended_the_fork():
    raising_agent()
    KNOB["qty"] = 9
    try:
        br = crashlab.fork(crashlab.load("last"), at_step=0)
    finally:
        KNOB["qty"] = 2
    assert br.error.type.endswith("SideEffectBlockedError")  # says it already
    assert not [f for f in br.flags if "were blocked" in f.message]
