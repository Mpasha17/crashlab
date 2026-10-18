"""Tiny agents for replay tests. Importable, so ``crashlab replay`` can resolve them.

``KNOBS`` lets a test change the agent's behaviour between record and replay to force a
specific divergence; ``CALLS`` counts real tool executions (must stay 0 during replay).
"""

from __future__ import annotations

import json
import socket
from typing import Any

import crashlab
from tests.fakes import FakeOpenAI, text_response, tool_call_response

KNOBS: dict[str, Any] = {}
CALLS: dict[str, int] = {"lookup": 0, "charge": 0}
CLIENTS: list[Any] = []  # every fake client mini() built (to inspect live fork requests)


def reset() -> None:
    KNOBS.clear()
    CALLS.update(lookup=0, charge=0)
    CLIENTS.clear()


@crashlab.tool()
def lookup(key: str) -> dict[str, Any]:
    """Read-only tool."""
    CALLS["lookup"] += 1
    if key == "missing":
        raise KeyError(f"no such key {key!r}")
    return {"key": key, "value": len(key)}


@crashlab.tool(side_effect=True)
def charge(amount: float) -> dict[str, Any]:
    """Moves money (side effects)."""
    CALLS["charge"] += 1
    return {"charged": amount}


def script() -> list[Any]:
    return [
        tool_call_response(1, "lookup", {"key": "abc"}),
        text_response(2, "done"),
    ]


def _client() -> Any:
    # A fresh fake per call: in replay it is never called (the cassette answers); in a
    # fork it answers the live calls from the fork step on.
    fake = FakeOpenAI(KNOBS.get("script") or script())
    CLIENTS.append(fake)
    return crashlab.wrap(fake)


@crashlab.agent(name="mini")
def mini(task: str) -> str:
    """chat -> lookup -> (charge) -> chat; KNOBS change the shape for divergence tests."""
    client = _client()
    system = KNOBS.get("system", "be brief")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    first = client.chat.completions.create(model="fake-model", messages=msgs)
    if KNOBS.get("crash"):
        raise RuntimeError("agent crashed")
    key = KNOBS.get("key", "abc")
    if KNOBS.get("key_from_model"):  # follow the model's tool call (fork tests)
        key = json.loads(first.choices[0].message.tool_calls[0].function.arguments)["key"]
    try:  # user code that swallows every exception (DivergenceError included)
        res = lookup(key) if not KNOBS.get("other_tool") else charge(1.0)
    except Exception as exc:
        res = {"error": f"{type(exc).__name__}: {exc}"}
    if KNOBS.get("charge"):
        charge(5.0)
    if KNOBS.get("stop_early"):
        return "stopped"
    msgs.append({"role": "tool", "tool_call_id": "call_1", "content": str(res)})
    out = client.chat.completions.create(model="fake-model", messages=msgs)
    if KNOBS.get("extra_call"):
        client.chat.completions.create(model="fake-model", messages=msgs)
    if KNOBS.get("network"):
        socket.create_connection(("192.0.2.1", 9), timeout=1)  # TEST-NET-1, never reachable
    text = out.choices[0].message.content
    return text + KNOBS.get("suffix", "")


@crashlab.agent(name="mini-async")
async def mini_async(task: str) -> str:
    from tests.fakes import AsyncFakeOpenAI

    client = crashlab.wrap(AsyncFakeOpenAI(script()))
    r = await client.chat.completions.create(model="fake-model", messages=[{"role": "user",
                                             "content": task}])  # fmt: skip
    res = lookup(r.choices[0].message.tool_calls[0].function.name)
    out = await client.chat.completions.create(
        model="fake-model", messages=[{"role": "tool", "tool_call_id": "c", "content": str(res)}]
    )
    return out.choices[0].message.content
