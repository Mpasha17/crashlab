"""Long agent runs: each step stores the whole conversation so far, so the work per step
grows with the run. Bounded regression checks (generous: CI machines are slower)."""

from __future__ import annotations

import time

import crashlab
from crashlab.store.redact import Redactor
from tests.fakes import FakeOpenAI, text_response, tool_call_response

TURNS = 120
TOOLS = [{"type": "function", "function": {"name": "lookup",
          "parameters": {"type": "object", "properties": {"i": {"type": "integer"}}}}}]  # fmt: skip


@crashlab.tool()
def lookup(i: int) -> str:
    return "result text " * 20 + str(i)


@crashlab.agent(name="long-run")
def long_agent() -> str:
    replies = [tool_call_response(i, "lookup", {"i": i}) for i in range(TURNS)]
    client = crashlab.wrap(FakeOpenAI([*replies, text_response(TURNS, "done")]))
    msgs = [{"role": "system", "content": "sys " * 200}, {"role": "user", "content": "go"}]
    for i in range(TURNS + 1):
        r = client.chat.completions.create(model="m", messages=msgs, tools=TOOLS)
        m = r.choices[0].message
        if not m.tool_calls:
            return m.content
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [tc.model_dump() for tc in m.tool_calls]})  # fmt: skip
        msgs.append({"role": "tool", "tool_call_id": m.tool_calls[0].id, "content": lookup(i)})
    return "unfinished"


def test_long_run_records_and_replays_in_bounded_time(crashlab_dir):
    t = time.perf_counter()
    assert long_agent() == "done"
    recorded = time.perf_counter() - t
    run = crashlab.load("last")
    assert len(run.steps) == 2 * TURNS + 1
    t = time.perf_counter()
    crashlab.replay(run)
    replayed = time.perf_counter() - t
    # ~1.2 s / ~1.4 s locally (was ~1.6 s record before the key memo + synchronous=NORMAL)
    # CI 3.10 has hit ~9s under load; local ~3s. Size assertion below is the real bound.
    assert recorded < 15 and replayed < 15, (recorded, replayed)
    size = next(crashlab_dir.rglob("cassette.jsonl")).stat().st_size
    assert size < 9_000_000, size  # 7.2 MB: request + raw_request per step; no new copies


def test_secret_key_names_checked_once_per_distinct_key():
    calls = []

    class Counting:
        def search(self, key):
            calls.append(key)
            return None

    r = Redactor()
    r.key_patterns.append(Counting())  # type: ignore[arg-type]
    r._key_cache = {}
    r.redact({"messages": [{"role": "user", "content": "x", "name": "n"}] * 5000})
    assert sorted(calls) == ["content", "messages", "name", "role"]
    r.add_key("role")  # a new rule must not be answered from the old cache
    assert r.redact({"role": "user"}) == {"role": "[REDACTED]"}
