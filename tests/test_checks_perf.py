"""Runtime checks stay linear in the number of steps on long runs."""

from __future__ import annotations

import crashlab
from crashlab.checks import detectors
from tests.fakes import FakeOpenAI, text_response, tool_call_response


def test_step_repetition_hashes_each_request_once(monkeypatch):
    calls = {"n": 0}
    real = detectors.near_identical_key

    def counting(request):
        calls["n"] += 1
        return real(request)

    monkeypatch.setattr(detectors, "near_identical_key", counting)
    n = 60
    script = [tool_call_response(i, "lookup", {"i": i}) for i in range(n)] + [
        text_response(n, "ok")
    ]
    client = crashlab.wrap(FakeOpenAI(script))

    @crashlab.tool()
    def lookup(i: int) -> int:
        return i

    msgs = [{"role": "user", "content": "go"}]
    with crashlab.record("long-run"):
        for i in range(n + 1):
            r = client.chat.completions.create(model="m", messages=list(msgs))
            m = r.choices[0].message
            if not m.tool_calls:
                break
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                tc.model_dump() for tc in m.tool_calls]})  # fmt: skip
            msgs.append({"role": "tool", "tool_call_id": m.tool_calls[0].id,
                         "content": str(lookup(i))})  # fmt: skip
    run = crashlab.load("last")
    assert len(run.steps) == 2 * n + 1
    # one hash per LLM request, not one per (request, earlier request) pair (~n^2 / 2)
    assert calls["n"] <= 2 * (n + 1), calls["n"]
