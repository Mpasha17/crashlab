"""Store robustness: odd strings and damaged files must not crash the recorded agent."""

from __future__ import annotations

import crashlab
from crashlab.capture.normalize import request_hash


@crashlab.tool()
def fetch_half_emoji(q: str) -> str:
    return "half an emoji: \ud83d"


@crashlab.agent(name="surrogates")
def surrogate_agent(text: str) -> str:
    return fetch_half_emoji(text) + text


def test_lone_surrogates_do_not_crash_the_agent():
    out = surrogate_agent("in\udc00put")
    assert out == "half an emoji: \ud83din\udc00put"
    run = crashlab.load("last")
    assert run.status == "ok"
    assert run.input == {"text": "in\\udc00put"}
    assert run.steps[0].result == "half an emoji: \\ud83d"
    assert crashlab.replay(run).status == "ok"


def test_request_hash_accepts_surrogates():
    assert len(request_hash({"messages": [{"content": "x\ud83d"}]})) == 64
