"""A replayed OpenAI chat stream is chunk-for-chunk the recorded one, tool-call argument
fragments included (a generator agent that yields raw chunks reproduces)."""

from __future__ import annotations

import pytest

import crashlab

from ._transport import httpx_module, sse

openai = pytest.importorskip("openai")
httpx = httpx_module()

BASE = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 7, "model": "m",
        "system_fingerprint": "fp_1"}  # fmt: skip


def _ch(delta, finish=None):
    return {**BASE, "choices": [{"index": 0, "delta": delta, "logprobs": None,
                                 "finish_reason": finish}]}  # fmt: skip


def _call(i, *, head=False, args=None, index=True):
    tc = {"index": i} if index else {}
    fn = {}
    if head:
        tc.update({"id": f"call_{i}", "type": "function"})
        fn["name"] = f"tool{i}"
    if args is not None:
        fn["arguments"] = args
    return {**tc, "function": fn}


def _events(index=True):
    return [
        _ch({"role": "assistant", "content": "", "refusal": None}),
        _ch({"content": "Let"}),
        _ch({"content": " me check."}),
        _ch({"tool_calls": [_call(0, head=True, args="", index=index)]}),
        _ch({"tool_calls": [_call(0, args='{"c":', index=index)]}),
        _ch({"tool_calls": [_call(0, args='"P"}', index=index)]}),
        *(
            [
                _ch(
                    {
                        "tool_calls": [
                            _call(1, head=True, args='{"x"', index=index),
                            _call(0, args="", index=index),
                        ]
                    }
                ),  # fmt: skip
                _ch({"tool_calls": [_call(1, args=": 1}", index=index)]}),
            ]
            if index
            else []
        ),
        _ch({}, "tool_calls"),
        {
            **BASE,
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        },  # fmt: skip
    ]


def _groq_events():
    """Groq-shaped: per-chunk ``created``, ``x_groq`` (id first, usage last), a reasoning
    ``channel`` on deltas, role on the first chunk only, usage only inside x_groq."""
    ev = [_ch({"role": "assistant", "content": None, "channel": "analysis"})]
    ev[0]["x_groq"] = {"id": "req_1", "seed": 7}
    for n, piece in enumerate(["The", " user", " asks"]):
        ev.append({**_ch({"reasoning": piece, "channel": "analysis"}), "created": 7 + n // 2})
    ev.append({**_ch({"content": "Hi", "channel": "final"}), "created": 9})
    ev.append({**_ch({"tool_calls": [_call(0, head=True, args='{"c":"P"}')]}), "created": 9})
    last = {**_ch({}, "tool_calls"), "created": 9}
    last["x_groq"] = {"usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}}
    return [*ev, last]


def _deepseek_events():
    """usage on the finish chunk itself (no separate usage-only chunk)."""
    finish = _ch({"content": ""}, "stop")
    finish["usage"] = {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    return [_ch({"role": "assistant", "content": ""}), _ch({"content": "Hi"}),
            _ch({"content": " there"}), finish]  # fmt: skip


STATE = {"live": True, "events": _events()}


def _handle(request):
    assert STATE["live"], "replay hit the network"
    body = sse(STATE["events"], named=False, done=True)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


@crashlab.agent(name="raw-chunks")
def raw_chunks():
    http = httpx.Client(transport=httpx.MockTransport(_handle))
    client = crashlab.wrap(openai.OpenAI(api_key="sk-x", base_url="http://t/v1", http_client=http))
    stream = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True,
        stream_options={"include_usage": True},
    )  # fmt: skip
    for chunk in stream:
        yield chunk.model_dump()


@pytest.mark.parametrize(
    "events",
    [_events(True), _events(False), _groq_events(), _deepseek_events()],
    ids=["indexed", "no-index", "groq-extras", "usage-on-finish"],
)
def test_replayed_chat_stream_matches_chunk_for_chunk(crashlab_dir, events):
    STATE.update(live=True, events=events)
    live = list(raw_chunks())
    assert len(live) == len(STATE["events"])
    STATE["live"] = False
    try:
        replayed = crashlab.replay(crashlab.load("last"))
    finally:
        STATE["live"] = True
    assert replayed.status == "ok"
    assert replayed.output == live


def test_litellm_stream_replays_chunk_for_chunk(crashlab_dir, monkeypatch):
    # LiteLLM's chunk objects keep keys sent as null (system_fingerprint, citations,
    # delta.provider_specific_fields): replay must send them the same way
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm = pytest.importorskip("litellm")
    wrapped = crashlab.wrap(litellm)

    @crashlab.agent(name="litellm-chunks")
    def litellm_chunks():
        for chunk in wrapped.completion(
            model="gpt-4o-mini", mock_response="Hello world, this is a mocked reply.",
            messages=[{"role": "user", "content": "hi"}], stream=True,
        ):  # fmt: skip
            yield chunk.model_dump()

    live = list(litellm_chunks())
    assert len(live) > 3
    replayed = crashlab.replay(crashlab.load("last"), fn=litellm_chunks)
    assert replayed.status == "ok" and replayed.output == live
