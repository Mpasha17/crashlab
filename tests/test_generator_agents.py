"""Generator / async-generator entry functions (streaming UIs yield chunk by chunk) record
one run spanning the iteration, with the yielded items as output, and replay exactly:
the replayed stream is re-chunked along the recorded chunk plan. They used to record
nothing (output "<generator object ...>") and every replay diverged."""

from __future__ import annotations

import asyncio

import pytest

import crashlab

from ._transport import httpx_module, sse

openai = pytest.importorskip("openai")
httpx = httpx_module()


def _chunk(delta: dict, finish=None) -> dict:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    return {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [choice]}  # fmt: skip


# live shape (Groq gpt-oss): role chunk with "", reasoning pieces, content pieces, finish,
# then a usage-only chunk without choices
CHUNKS = [
    _chunk({"role": "assistant", "content": ""}),
    _chunk({"reasoning": "Think"}),
    _chunk({"reasoning": "ing."}),
    _chunk({"content": "Hel"}),
    _chunk({"content": "lo, "}),
    _chunk({"content": "wörld 🌍"}),
    _chunk({}, "stop"),
    {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [],
     "usage": {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9}},
]  # fmt: skip
LIVE = {"on": True}


def _handle(request):
    assert LIVE["on"], "replay hit the network"
    body = sse(CHUNKS, named=False, done=True)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


async def _ahandle(request):
    return _handle(request)


def _client(is_async=False):
    kw = {"api_key": "sk-x", "base_url": "http://t/v1"}
    if is_async:
        http = httpx.AsyncClient(transport=httpx.MockTransport(_ahandle))
        return crashlab.wrap(openai.AsyncOpenAI(http_client=http, **kw))
    http = httpx.Client(transport=httpx.MockTransport(_handle))
    return crashlab.wrap(openai.OpenAI(http_client=http, **kw))


MSGS = [{"role": "user", "content": "hi"}]


@crashlab.agent(name="gen")
def gen(q: str):
    stream = _client().chat.completions.create(
        model="m", messages=[{"role": "user", "content": q}], stream=True,
        stream_options={"include_usage": True},
    )  # fmt: skip
    for c in stream:
        if c.choices:
            yield c.choices[0].delta.content


@crashlab.agent(name="agen")
async def agen(q: str):
    stream = await _client(True).chat.completions.create(
        model="m", messages=[{"role": "user", "content": q}], stream=True,
        stream_options={"include_usage": True},
    )  # fmt: skip
    async for c in stream:
        if c.choices:
            yield c.choices[0].delta.content


EXPECTED = ["", None, None, "Hel", "lo, ", "wörld 🌍", None]


def _offline_replay(run):
    LIVE["on"] = False
    try:
        return crashlab.replay(run)
    finally:
        LIVE["on"] = True


def test_sync_generator_agent_records_and_replays_chunk_for_chunk(crashlab_dir):
    assert list(gen("hi")) == EXPECTED
    run = crashlab.load("last")
    assert run.name == "gen" and run.status == "ok" and run.output == EXPECTED
    assert [s.operation for s in run.steps] == ["chat"]
    assert run.usage.total_tokens == 9
    replayed = _offline_replay(run)
    assert replayed.status == "ok" and replayed.output == EXPECTED


def test_async_generator_agent_records_and_replays(crashlab_dir):
    async def consume():
        return [t async for t in agen("hi")]

    assert asyncio.run(consume()) == EXPECTED
    run = crashlab.load("last")
    assert run.output == EXPECTED and [s.operation for s in run.steps] == ["chat"]
    assert _offline_replay(run).output == EXPECTED


def test_consumer_stopping_early_closes_the_run_and_replay_takes_as_many(crashlab_dir):
    g = gen("hi")
    assert [next(g), next(g)] == ["", None]
    g.close()
    run = crashlab.load("last")
    assert run.status == "ok" and run.output == ["", None]
    assert "generator-closed" in run.tags
    assert _offline_replay(run).output == ["", None]


def test_consumer_code_between_items_is_not_part_of_the_run(crashlab_dir, monkeypatch):
    from crashlab.capture import _chat

    monkeypatch.setattr(_chat, "_warned_outside_run", False)  # warns once per process
    other = _client()
    items = []
    with pytest.warns(UserWarning, match="outside a recorded run"):
        for item in gen("hi"):
            items.append(item)
            if len(items) == 2:  # the consumer's own call between two items
                other.chat.completions.create(model="m", messages=MSGS, stream=True).close()
    assert [s.operation for s in crashlab.load("last").steps] == ["chat"]


def test_generator_agent_error_is_recorded(crashlab_dir):
    @crashlab.agent(name="boom")
    def boom():
        yield 1
        raise ValueError("bad chunk")

    with pytest.raises(ValueError):
        list(boom())
    run = crashlab.load("last")
    assert run.status == "error" and run.output == [1] and "bad chunk" in run.error.message


def test_replayed_stream_falls_back_to_one_chunk_when_the_plan_does_not_fit():
    from crashlab.replay.synth import openai_chat_chunks

    raw = {"choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "[REDACTED]"}}]}  # fmt: skip
    plan = [[["c", 3]], [["c", 4]], [["f"]]]  # recorded before redaction changed the text
    chunks = openai_chat_chunks(raw, plan=plan)
    assert chunks[0]["choices"][0]["delta"]["content"] == "[REDACTED]"
    fitting = openai_chat_chunks(
        {"choices": [{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": "abcdefg"}}]},
        plan=plan,
    )  # fmt: skip
    assert [c["choices"][0]["delta"].get("content") for c in fitting] == ["abc", "defg", None]
