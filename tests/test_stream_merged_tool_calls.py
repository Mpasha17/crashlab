"""Parallel tool calls streamed under one index are merged by every OpenAI-compatible
client; crashlab records that it happened and bad_tool_args says why the call is broken.

Live finding (OpenAI Agents SDK `as_tool`, Gemini OpenAI endpoint, Runner.run_streamed):
two parallel calls of tool `research` arrived as two deltas without an index and with
different ids. The SDK (and crashlab's record) saw one call named "researchresearch" with
arguments '{"input":"python"}{"input":"rust"}'; the run then failed with a 400."""

from __future__ import annotations

import pytest

import crashlab
from crashlab.checks import ChecksConfig, evaluate
from tests._transport import Script, httpx_module, sse

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "research",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
        },
    }
]


@pytest.fixture
def openai_mod():
    return pytest.importorskip("openai")


@pytest.fixture(autouse=True)
def _no_pyproject(monkeypatch):
    monkeypatch.setenv("CRASHLAB_NO_PYPROJECT", "1")


def _client(mod, script):
    httpx = httpx_module()
    return mod.OpenAI(
        api_key="sk-test", base_url="http://localhost:11434/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(script.handler(httpx))),
    )  # fmt: skip


def _chunks(calls, index=None):
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    out = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}]
    for i, (cid, q) in enumerate(calls):
        tc = {"id": cid, "type": "function",
              "function": {"name": "research", "arguments": f'{{"input":"{q}"}}'}}  # fmt: skip
        if index is not None:
            tc["index"] = index(i)
        out.append({**base, "choices": [{"index": 0, "delta": {"tool_calls": [tc]}}]})
    out.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    return out


@crashlab.agent(name="merged-calls", ignore_args=["client"])
def ask(q: str, client=None) -> list:
    stream = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": q}], tools=TOOLS, stream=True
    )
    return [chunk.model_dump() for chunk in stream]


def _record(openai_mod, chunks):
    script = Script([(sse(chunks, named=False), "text/event-stream")])
    ask("hi", client=crashlab.wrap(_client(openai_mod, script)))
    return crashlab.load("last")


@pytest.mark.parametrize("index", [None, lambda i: 0], ids=["no-index", "same-index"])
def test_parallel_calls_under_one_index_are_flagged(openai_mod, crashlab_dir, index):
    run = _record(openai_mod, _chunks([("call_a", "python"), ("call_b", "rust")], index))
    step = run.steps[0]
    assert step.attributes["crashlab.stream.tool_calls_merged"] == 2
    # the record shows what the client saw: one merged call
    (call,) = step.response["tool_calls"]
    assert call["name"] == "researchresearch"
    (f,) = [f for f in evaluate(run, ChecksConfig()) if f.kind == "bad_tool_args"]
    assert "unknown tool 'researchresearch'" in f.message
    assert "2 parallel tool calls were streamed under one index" in f.message


def test_distinct_indexes_are_not_flagged(openai_mod, crashlab_dir):
    run = _record(openai_mod, _chunks([("call_a", "python"), ("call_b", "rust")], lambda i: i))
    step = run.steps[0]
    assert "crashlab.stream.tool_calls_merged" not in step.attributes
    assert [c["name"] for c in step.response["tool_calls"]] == ["research", "research"]
    assert [f for f in evaluate(run, ChecksConfig()) if f.kind == "bad_tool_args"] == []


def test_one_call_split_over_deltas_is_not_flagged(openai_mod, crashlab_dir):
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    parts = [{"index": 0, "id": "call_a", "type": "function",
              "function": {"name": "research", "arguments": '{"inp'}},
             {"index": 0, "function": {"arguments": 'ut":"py"}'}},
             # some providers repeat the same id on every delta
             {"index": 0, "id": "call_a", "function": {"arguments": ""}}]  # fmt: skip
    chunks = [{**base, "choices": [{"index": 0, "delta": {"tool_calls": [p]}}]} for p in parts]
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    run = _record(openai_mod, chunks)
    assert "crashlab.stream.tool_calls_merged" not in run.steps[0].attributes
    assert evaluate(run, ChecksConfig()) == []


def test_replay_reproduces_the_merge(openai_mod, crashlab_dir):
    run = _record(openai_mod, _chunks([("call_a", "python"), ("call_b", "rust")]))
    replayed = crashlab.replay(
        run, overrides={"client": crashlab.wrap(_client(openai_mod, Script([500])))}
    )
    assert replayed.status == "ok"
    assert replayed.steps[0].attributes["crashlab.stream.tool_calls_merged"] == 2
