"""Regressions from live runs against OpenAI-compatible providers (Groq, Gemini).

Payload shapes mirror what those endpoints actually return; values are synthetic. No network.
"""

from __future__ import annotations

import base64
import json

import crashlab
from crashlab.capture import normalize as N
from tests.fakes import FakeOpenAI

# Opaque replay tokens are base64 and can be long; they must survive verbatim.
SIGNATURE = base64.b64encode(bytes(range(256)) * 12).decode()  # ~4 KB
assert len(SIGNATURE) > N.BINARY_PLACEHOLDER_MIN

GEMINI_TOOL_CALL = {
    "id": "call_g1",
    "type": "function",
    "function": {"name": "get_order", "arguments": '{"order_id":"A17"}'},
    "extra_content": {"google": {"thought_signature": SIGNATURE}},
}


def _completion(message, finish, usage):
    return {
        "id": "c1",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    }


def test_opaque_provider_tokens_are_not_elided():
    raw = {
        "messages": [
            {"role": "assistant", "tool_calls": [GEMINI_TOOL_CALL]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hm", "signature": SIGNATURE},
                {"type": "redacted_thinking", "data": SIGNATURE},
            ]},
        ],
        "input": [{"type": "reasoning", "encrypted_content": SIGNATURE, "summary": []}],
        "image": {"type": "base64", "media_type": "image/png", "data": SIGNATURE},
        "audio": {"data": SIGNATURE, "format": "wav"},
        "url": f"data:image/png;base64,{SIGNATURE}",
    }  # fmt: skip
    out = N.strip_binary(raw)
    msgs = out["messages"]
    assert msgs[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == SIGNATURE
    assert msgs[1]["content"][0]["signature"] == SIGNATURE
    assert msgs[1]["content"][1]["data"] == SIGNATURE
    assert out["input"][0]["encrypted_content"] == SIGNATURE
    # real media is still elided
    assert out["image"]["data"].startswith("<binary ")
    assert out["audio"]["data"].startswith("<binary ")
    assert out["url"].startswith("<image/png ")


def test_reasoning_fields_become_reasoning_parts():
    for key in ("reasoning", "reasoning_content"):  # Groq/OpenRouter vs DeepSeek/vLLM/LiteLLM
        msg = {"role": "assistant", "content": "Done.", key: "Need to look up the order."}
        norm, _ = N.normalize_openai_chat_response(_completion(msg, "stop", None))
        assert norm["messages"][0]["parts"] == [
            {"type": "reasoning", "content": "Need to look up the order."},
            {"type": "text", "content": "Done."},
        ]
    # echoed back in a request: same parts
    req = N.normalize_messages([{"role": "assistant", "content": None, "reasoning": "r"}])
    assert req[0]["parts"] == [{"type": "reasoning", "content": "r"}]
    # user messages are untouched
    assert N.normalize_messages([{"role": "user", "content": "x", "reasoning": "r"}])[0][
        "parts"
    ] == [{"type": "text", "content": "x"}]


def test_stop_with_tool_calls_normalizes_to_tool_calls():
    msg = {"role": "assistant", "content": None, "tool_calls": [GEMINI_TOOL_CALL]}
    norm, _ = N.normalize_openai_chat_response(_completion(msg, "stop", None))
    assert norm["finish_reason"] == "tool_calls"
    assert norm["finish_reasons"] == ["tool_calls"]
    assert norm["provider_finish_reason"] == "stop"
    text, _ = N.normalize_openai_chat_response(_completion({"content": "hi"}, "stop", None))
    assert text["finish_reason"] == "stop"


def _chunk(delta=None, finish=None, usage=None):
    c = {"id": "s1", "object": "chat.completion.chunk", "model": "m", "choices": []}
    if delta is not None or finish is not None:
        c["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage:
        c["usage"] = usage
    return c


def test_gemini_style_stream_reasoning_signature_and_finish():
    tc = {"index": 0, **GEMINI_TOOL_CALL}
    chunks = [
        _chunk({"role": "assistant", "reasoning": "Look it "}),
        _chunk({"reasoning": "up."}),
        _chunk({"tool_calls": [tc]}, finish="stop"),  # Gemini: "stop" on a tool turn
        _chunk(usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 101}),
    ]
    client = crashlab.wrap(FakeOpenAI([chunks]))
    with crashlab.record("compat-stream"):
        for _ in client.chat.completions.create(model="m", messages=[], stream=True):
            pass
    run = crashlab.load("last")
    step = run.steps[0]
    assert step.response["messages"][0]["parts"][0] == {
        "type": "reasoning",
        "content": "Look it up.",
    }
    assert step.response["tool_calls"] == [
        {"id": "call_g1", "type": "function", "name": "get_order", "arguments": {"order_id": "A17"}}
    ]
    assert step.response["finish_reason"] == "tool_calls"
    assert step.response["provider_finish_reason"] == "stop"
    raw_tc = step.raw_response["choices"][0]["message"]["tool_calls"][0]
    assert raw_tc["extra_content"]["google"]["thought_signature"] == SIGNATURE
    assert step.raw_response["choices"][0]["message"]["reasoning"] == "Look it up."
    # provider total (incl. thinking tokens) is kept, not recomputed as in+out
    assert step.attributes["crashlab.usage.total_tokens"] == 101
    assert (run.usage.input_tokens, run.usage.output_tokens, run.usage.total_tokens) == (
        50,
        20,
        101,
    )


def test_run_total_sums_provider_totals_and_falls_back_to_in_plus_out():
    with_total = _completion(
        {"content": "a"}, "stop", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 40}
    )
    no_total = _completion({"content": "b"}, "stop", {"prompt_tokens": 7, "completion_tokens": 3})
    client = crashlab.wrap(FakeOpenAI([with_total, no_total]))  # fmt: skip
    with crashlab.record("compat-usage"):
        client.chat.completions.create(model="m", messages=[])
        client.chat.completions.create(model="m", messages=[])
    u = crashlab.load("last").usage
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (17, 8, 50)


def test_refund_bot_echoes_tool_call_extras():
    from examples.refund_bot.agent import _assistant_message
    from pydantic import BaseModel, ConfigDict

    class SDKToolCall(BaseModel):  # openai SDK models keep unknown fields, like this
        model_config = ConfigDict(extra="allow")
        id: str
        type: str
        function: dict

    class SDKMessage(BaseModel):
        content: str | None = None
        tool_calls: list[SDKToolCall] | None = None

    msg = _assistant_message(SDKMessage(tool_calls=[SDKToolCall(**GEMINI_TOOL_CALL)]))
    assert msg["tool_calls"][0] == GEMINI_TOOL_CALL
    assert json.dumps(msg).count(SIGNATURE) == 1
