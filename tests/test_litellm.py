from __future__ import annotations

import types

import pytest

import crashlab
from crashlab.integrations import litellm as cl_litellm
from tests.fakes import (
    AsyncFakeStream,
    FakeStream,
    chunks_for_text,
    text_response,
    tool_call_response,
)


def _fake_litellm():
    mod = types.ModuleType("litellm")
    mod.calls = []

    def completion(model, messages, stream=False, **kwargs):
        mod.calls.append((model, messages, kwargs))
        if stream:
            return FakeStream(chunks_for_text("lite stream", model=model))
        if kwargs.get("tools"):
            return tool_call_response(1, "get_order", {"order_id": "A17"}, model=model)
        return text_response(1, "lite hello", model=model)

    async def acompletion(model, messages, stream=False, **kwargs):
        if stream:
            return AsyncFakeStream(chunks_for_text("lite astream", model=model))
        return text_response(2, "lite async", model=model)

    mod.completion = completion
    mod.acompletion = acompletion
    return mod


def test_patch_unpatch_fake_module():
    mod = _fake_litellm()
    orig = mod.completion, mod.acompletion
    cl_litellm.patch(mod)
    cl_litellm.patch(mod)  # idempotent
    assert cl_litellm.is_patched() and mod.completion is not orig[0]
    with pytest.raises(RuntimeError):
        cl_litellm.patch(_fake_litellm())
    with crashlab.record("lite"):
        # positional model + messages, as litellm users often call it
        key = "sk-ant-api03-SECRETSECRETSECRET1234"
        msgs = [{"role": "user", "content": "hi"}]
        r = mod.completion("anthropic/claude-x", msgs, temperature=0, api_key=key)
        assert r.choices[0].message.content == "lite hello"
        list(mod.completion(model="gpt-4o-mini", messages=[], stream=True))
    s1, s2 = crashlab.load("last").steps
    assert s1.name == "anthropic/claude-x"
    assert s1.attributes["gen_ai.provider.name"] == "anthropic"
    assert s1.attributes["crashlab.api"] == "litellm.completion"
    assert s1.request["messages"] == [
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
    ]
    assert s1.request["params"] == {"model": "anthropic/claude-x", "temperature": 0}
    assert s1.raw_request["api_key"] == "[REDACTED]"
    assert s2.attributes["gen_ai.provider.name"] == "litellm"
    assert s2.response["messages"][0]["parts"][0]["content"] == "lite stream"
    assert s2.attributes["crashlab.stream.complete"] is True
    cl_litellm.unpatch()
    assert (mod.completion, mod.acompletion) == orig and not cl_litellm.is_patched()


def test_api_base_host_is_recorded_and_prices_the_call():
    # live (DSPy, Google ADK): openai/<model> routed to Groq via api_base had no cost
    mod = _fake_litellm()
    cl_litellm.patch(mod)
    try:
        with crashlab.record("lite-host"):
            mod.completion(model="openai/openai/gpt-oss-20b", messages=[],
                           api_base="https://api.groq.com/openai/v1")  # fmt: skip
            mod.completion(model="gpt-4o-mini", messages=[])
    finally:
        cl_litellm.unpatch()
    s1, s2 = crashlab.load("last").steps
    assert s1.attributes["server.address"] == "api.groq.com"
    assert "server.address" not in s2.attributes


async def test_fake_module_async():
    mod = _fake_litellm()
    crashlab.wrap(mod)  # wrap(litellm) == patch(litellm)
    async with crashlab.record("lite-async"):
        r = await mod.acompletion(model="m", messages=[])
        assert r.choices[0].message.content == "lite async"
        stream = await mod.acompletion(model="m", messages=[], stream=True)
        async for _ in stream:
            pass
    s1, s2 = crashlab.load("last").steps
    assert s1.response["messages"][0]["parts"][0]["content"] == "lite async"
    assert s2.response["messages"][0]["parts"][0]["content"] == "lite astream"


def test_tool_calls_via_fake_litellm():
    mod = _fake_litellm()
    cl_litellm.patch(mod)

    @crashlab.tool()
    def get_order(order_id: str) -> dict:
        return {"order_id": order_id}

    with crashlab.record("lite-tools"):
        r = mod.completion(model="m", messages=[], tools=[{"type": "function", "function": {
            "name": "get_order", "parameters": {"type": "object"}}}])  # fmt: skip
        tc = r.choices[0].message.tool_calls[0]
        get_order(order_id="A17")
    chat, tool = crashlab.load("last").steps
    assert chat.response["tool_calls"][0]["id"] == tc.id
    assert tool.attributes["gen_ai.tool.call.id"] == tc.id


# --- real litellm (mock_response: no network) ------------------------------------------


@pytest.fixture
def litellm_mod():
    return pytest.importorskip("litellm")


def test_real_litellm_mock_response(litellm_mod):
    cl_litellm.patch()
    with crashlab.record("real-lite"):
        r = litellm_mod.completion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}],
            mock_response="hello from litellm",
        )  # fmt: skip
        chunks = list(
            litellm_mod.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="streamed from litellm",
                stream=True,
            )  # fmt: skip
        )
    assert r.choices[0].message.content == "hello from litellm"
    assert len(chunks) > 1
    s1, s2 = crashlab.load("last").steps
    assert s1.response["messages"][0]["parts"][0]["content"] == "hello from litellm"
    assert s1.attributes["gen_ai.usage.input_tokens"] is not None
    assert "mock_response" not in s1.request["params"]
    assert s2.response["messages"][0]["parts"][0]["content"] == "streamed from litellm"
    assert s1.request_hash == s2.request_hash


async def test_real_litellm_acompletion(litellm_mod):
    cl_litellm.patch()
    async with crashlab.record("real-lite-async"):
        r = await litellm_mod.acompletion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}],
            mock_response="async litellm",
        )  # fmt: skip
    assert r.choices[0].message.content == "async litellm"
    step = crashlab.load("last").steps[0]
    assert step.response["messages"][0]["parts"][0]["content"] == "async litellm"
