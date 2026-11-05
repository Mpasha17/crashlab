"""Replayed tool errors keep what handlers read: ``e.args`` (``KeyError('b')``) and the
exception's own attributes (``e.sku`` on a domain error, ``e.retry_after``), and an async
tool cancelled from outside (``asyncio.wait_for`` timeout) is cancelled again on replay
instead of raising ``CancelledError`` into the agent."""

from __future__ import annotations

import asyncio
import sys

import pytest

import crashlab


class OutOfStock(Exception):
    def __init__(self, sku: str, left: int) -> None:
        super().__init__(f"{sku} out of stock")
        self.sku = sku
        self.left = left
        self.retry_after = 1.5
        self._internal = object()  # not JSON: skipped


@crashlab.tool()
def reserve(sku: str) -> dict:
    raise OutOfStock(sku, 0)


@crashlab.tool()
def lookup(key: str) -> int:
    return {"a": 1}[key]


@crashlab.agent(name="exc-state")
def exc_state() -> str:
    out = []
    try:
        reserve("w1")
    except OutOfStock as e:
        out.append(f"{e.sku} left={e.left} retry={e.retry_after}")
    try:
        lookup("b")
    except KeyError as e:
        out.append(f"{e!r} missing={e.args[0]}")
    return " | ".join(out)


def test_exception_attributes_and_args_replayed():
    assert exc_state() == "w1 left=0 retry=1.5 | KeyError('b') missing=b"
    run = crashlab.load("last")
    assert run.steps[0].error.state["attrs"] == {"sku": "w1", "left": 0, "retry_after": 1.5}
    assert crashlab.replay(run).output == run.output


def test_secret_in_exception_attribute_redacted():
    class AuthFailed(Exception):
        def __init__(self) -> None:
            super().__init__("denied")
            self.api_key = "sk-proj-" + "A" * 30

    @crashlab.tool()
    def login() -> None:
        raise AuthFailed()

    @crashlab.agent(name="auth")
    def auth() -> str:
        try:
            login()
        except AuthFailed:
            return "denied"
        return "ok"

    auth()
    assert "AAAAAAAAAAAAAAAA" not in crashlab.load("last").model_dump_json()


@crashlab.tool()
async def slow(x: int) -> int:
    await asyncio.sleep(0.5)
    return x


@crashlab.agent(name="wait-for")
async def wait_for_agent() -> str:
    try:
        await asyncio.wait_for(slow(1), 0.05)
    except (asyncio.TimeoutError, TimeoutError):
        return "timed out"
    return "finished"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Task.cancelling() is 3.11+")
def test_externally_cancelled_async_tool_replays_as_timeout():
    assert asyncio.run(wait_for_agent()) == "timed out"
    run = crashlab.load("last")
    assert run.steps[0].error.type.endswith("CancelledError")
    replayed = crashlab.replay(run)
    assert replayed.output == "timed out"


@crashlab.tool()
async def cancels_itself() -> int:
    raise asyncio.CancelledError()


@crashlab.agent(name="self-cancel")
async def self_cancel() -> str:
    try:
        await cancels_itself()
    except asyncio.CancelledError:
        return "caught"
    return "no"


def test_cancelled_error_raised_by_the_tool_itself_still_raised_on_replay():
    """No outer timeout: replay must not hang waiting for a cancel that never comes."""
    assert asyncio.run(self_cancel()) == "caught"
    run = crashlab.load("last")
    replayed = crashlab.replay(run)
    assert replayed.output == "caught"


@pytest.mark.parametrize("args", [(), ("a", 1), ("x" * 5000,)])
def test_args_round_trip_only_when_small(args):
    from crashlab.capture._steps import error_info
    from crashlab.replay.rebuild import rebuild_exception

    exc = rebuild_exception(error_info(ValueError(*args)))
    if len(repr(args)) < 2000:
        assert exc.args == args
    assert isinstance(exc, ValueError)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Task.cancelling() is 3.11+")
def test_llm_call_cancelled_by_wait_for_replays_as_timeout():
    """A wrapped AsyncOpenAI call under ``asyncio.wait_for`` that timed out (agent step
    timeouts) replayed as a bare CancelledError: 'replay raised CancelledError'."""
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")

    async def handler(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={})

    @crashlab.agent(name="llm-timeout")
    async def agent() -> str:
        client = crashlab.wrap(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="http://llm.invalid/v1",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
        )
        msgs = [{"role": "user", "content": "hi"}]
        try:
            await asyncio.wait_for(client.chat.completions.create(model="m", messages=msgs), 0.05)
        except (asyncio.TimeoutError, TimeoutError):
            return "timed out"
        return "done"

    assert asyncio.run(agent()) == "timed out"
    run = crashlab.load("last")
    assert run.steps[0].error.state["cancelled_outside_ms"] >= 0
    assert crashlab.replay(run, fn=agent).output == "timed out"


def test_state_left_out_when_empty_so_older_readers_load_the_step():
    """crashlab<=0.0.2 validates with extra='forbid': a ``state`` key on every error made
    it skip the step as damaged. Write it only when there is state."""
    from crashlab.models import ErrorInfo

    assert "state" not in ErrorInfo(type="ValueError", message="x").model_dump(mode="json")
    full = ErrorInfo(type="KeyError", message="'b'", state={"args": ["b"]})
    assert full.model_dump(mode="json")["state"] == {"args": ["b"]}
    assert ErrorInfo.model_validate(full.model_dump(mode="json")) == full


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Task.cancelling() is 3.11+")
def test_stream_cancelled_mid_way_by_wait_for_replays_as_timeout():
    """The agent's timeout fired while it was reading an AsyncOpenAI stream: replay
    served the 4 chunks and raised CancelledError at once instead of timing out."""
    import json as _json

    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")

    class SlowSSE(httpx.AsyncByteStream):
        async def __aiter__(self):
            for i in range(10):
                chunk = {"id": "c", "object": "chat.completion.chunk", "created": 1,
                         "model": "m", "choices": [{"index": 0, "delta": {"content": f"w{i} "},
                                                    "finish_reason": None}]}  # fmt: skip
                yield f"data: {_json.dumps(chunk)}\n\n".encode()
                await asyncio.sleep(0.05)
            yield b"data: [DONE]\n\n"

    async def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SlowSSE())

    @crashlab.agent(name="stream-timeout")
    async def agent() -> str:
        client = crashlab.wrap(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="http://llm.invalid/v1",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
        )
        got: list[str] = []

        async def consume() -> None:
            stream = await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}], stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    got.append(chunk.choices[0].delta.content)

        try:
            await asyncio.wait_for(consume(), 0.18)
        except (asyncio.TimeoutError, TimeoutError):
            return f"timeout after {len(got)} chunks"
        return "".join(got)

    out = asyncio.run(agent())
    assert out.startswith("timeout after")
    assert crashlab.replay(crashlab.load("last"), fn=agent).output == out
