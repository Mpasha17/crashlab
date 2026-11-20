"""Native google.genai.Client support, with duck-typed fakes (no SDK, no network)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import crashlab
from crashlab.capture.genai import looks_like_genai
from crashlab.capture.openai import looks_like_openai
from crashlab.faults import FAULT_ATTR, llm_bad_tool_call, llm_empty, llm_error, llm_truncate
from crashlab.pricing import step_cost

# --- fakes ---------------------------------------------------------------------------


class FunctionCall:
    def __init__(self, name: str, args: dict | None = None, id: str | None = None) -> None:
        self.name = name
        self.args = args or {}
        self.id = id


class Part:
    def __init__(self, text: str | None = None, function_call: FunctionCall | None = None) -> None:
        self.text = text
        self.function_call = function_call

    def to_dict(self) -> dict:
        if self.function_call is not None:
            return {
                "function_call": {
                    "name": self.function_call.name,
                    "args": self.function_call.args,
                    "id": self.function_call.id,
                }
            }
        return {"text": self.text}


class Content:
    def __init__(self, parts: list[Part] | None = None, role: str = "model") -> None:
        self.parts = parts or []
        self.role = role

    def to_dict(self) -> dict:
        return {"role": self.role, "parts": [p.to_dict() for p in self.parts]}


def _as_part(value) -> Part:
    if isinstance(value, Part):
        return value
    if isinstance(value, dict):
        fc = value.get("function_call")
        if isinstance(fc, dict):
            fc = FunctionCall(name=fc.get("name"), args=fc.get("args") or {}, id=fc.get("id"))
        elif fc is not None and not isinstance(fc, FunctionCall):
            fc = FunctionCall(name=getattr(fc, "name", None), args=getattr(fc, "args", {}) or {})
        return Part(text=value.get("text"), function_call=fc)
    return Part(text=str(value))


def _as_content(value) -> Content:
    if isinstance(value, Content):
        return value
    if isinstance(value, dict):
        return Content(
            parts=[_as_part(p) for p in value.get("parts") or []], role=value.get("role") or "model"
        )
    return Content(parts=[], role="model")


def _as_candidate(value) -> Candidate:
    if isinstance(value, Candidate):
        return value
    if isinstance(value, dict):
        return Candidate(_as_content(value.get("content")), value.get("finish_reason") or "STOP")
    return Candidate(Content(), "STOP")


def _as_usage(value) -> UsageMetadata | None:
    if value is None or isinstance(value, UsageMetadata):
        return value
    if isinstance(value, dict):
        return UsageMetadata(
            prompt_token_count=value.get("prompt_token_count") or 0,
            candidates_token_count=value.get("candidates_token_count") or 0,
            total_token_count=value.get("total_token_count") or 0,
            cached_content_token_count=value.get("cached_content_token_count") or 0,
            thoughts_token_count=value.get("thoughts_token_count"),
        )
    return None


class Candidate:
    def __init__(self, content: Content | dict | None = None, finish_reason: str = "STOP") -> None:
        self.content = _as_content(content) if not isinstance(content, Content) else content
        self.finish_reason = finish_reason

    def to_dict(self) -> dict:
        return {"content": self.content.to_dict(), "finish_reason": self.finish_reason}


class UsageMetadata:
    def __init__(
        self,
        prompt_token_count: int = 0,
        candidates_token_count: int = 0,
        total_token_count: int = 0,
        cached_content_token_count: int = 0,
        thoughts_token_count: int | None = None,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        self.cached_content_token_count = cached_content_token_count
        self.thoughts_token_count = thoughts_token_count

    def to_dict(self) -> dict:
        data = {
            "prompt_token_count": self.prompt_token_count,
            "candidates_token_count": self.candidates_token_count,
            "total_token_count": self.total_token_count,
            "cached_content_token_count": self.cached_content_token_count,
        }
        if self.thoughts_token_count is not None:
            data["thoughts_token_count"] = self.thoughts_token_count
        return data


class GenerateContentResponse:
    def __init__(
        self,
        candidates: list[Candidate] | None = None,
        usage_metadata: UsageMetadata | None = None,
        model_version: str | None = None,
        text: str | None = None,
        function_calls: list[FunctionCall] | None = None,
        **_: object,
    ) -> None:
        if candidates is None and (text or function_calls):
            parts: list[Part] = []
            if text:
                parts.append(Part(text=text))
            for fc in function_calls or []:
                parts.append(Part(function_call=fc))
            candidates = [Candidate(Content(parts, role="model"), "STOP")]
        self.candidates = [_as_candidate(c) for c in (candidates or [])]
        self.usage_metadata = _as_usage(usage_metadata)
        self.model_version = model_version

    @property
    def text(self) -> str:
        parts = []
        if self.candidates:
            for p in self.candidates[0].content.parts:
                if p.text:
                    parts.append(p.text)
        return "".join(parts)

    @property
    def function_calls(self) -> list[FunctionCall]:
        out: list[FunctionCall] = []
        if self.candidates:
            for p in self.candidates[0].content.parts:
                if p.function_call is not None:
                    out.append(p.function_call)
        return out

    def to_dict(self) -> dict:
        usage = self.usage_metadata.to_dict() if self.usage_metadata is not None else None
        return {
            "text": self.text,
            "function_calls": [
                {"name": fc.name, "args": fc.args, "id": fc.id} for fc in self.function_calls
            ],
            "candidates": [c.to_dict() for c in self.candidates],
            "usage_metadata": usage,
            "model_version": self.model_version,
        }


def text_response(
    text: str,
    *,
    model: str = "gemini-2.5-flash",
    prompt: int = 10,
    candidates: int = 5,
    cached: int = 0,
    thoughts: int | None = None,
    total: int | None = None,
) -> GenerateContentResponse:
    usage = UsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=total if total is not None else prompt + candidates + (thoughts or 0),
        cached_content_token_count=cached,
        thoughts_token_count=thoughts,
    )
    return GenerateContentResponse(
        candidates=[Candidate(Content([Part(text=text)]))],
        usage_metadata=usage,
        model_version=model,
    )


def tool_response(
    name: str, args: dict, *, call_id: str = "call-1", model: str = "gemini-2.5-flash"
):
    fc = FunctionCall(name=name, args=args, id=call_id)
    return GenerateContentResponse(
        candidates=[
            Candidate(Content([Part(text="Let me check."), Part(function_call=fc)]), "STOP")
        ],
        usage_metadata=UsageMetadata(12, 8, 20),
        model_version=model,
    )


def stream_for(text: str, **usage_kw: object):
    final = text_response(text, **usage_kw)  # type: ignore[arg-type]
    mid = max(1, len(text) // 2)
    first = text[:mid]
    second = text[mid:]

    def chunk(piece: str, last: bool) -> GenerateContentResponse:
        return GenerateContentResponse(
            candidates=[
                Candidate(
                    Content([Part(text=piece)]),
                    finish_reason="STOP" if last else None,  # type: ignore[arg-type]
                )
            ],
            usage_metadata=final.usage_metadata if last else None,
            model_version=final.model_version if last else None,
        )

    return [chunk(first, False), chunk(second, True)]


class FakeModels:
    def __init__(self, owner: FakeClient) -> None:
        self.owner = owner

    def generate_content(self, *, model, contents, config=None, **kwargs):
        self.owner.calls.append(("generate_content", model, contents, config))
        return self.owner._next()

    def generate_content_stream(self, *, model, contents, config=None, **kwargs):
        self.owner.calls.append(("generate_content_stream", model, contents, config))
        result = self.owner._next()
        if isinstance(result, list):
            return iter(result)
        return iter([result])


class FakeChat:
    def __init__(self, owner: FakeClient, model: str, config=None, history=None) -> None:
        self.owner = owner
        self._model = model
        self._config = config
        self._history = list(history or [])

    def get_history(self, curated: bool = True):
        return list(self._history)

    def send_message(self, message, config=None, **kwargs):
        self.owner.calls.append(("send_message", self._model, message, config))
        resp = self.owner._next()
        self._history.append({"role": "user", "parts": [{"text": message}]})
        self._history.append({"role": "model", "parts": [{"text": resp.text}]})
        return resp

    def send_message_stream(self, message, config=None, **kwargs):
        self.owner.calls.append(("send_message_stream", self._model, message, config))
        result = self.owner._next()
        chunks = result if isinstance(result, list) else [result]
        text = "".join(getattr(c, "text", "") or "" for c in chunks)
        self._history.append({"role": "user", "parts": [{"text": message}]})
        self._history.append({"role": "model", "parts": [{"text": text}]})
        return iter(chunks)


class FakeChats:
    def __init__(self, owner: FakeClient) -> None:
        self.owner = owner

    def create(self, *, model, config=None, history=None, **kwargs):
        return FakeChat(self.owner, model, config=config, history=history)


class AsyncFakeModels:
    def __init__(self, owner: FakeClient) -> None:
        self.owner = owner

    async def generate_content(self, *, model, contents, config=None, **kwargs):
        return self.owner.models.generate_content(model=model, contents=contents, config=config)

    async def generate_content_stream(self, *, model, contents, config=None, **kwargs):
        for chunk in self.owner.models.generate_content_stream(
            model=model, contents=contents, config=config
        ):
            yield chunk


class AsyncFakeChat:
    def __init__(self, owner: FakeClient, model: str, config=None, history=None) -> None:
        self.inner = FakeChat(owner, model, config=config, history=history)

    def get_history(self, curated: bool = True):
        return self.inner.get_history(curated)

    async def send_message(self, message, config=None, **kwargs):
        return self.inner.send_message(message, config=config)

    async def send_message_stream(self, message, config=None, **kwargs):
        for chunk in self.inner.send_message_stream(message, config=config):
            yield chunk


class AsyncFakeChats:
    def __init__(self, owner: FakeClient) -> None:
        self.owner = owner

    def create(self, *, model, config=None, history=None, **kwargs):
        return AsyncFakeChat(self.owner, model, config=config, history=history)


class FakeClient:
    """Duck-typed ``google.genai.Client``: models + chats + aio, no network."""

    def __init__(self, script: list | None = None) -> None:
        self._script = list(script or [])
        self.calls: list = []
        self.models = FakeModels(self)
        self.chats = FakeChats(self)
        self.aio = SimpleNamespace(models=AsyncFakeModels(self), chats=AsyncFakeChats(self))
        self.base_url = "https://generativelanguage.googleapis.com/"

    def _next(self):
        if not self._script:
            raise AssertionError("fake google genai client has no more scripted responses")
        return self._script.pop(0)


# --- detection -----------------------------------------------------------------------


def test_looks_like_genai_and_not_openai():
    client = FakeClient()
    assert looks_like_genai(client) is True
    assert looks_like_openai(client) is False
    assert looks_like_genai(SimpleNamespace()) is False


def test_wrap_is_idempotent_and_sets_provider():
    client = crashlab.wrap(FakeClient([text_response("hi")]))
    assert crashlab.wrap(client) is client
    with crashlab.record("g-id"):
        client.models.generate_content(model="gemini-2.5-flash", contents="hi")
    step = crashlab.load("last").steps[0]
    assert step.attributes["gen_ai.provider.name"] == "google_genai"
    assert step.attributes["crashlab.api"] == "google.genai.models"
    assert step.attributes["server.address"] == "generativelanguage.googleapis.com"


def test_module_does_not_import_google_genai():
    assert "google.genai" not in sys.modules
    import crashlab.capture.genai as g  # noqa: F401

    assert "google.genai" not in sys.modules
    assert "google-genai" not in sys.modules


# --- generate_content ----------------------------------------------------------------


def test_generate_content_records_text_usage_and_cost():
    client = crashlab.wrap(
        FakeClient([text_response("hello there", prompt=100, candidates=10, cached=20)])
    )
    with crashlab.record("g-text"):
        resp = client.models.generate_content(model="gemini-2.5-flash", contents="hi")
    assert resp.text == "hello there"
    run = crashlab.load("last")
    step = run.steps[0]
    assert step.operation == "chat" and step.name == "gemini-2.5-flash"
    assert step.request["messages"][0]["parts"][0]["content"] == "hi"
    assert step.response["messages"][0]["parts"][0]["content"] == "hello there"
    assert step.response["finish_reason"] == "stop"
    assert step.attributes["gen_ai.usage.input_tokens"] == 100
    assert step.attributes["gen_ai.usage.output_tokens"] == 10
    assert step.attributes["gen_ai.usage.cache_read.input_tokens"] == 20
    assert step.attributes["crashlab.usage.total_tokens"] == 110
    # 2.5 Flash: $0.30 in (fresh = 80) / $0.03 cached / $2.50 out per 1M
    assert step_cost(step) == pytest.approx((80 * 0.30 + 20 * 0.03 + 10 * 2.50) / 1e6)
    assert run.usage.llm_calls == 1


def test_contents_and_config_shapes():
    script = [
        text_response("ok"),
        text_response("ok2"),
        text_response("ok3"),
    ]
    raw = FakeClient(script)
    client = crashlab.wrap(raw)

    config = SimpleNamespace(
        system_instruction="You are helpful.",
        temperature=0.2,
        max_output_tokens=256,
        tools=[
            SimpleNamespace(
                function_declarations=[
                    SimpleNamespace(
                        name="get_order",
                        description="Look up an order",
                        parameters={
                            "type": "object",
                            "properties": {"order_id": {"type": "string"}},
                        },
                    )
                ]
            )
        ],
    )
    with crashlab.record("g-shapes"):
        client.models.generate_content(
            model="gemini-2.5-flash", contents="plain string", config=config
        )
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents={"role": "user", "parts": [{"text": "from a dict"}]},
        )
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {"role": "user", "parts": [{"text": "hello"}]},
                {"role": "model", "parts": [{"text": "hi"}]},
                {"role": "user", "parts": [{"text": "again"}]},
            ],
        )
    steps = crashlab.load("last").chat_steps()
    assert steps[0].request["system"] == "You are helpful."
    assert steps[0].request["params"]["temperature"] == 0.2
    assert steps[0].request["params"]["max_tokens"] == 256
    assert steps[0].attributes["gen_ai.request.temperature"] == 0.2
    assert steps[0].attributes["gen_ai.request.max_tokens"] == 256
    assert steps[0].request["tools"][0]["name"] == "get_order"
    assert steps[0].request["tools"][0]["description"] == "Look up an order"
    assert steps[1].request["messages"][0]["parts"][0]["content"] == "from a dict"
    assert [m["role"] for m in steps[2].request["messages"]] == ["user", "assistant", "user"]
    assert steps[2].request["messages"][-1]["parts"][0]["content"] == "again"


def test_function_calls_normalized():
    client = crashlab.wrap(FakeClient([tool_response("get_order", {"order_id": "A17"})]))
    with crashlab.record("g-tools"):
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Refund A17",
            config={
                "tools": [
                    {
                        "function_declarations": [
                            {"name": "get_order", "parameters": {"type": "object"}}
                        ]
                    }
                ]
            },
        )
    assert resp.function_calls[0].name == "get_order"
    assert resp.function_calls[0].args == {"order_id": "A17"}
    step = crashlab.load("last").steps[0]
    assert step.response["finish_reason"] == "tool_calls"
    assert step.response["provider_finish_reason"] == "STOP"
    assert step.response["tool_calls"][0]["name"] == "get_order"
    assert step.response["tool_calls"][0]["arguments"] == {"order_id": "A17"}


# --- streaming -----------------------------------------------------------------------


def test_generate_content_stream_records_assembled_text():
    client = crashlab.wrap(FakeClient([stream_for("streamed hello")]))
    with crashlab.record("g-stream"):
        pieces = [
            chunk.text
            for chunk in client.models.generate_content_stream(
                model="gemini-2.5-flash", contents="go"
            )
        ]
    assert "".join(pieces) == "streamed hello"
    step = crashlab.load("last").steps[0]
    assert step.response["messages"][0]["parts"][0]["content"] == "streamed hello"
    assert step.attributes["crashlab.stream"] is True
    assert step.attributes["crashlab.stream.complete"] is True
    assert step.attributes["gen_ai.usage.input_tokens"] == 10


# --- chats ---------------------------------------------------------------------------


def test_chats_send_message_includes_history_and_replays():
    fake = FakeClient([text_response("first"), text_response("second")])
    client = crashlab.wrap(fake)

    @crashlab.agent(name="g-chat")
    def run(user: str = "hello") -> str:
        chat = client.chats.create(model="gemini-2.5-flash")
        a = chat.send_message(user)
        b = chat.send_message("follow up")
        return f"{a.text}|{b.text}"

    assert run() == "first|second"
    rec = crashlab.load("last")
    steps = rec.chat_steps()
    assert steps[0].request["messages"][0]["parts"][0]["content"] == "hello"
    # second turn saw the first exchange
    roles = [m["role"] for m in steps[1].request["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert steps[1].request["messages"][-1]["parts"][0]["content"] == "follow up"

    assert crashlab.replay(rec, fn=run).output == "first|second"
    from crashlab.replay.rebuild import rebuild

    obj = rebuild(steps[0].raw_response, None)
    assert obj.text == "first"
    assert obj.candidates[0].content.parts[0].text == "first"


def test_chat_send_message_stream():
    client = crashlab.wrap(FakeClient([stream_for("chunky")]))
    with crashlab.record("g-chat-stream"):
        chat = client.chats.create(model="gemini-2.5-flash")
        text = "".join(c.text or "" for c in chat.send_message_stream("hi"))
    assert text == "chunky"
    step = crashlab.load("last").steps[0]
    assert step.response["messages"][0]["parts"][0]["content"] == "chunky"
    assert step.attributes["crashlab.stream"] is True


# --- async ---------------------------------------------------------------------------


async def test_async_generate_content_and_stream():
    client = crashlab.wrap(FakeClient([text_response("async hi"), stream_for("async stream")]))
    async with crashlab.record("g-async"):
        resp = await client.aio.models.generate_content(model="gemini-2.5-flash", contents="q")
        assert resp.text == "async hi"
        pieces = []
        async for chunk in client.aio.models.generate_content_stream(
            model="gemini-2.5-flash", contents="q2"
        ):
            pieces.append(chunk.text)
        assert "".join(pieces) == "async stream"
    steps = crashlab.load("last").chat_steps()
    assert steps[0].response["messages"][0]["parts"][0]["content"] == "async hi"
    assert steps[1].response["messages"][0]["parts"][0]["content"] == "async stream"
    assert steps[1].attributes["crashlab.stream"] is True


async def test_async_chats():
    client = crashlab.wrap(FakeClient([text_response("ac1"), text_response("ac2")]))
    async with crashlab.record("g-achat"):
        chat = client.aio.chats.create(model="gemini-2.5-flash")
        a = await chat.send_message("one")
        b = await chat.send_message("two")
        assert a.text == "ac1" and b.text == "ac2"
    steps = crashlab.load("last").chat_steps()
    assert [m["role"] for m in steps[1].request["messages"]] == ["user", "assistant", "user"]


# --- replay / fork -------------------------------------------------------------------


def test_replay_without_sdk_exposes_text_and_parts():
    client = crashlab.wrap(
        FakeClient([text_response("offline"), tool_response("lookup", {"k": 1})])
    )

    @crashlab.agent(name="g-replay")
    def run(user: str = "hi") -> str:
        a = client.models.generate_content(model="gemini-2.5-flash", contents=user)
        b = client.models.generate_content(model="gemini-2.5-flash", contents="next")
        names = [
            getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else None)
            for fc in (b.function_calls or [])
        ]
        parts = b.candidates[0].content.parts
        return f"{a.text}|{','.join(n for n in names if n)}|{bool(parts)}"

    assert run().startswith("offline|lookup|")
    rec = crashlab.load("last")
    assert crashlab.replay(rec, fn=run).output.startswith("offline|lookup|")
    raw = rec.chat_steps()[0].raw_response
    from crashlab.replay.rebuild import ReplayObject, rebuild

    obj = rebuild(raw, "google.genai.types.GenerateContentResponse")
    assert isinstance(obj, ReplayObject)
    assert obj.text == "offline"
    assert obj.candidates[0].content.parts[0].text == "offline"


def test_replay_stream_offline():
    client = crashlab.wrap(FakeClient([stream_for("abc def ghi")]))

    @crashlab.agent(name="g-rstream")
    def run(user: str = "x") -> str:
        return "".join(
            c.text or ""
            for c in client.models.generate_content_stream(model="gemini-2.5-flash", contents=user)
        )

    assert run() == "abc def ghi"
    rec = crashlab.load("last")
    assert crashlab.replay(rec, fn=run).output == "abc def ghi"


def test_fork_replays_then_can_continue():
    client = crashlab.wrap(
        FakeClient([text_response("first"), text_response("second"), text_response("forked")])
    )

    @crashlab.agent(name="g-fork")
    def run() -> str:
        a = client.models.generate_content(model="gemini-2.5-flash", contents="a")
        b = client.models.generate_content(model="gemini-2.5-flash", contents="b")
        return f"{a.text}|{b.text}"

    assert run() == "first|second"
    rec = crashlab.load("last")
    # first step from the cassette, second call goes live
    branch = crashlab.fork(rec, at_step=1, fn=run)
    assert branch.output == "first|forked"
    assert crashlab.replay(rec, fn=run).output == "first|second"


# --- faults --------------------------------------------------------------------------


def test_llm_error_is_recorded_and_not_reinjected_on_replay():
    client = crashlab.wrap(FakeClient([text_response("should not be used")]))

    @crashlab.agent(name="g-err")
    def run(user: str = "hi") -> str:
        return client.models.generate_content(model="gemini-2.5-flash", contents=user).text

    with pytest.raises(crashlab.faults.InjectedFaultError, match="rate limited"):
        with crashlab.faults(llm_error(status=429, message="rate limited", on_calls=[1])):
            run()
    rec = crashlab.load("last")
    step = rec.steps[0]
    assert step.attributes[FAULT_ATTR]["kind"] == "llm_error"
    assert step.error is not None
    # strict replay reproduces the recorded error and does not call the model
    rep = crashlab.replay(rec, fn=run)
    assert rep.status != "ok"
    assert "rate limited" in ((rep.error.message if rep.error else "") or "")
    assert len(client.unwrapped.calls) == 0


def test_llm_empty_truncate_and_bad_tool_call():
    # empty
    client = crashlab.wrap(FakeClient([text_response("secret answer")]))
    with crashlab.record("g-empty"):
        with crashlab.faults(llm_empty(on_calls=[1])):
            resp = client.models.generate_content(model="gemini-2.5-flash", contents="q")
    assert (resp.text or "") == ""
    assert list(resp.function_calls or []) == []
    step = crashlab.load("last").steps[0]
    assert step.attributes[FAULT_ATTR]["kind"] == "llm_empty"
    assert step.response["messages"][0]["parts"] == [] or not any(
        p.get("type") == "text" and p.get("content") for p in step.response["messages"][0]["parts"]
    )

    # truncate
    client = crashlab.wrap(FakeClient([text_response("abcdefghijklmnopqrstuvwxyz")]))
    with crashlab.record("g-trunc"):
        with crashlab.faults(llm_truncate(max_chars=5, on_calls=[1])):
            resp = client.models.generate_content(model="gemini-2.5-flash", contents="q")
    assert resp.text == "abcde"
    step = crashlab.load("last").steps[0]
    assert step.attributes[FAULT_ATTR]["kind"] == "llm_truncate"
    assert step.response["messages"][0]["parts"][0]["content"] == "abcde"

    # bad tool call
    client = crashlab.wrap(FakeClient([text_response("I will call a tool")]))
    with crashlab.record("g-badtc"):
        with crashlab.faults(llm_bad_tool_call(name="not_a_real_tool", on_calls=[1])):
            resp = client.models.generate_content(model="gemini-2.5-flash", contents="q")
    names = [
        getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else None)
        for fc in (resp.function_calls or [])
    ]
    assert "not_a_real_tool" in names
    step = crashlab.load("last").steps[0]
    assert step.attributes[FAULT_ATTR]["kind"] == "llm_bad_tool_call"
    assert step.response["finish_reason"] == "tool_calls"
    assert any(tc["name"] == "not_a_real_tool" for tc in step.response["tool_calls"])


def test_wrap_agent_label_is_recorded():
    client = crashlab.wrap(FakeClient([text_response("ok")]), agent="researcher")
    with crashlab.record("g-agent"):
        client.models.generate_content(model="gemini-2.5-flash", contents="q")
    assert crashlab.load("last").steps[0].attributes["gen_ai.agent.name"] == "researcher"


def test_thinking_tokens_in_usage():
    client = crashlab.wrap(
        FakeClient([text_response("done", prompt=10, candidates=5, thoughts=20, total=35)])
    )
    with crashlab.record("g-think"):
        client.models.generate_content(model="gemini-2.5-flash", contents="q")
    step = crashlab.load("last").steps[0]
    assert step.attributes["gen_ai.usage.reasoning.output_tokens"] == 20
    assert step.attributes["crashlab.usage.total_tokens"] == 35
    # thinking tokens billed as output: max(out, total - inp) = 25
    assert step_cost(step) == pytest.approx((10 * 0.30 + 25 * 2.50) / 1e6)


def test_provider_forced_google_genai():
    client = crashlab.wrap(FakeClient([text_response("ok")]), provider="google_genai")
    with crashlab.record("g-forced"):
        client.models.generate_content(model="gemini-2.5-flash", contents="q")
    assert crashlab.load("last").steps[0].attributes["gen_ai.provider.name"] == "google_genai"
