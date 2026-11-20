"""crashlab.integrations.langchain.CrashlabCallbackHandler.

No module-level openai / anthropic / litellm / langchain / mcp / yaml imports.
"""

from __future__ import annotations

import uuid
from typing import Any

import crashlab
from crashlab.integrations.langchain import CrashlabCallbackHandler
from tests.fakes import FakeOpenAI, text_response


class Msg:
    def __init__(self, content: str, typ: str = "human", tool_calls: Any = None) -> None:
        self.content = content
        self.type = typ
        self.tool_calls = tool_calls
        self.additional_kwargs: dict[str, Any] = {}


class Gen:
    def __init__(self, text: str, tool_calls: Any = None) -> None:
        self.text = text
        self.message = Msg(text, typ="ai", tool_calls=tool_calls)
        self.generation_info = {"finish_reason": "stop"}


class LLMResult:
    def __init__(self, text: str, model: str = "gpt-x", tool_calls: Any = None) -> None:
        self.generations = [[Gen(text, tool_calls=tool_calls)]]
        self.llm_output = {
            "model_name": model,
            "token_usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }


def _uid() -> uuid.UUID:
    return uuid.uuid4()


def test_import_does_not_need_langchain():
    import crashlab.integrations.langchain as lc

    handler = lc.CrashlabCallbackHandler()
    assert handler.name == "crashlab"
    assert callable(handler.on_llm_end)


def test_records_chat_model_and_tool_when_the_client_is_not_wrapped():
    handler = CrashlabCallbackHandler()
    llm_id, tool_id = _uid(), _uid()
    with crashlab.record("lc-raw") as run:
        handler.on_chat_model_start(
            {"name": "ChatOpenAI", "kwargs": {"model": "gpt-x"}},
            [[Msg("be brief", typ="system"), Msg("hello", typ="human")]],
            run_id=llm_id,
            invocation_params={"model": "gpt-x", "temperature": 0},
        )
        handler.on_llm_end(LLMResult("hi there", "gpt-x"), run_id=llm_id)
        handler.on_tool_start(
            {"name": "lookup"},
            '{"key": "a"}',
            run_id=tool_id,
            inputs={"key": "a"},
        )
        handler.on_tool_end({"key": "a", "value": 1}, run_id=tool_id)
        run.output = "hi there"
    src = crashlab.load(run.run_id)
    chat, tool = src.steps
    assert chat.operation == "chat" and chat.name == "gpt-x"
    assert chat.request["system"] == "be brief"
    assert chat.request["messages"][0]["role"] == "user"
    assert chat.request["params"]["temperature"] == 0
    assert chat.response["messages"][0]["parts"][0]["content"] == "hi there"
    assert chat.attributes["crashlab.api"] == "langchain.callback"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 3
    assert tool.operation == "execute_tool" and tool.name == "lookup"
    assert tool.request["args"] == {"key": "a"}
    assert tool.result == {"key": "a", "value": 1}


def test_on_llm_start_prompts_and_errors():
    handler = CrashlabCallbackHandler()
    ok, err, tool_err = _uid(), _uid(), _uid()
    with crashlab.record("lc-err"):
        handler.on_llm_start({"name": "LLM"}, ["Say hi"], run_id=ok,
                             invocation_params={"model": "m"})  # fmt: skip
        handler.on_llm_end(LLMResult("hello", "m"), run_id=ok)
        handler.on_chat_model_start({}, [[Msg("q")]], run_id=err,
                                    invocation_params={"model": "m"})  # fmt: skip
        handler.on_llm_error(RuntimeError("429"), run_id=err)
        handler.on_tool_start({"name": "boom"}, "x", run_id=tool_err, inputs={"x": 1})
        handler.on_tool_error(ValueError("nope"), run_id=tool_err)
    src = crashlab.load("last")
    assert src.steps[0].response["messages"][0]["parts"][0]["content"] == "hello"
    assert src.steps[1].error is not None and "429" in src.steps[1].error.message
    assert src.steps[2].error is not None and src.steps[2].error.is_tool_error


def test_does_not_double_record_when_the_client_is_already_wrapped():
    handler = CrashlabCallbackHandler()
    client = crashlab.wrap(FakeOpenAI([text_response(1, "from wrap")]))
    llm_id, tool_id = _uid(), _uid()

    @crashlab.tool()
    def lookup(key: str) -> dict:
        return {"key": key}

    with crashlab.record("lc-wrapped") as run:
        handler.on_chat_model_start(
            {},
            [[Msg("hello")]],
            run_id=llm_id,
            invocation_params={"model": "fake-model"},
        )
        client.chat.completions.create(
            model="fake-model", messages=[{"role": "user", "content": "hello"}]
        )
        handler.on_llm_end(LLMResult("from callback", "fake-model"), run_id=llm_id)
        handler.on_tool_start({"name": "lookup"}, "a", run_id=tool_id, inputs={"key": "a"})
        lookup(key="a")
        handler.on_tool_end({"key": "a"}, run_id=tool_id)
        run.output = "ok"
    src = crashlab.load(run.run_id)
    assert [s.operation for s in src.steps] == ["chat", "execute_tool"]
    assert src.steps[0].response["messages"][0]["parts"][0]["content"] == "from wrap"
    assert src.steps[0].attributes.get("crashlab.api") != "langchain.callback"
    assert src.steps[1].result == {"key": "a"}


def test_langgraph_node_labels_wrapped_calls_via_agent_scope():
    handler = CrashlabCallbackHandler()
    client = crashlab.wrap(FakeOpenAI([text_response(1, "notes"), text_response(2, "done")]))
    research, reply = _uid(), _uid()
    with crashlab.record("lc-graph") as run:
        handler.on_chain_start(
            {"name": "RunnableSequence"},
            {},
            run_id=research,
            metadata={"langgraph_node": "researcher"},
        )
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "research"}]
        )
        handler.on_chain_end({}, run_id=research)
        handler.on_chain_start({"name": "writer"}, {}, run_id=reply, name="writer")
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "write"}])
        handler.on_chain_end({}, run_id=reply)
        run.output = "done"
    src = crashlab.load(run.run_id)
    assert src.steps[0].agent == "researcher"
    assert src.steps[1].agent == "writer"
    assert src.agents == ["researcher", "writer"]


def test_generic_chain_names_do_not_relabel():
    handler = CrashlabCallbackHandler()
    client = crashlab.wrap(FakeOpenAI([text_response(1, "ok")]))
    with crashlab.record("lc-generic") as run:
        handler.on_chain_start({"name": "RunnableSequence"}, {}, run_id=_uid())
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
        run.output = "ok"
    assert crashlab.load(run.run_id).steps[0].agent is None


def test_events_outside_a_run_are_ignored():
    handler = CrashlabCallbackHandler()
    handler.on_chat_model_start({}, [[Msg("q")]], run_id=_uid())
    handler.on_llm_end(LLMResult("x"), run_id=_uid())
    handler.on_tool_start({"name": "t"}, "1", run_id=_uid())
    handler.on_chain_start({}, {}, run_id=_uid(), metadata={"langgraph_node": "x"})
