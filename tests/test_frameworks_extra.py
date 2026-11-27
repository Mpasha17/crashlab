"""crashlab.integrations.frameworks_extra: LlamaIndex, smolagents, Haystack.

Duck-typed fakes only. No module-level llama_index / smolagents / haystack / provider SDK
imports, so the file runs in the core-only (venv-nosdk) environment.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import pytest

import crashlab
from crashlab import rag
from crashlab.integrations import (
    HaystackTracer,
    LlamaIndexHandler,
    crashlab_step_callback,
    llama_index_handler,
    wrap_pipeline,
    wrap_query_engine,
    wrap_smolagent,
)
from crashlab.integrations import frameworks_extra as fx

# ---------------------------------------------------------------------------
# importability
# ---------------------------------------------------------------------------


def test_import_does_not_need_optional_sdks():
    code = (
        "import sys, crashlab.integrations.frameworks_extra as m;"
        "bad=[n for n in ('llama_index','smolagents','haystack','openai','anthropic') "
        "if n in sys.modules];"
        "assert not bad, bad; assert m.llama_index_handler()"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(crashlab.__file__).resolve().parents[1])}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_public_reexports_match_submodule():
    assert llama_index_handler is fx.llama_index_handler
    assert wrap_smolagent is fx.wrap_smolagent
    assert wrap_pipeline is fx.wrap_pipeline
    assert isinstance(llama_index_handler(), LlamaIndexHandler)
    assert callable(crashlab_step_callback())
    assert HaystackTracer is fx.HaystackTracer


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


class _Ev:
    def __init__(self, sid: str = "s1", **kw: Any) -> None:
        self.span_id = sid
        self.id_ = f"ev-{sid}"
        for k, v in kw.items():
            setattr(self, k, v)


def _event(kind: str, **kw: Any) -> Any:
    return type(kind, (_Ev,), {"class_name": classmethod(lambda cls: kind)})(**kw)


class Role:  # MessageRole-like enum
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"MessageRole.{self.value.upper()}"


class LIMessage:
    def __init__(self, role: str, content: str, **extra: Any) -> None:
        self.role = Role(role)
        self.content = content
        self.additional_kwargs = extra.get("additional_kwargs", {})
        self.blocks: list[Any] = []


class LIChatResponse:
    def __init__(self, message: LIMessage, raw: Any = None) -> None:
        self.message = message
        self.raw = raw
        self.additional_kwargs: dict[str, Any] = {}


class LINode:
    def __init__(self, node_id: str, text: str, **meta: Any) -> None:
        self.node_id = node_id
        self.text = text
        self.metadata = meta

    def get_content(self) -> str:
        return self.text


class NodeWithScore:
    def __init__(self, node: LINode, score: float) -> None:
        self.node = node
        self.score = score


class FunctionTool:
    def __init__(self, name: str) -> None:
        self.metadata = types.SimpleNamespace(name=name, description="d")


class ToolOutput:
    def __init__(self, content: str, raw_output: Any = None, is_error: bool = False) -> None:
        self.content = content
        self.raw_output = raw_output
        self.is_error = is_error


class FunctionAgent:
    def __init__(self, name: str) -> None:
        self.name = name


def _bound(**arguments: Any) -> Any:
    return types.SimpleNamespace(arguments=arguments)


def _llm_roundtrip(h: LlamaIndexHandler, sid: str, text: str, tool_call: bool = False) -> None:
    msgs = [LIMessage("system", "be brief"), LIMessage("user", "hi")]
    h.handle(
        _event("LLMChatStartEvent", sid=sid, messages=msgs, model_dict={"model": "gpt-li"}),
    )
    extra = (
        {
            "tool_calls": [
                types.SimpleNamespace(
                    id="c1",
                    function=types.SimpleNamespace(name="lookup", arguments='{"q": "x"}'),
                )
            ]
        }
        if tool_call
        else {}
    )
    raw = types.SimpleNamespace(
        usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=3), model="gpt-li"
    )
    reply = LIChatResponse(LIMessage("assistant", text, additional_kwargs=extra), raw)
    h.handle(_event("LLMChatEndEvent", sid=sid, messages=msgs, response=reply))


def test_li_chat_events_record_model_usage_and_tool_calls():
    h = llama_index_handler()
    with crashlab.record("li-chat"):
        _llm_roundtrip(h, "a", "calling", tool_call=True)
    run = crashlab.load("last")
    [chat] = run.chat_steps()
    assert chat.name == "gpt-li"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 7
    assert chat.attributes["gen_ai.usage.output_tokens"] == 3
    assert chat.request["system"] == "be brief"
    assert chat.request["messages"][0]["role"] == "user"
    assert chat.response["tool_calls"][0]["name"] == "lookup"
    assert chat.response["tool_calls"][0]["arguments"] == {"q": "x"}
    assert chat.response["finish_reason"] == "tool_calls"
    assert chat.response["usage"]["input_tokens"] == 7


def test_li_handler_is_noop_outside_a_run():
    h = llama_index_handler()
    _llm_roundtrip(h, "a", "x")
    h.span_enter(id_="FunctionTool.call-1-2-3-4-5", bound_args=_bound(), instance=FunctionTool("t"))
    h.span_exit(id_="FunctionTool.call-1-2-3-4-5", result=ToolOutput("r"))
    h.span_drop(id_="zzz", err=RuntimeError("x"))
    h.close()


def test_li_retrieval_is_visible_to_rag():
    h = llama_index_handler()
    nodes = [
        NodeWithScore(LINode("doc-1", "Refunds take 14 days.", source="policy.md"), 0.91),
        NodeWithScore(LINode("doc-2", "Shipping is free over 50 dollars."), 0.4),
    ]
    with crashlab.record("li-rag") as run:
        h.handle(_event("RetrievalStartEvent", sid="r1", str_or_query_bundle="refund window?"))
        h.handle(
            _event("RetrievalEndEvent", sid="r1", str_or_query_bundle="refund window?", nodes=nodes)
        )
        _llm_roundtrip(h, "c", "Refunds take 14 days [doc-1].")
        run.output = "Refunds take 14 days [doc-1]."
    run = crashlab.load("last")
    docs = rag.retrieved(run)
    assert [d["id"] for _, d in docs] == ["doc-1", "doc-2"]
    assert docs[0][1]["text"] == "Refunds take 14 days."
    assert docs[0][1]["score"] == pytest.approx(0.91)
    assert docs[0][1]["metadata"] == {"source": "policy.md"}
    [call] = run.tool_calls("search_docs")
    assert call.request["args"] == {"query": "refund window?"}
    assert rag.retrieved_ids(run, expected=["doc-1"]) is None
    assert rag.citations_were_retrieved(run, run.output) is None


def test_li_custom_retrieval_tool_name():
    h = llama_index_handler(retrieval_tool="kb")
    with crashlab.record("li-kb"):
        h.handle(_event("RetrievalEndEvent", sid="r", str_or_query_bundle="q", nodes=[]))
    run = crashlab.load("last")
    assert run.tool_calls("kb") and not run.tool_calls("search_docs")


def test_li_tool_span_records_args_result_and_agent_scope():
    h = llama_index_handler()
    tool_id = "FunctionTool.call-aaaa-bbbb-cccc-dddd-eeee"
    agent_id = "FunctionAgent.take_step-1111-2222-3333-4444-5555"
    with crashlab.record("li-tools"):
        h.span_enter(id_=agent_id, bound_args=_bound(), instance=FunctionAgent("ResearchAgent"))
        _llm_roundtrip(h, "x", "using tool", tool_call=True)
        tool = FunctionTool("lookup")
        h.span_enter(id_=tool_id, bound_args=_bound(self=tool, kwargs={"q": "py"}), instance=tool)
        h.span_exit(id_=tool_id, instance=tool, result=ToolOutput("py 3.14", raw_output="py 3.14"))
        h.span_exit(id_=agent_id, instance=None, result=None)
        _llm_roundtrip(h, "y", "after scope")
    run = crashlab.load("last")
    [tool_step] = run.tool_calls("lookup")
    assert tool_step.request["args"] == {"q": "py"}
    assert tool_step.result == "py 3.14"
    assert tool_step.agent == "ResearchAgent"
    assert run.chat_steps()[0].agent == "ResearchAgent"
    assert run.chat_steps()[1].agent is None


def test_li_multi_agent_attribution():
    h = llama_index_handler()
    with crashlab.record("li-multi"):
        for n, agent in enumerate(("ResearchAgent", "WriteAgent")):
            sid = f"FunctionAgent.take_step-{n}-2-3-4-5"
            h.span_enter(id_=sid, bound_args=_bound(), instance=FunctionAgent(agent))
            _llm_roundtrip(h, f"s{n}", f"from {agent}")
            h.span_exit(id_=sid, result=None)
    assert crashlab.load("last").agents == ["ResearchAgent", "WriteAgent"]


def test_li_tool_error_output_and_span_drop():
    h = llama_index_handler()
    tid = "FunctionTool.call-1-2-3-4-5"
    with crashlab.record("li-err"):
        tool = FunctionTool("boom")
        h.span_enter(id_=tid, bound_args=_bound(kwargs={"n": 1}), instance=tool)
        h.span_exit(id_=tid, result=ToolOutput("inventory empty", is_error=True))
        tool2 = FunctionTool("crash")
        t2 = "FunctionTool.acall-9-2-3-4-5"
        h.span_enter(id_=t2, bound_args=_bound(args=({"k": "v"},)), instance=tool2)
        h.span_drop(id_=t2, err=ValueError("kaput"))
        h.handle(
            _event("LLMChatStartEvent", sid="L", messages=[LIMessage("user", "hi")], model_dict={})
        )
        h.handle(_event("SpanDropEvent", sid="L", err_str="rate limited"))
    run = crashlab.load("last")
    boom = run.tool_calls("boom")[0]
    assert boom.error and boom.error.is_tool_error and "inventory empty" in boom.error.message
    crash = run.tool_calls("crash")[0]
    assert crash.request["args"] == {"k": "v"}
    assert "kaput" in crash.error.message
    [chat] = run.chat_steps()
    assert chat.error and "rate limited" in chat.error.message


def test_li_agent_tool_call_event_fallback_without_tool_span():
    h = llama_index_handler()
    with crashlab.record("li-evt"):
        tool = types.SimpleNamespace(name="lookup", description="d")
        h.handle(_event("AgentToolCallEvent", sid="t", arguments='{"q": "z"}', tool=tool))
        h.handle(_event("AgentRunStepEndEvent", sid="t"))
    [step] = crashlab.load("last").tool_calls("lookup")
    assert step.request["args"] == {"q": "z"}
    assert step.attributes["crashlab.llama_index.result_unavailable"] is True


def test_li_exception_event_without_pending_span():
    h = llama_index_handler()
    with crashlab.record("li-exc"):
        h.handle(_event("ExceptionEvent", sid="nothing", exception=RuntimeError("bad index")))
    run = crashlab.load("last")
    [step] = [s for s in run.steps if s.operation == "custom"]
    assert "bad index" in step.error.message


def test_li_no_double_recording_with_wrapped_client_and_tool():
    from tests.fakes import FakeOpenAI, text_response

    client = crashlab.wrap(FakeOpenAI([text_response(1, "from wrap")]))

    @crashlab.tool()
    def lookup(q: str) -> str:
        return "real"

    h = llama_index_handler()
    with crashlab.record("li-dedupe"):
        msgs = [LIMessage("user", "hi")]
        h.handle(_event("LLMChatStartEvent", sid="a", messages=msgs, model_dict={"model": "m"}))
        client.chat.completions.create(
            model="fake-model", messages=[{"role": "user", "content": "hi"}]
        )
        h.handle(
            _event(
                "LLMChatEndEvent",
                sid="a",
                messages=msgs,
                response=LIChatResponse(LIMessage("assistant", "from callback")),
            )
        )
        tool = FunctionTool("lookup")
        tid = "FunctionTool.call-1-2-3-4-5"
        h.span_enter(id_=tid, bound_args=_bound(kwargs={"q": "a"}), instance=tool)
        lookup(q="a")
        h.span_exit(id_=tid, result=ToolOutput("from callback", raw_output="from callback"))
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat", "execute_tool"]
    assert run.steps[0].response["messages"][0]["parts"][0]["content"] == "from wrap"
    assert run.steps[1].result == "real"


def test_li_completion_events_and_attach_detach_with_fake_dispatcher():
    h = llama_index_handler()

    class Dispatcher:
        def __init__(self) -> None:
            self.event_handlers: list[Any] = []
            self.span_handlers: list[Any] = []

        def add_event_handler(self, handler: Any) -> None:
            self.event_handlers.append(handler)

        def add_span_handler(self, handler: Any) -> None:
            self.span_handlers.append(handler)

    d = Dispatcher()
    assert h.attach(d) is h
    assert d.event_handlers == [h] and d.span_handlers == [h]
    assert h.class_name() == "CrashlabLlamaIndexHandler"
    with crashlab.record("li-comp"):
        h.handle(
            _event("LLMCompletionStartEvent", sid="c", prompt="2+2?", model_dict={"model": "m"})
        )
        h.handle(
            _event(
                "LLMCompletionEndEvent",
                sid="c",
                prompt="2+2?",
                response=types.SimpleNamespace(text="4", raw=None),
            )
        )
    h.detach()
    assert d.event_handlers == [] and d.span_handlers == []
    [chat] = crashlab.load("last").chat_steps()
    assert chat.response["messages"][0]["parts"][0]["content"] == "4"


def test_li_completion_nested_in_chat_is_not_recorded_twice():
    h = llama_index_handler()
    msgs = [LIMessage("user", "hi")]
    with crashlab.record("li-nested"):
        h.handle(_event("LLMChatStartEvent", sid="outer", messages=msgs, model_dict={"model": "m"}))
        h.handle(_event("LLMCompletionStartEvent", sid="inner", prompt="hi", model_dict={}))
        h.handle(
            _event(
                "LLMCompletionEndEvent",
                sid="inner",
                prompt="hi",
                response=types.SimpleNamespace(text="inner", raw=None),
            )
        )
        h.handle(
            _event(
                "LLMChatEndEvent",
                sid="outer",
                messages=msgs,
                response=LIChatResponse(LIMessage("assistant", "outer")),
            )
        )
    [chat] = crashlab.load("last").chat_steps()
    assert chat.response["messages"][0]["parts"][0]["content"] == "outer"


def test_li_node_metadata_carries_ref_doc_id():
    node = LINode("n1", "text", source="a.md")
    node.ref_doc_id = "doc-A"
    assert fx._doc_of(NodeWithScore(node, 0.5))["metadata"] == {
        "source": "a.md",
        "ref_doc_id": "doc-A",
    }


def test_handler_failures_warn_but_never_raise():
    h = llama_index_handler()

    class Bad:
        @property
        def messages(self) -> Any:
            raise RuntimeError("boom")

    Bad.__name__ = "LLMChatStartEvent"
    with crashlab.record("li-bad"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h.handle(Bad())
    assert any("LlamaIndex handler" in str(w.message) for w in caught)


class FakeQueryEngine:
    def __init__(self, nodes: list[Any], boom: bool = False) -> None:
        self.nodes = nodes
        self.boom = boom

    def query(self, q: Any) -> Any:
        if self.boom:
            raise RuntimeError("index offline")
        return types.SimpleNamespace(response="14 days [doc-1]", source_nodes=self.nodes)

    async def aquery(self, q: Any) -> Any:
        return self.query(q)


def test_wrap_query_engine_records_sources_for_rag_and_is_idempotent():
    nodes = [NodeWithScore(LINode("doc-1", "Refunds take 14 days."), 0.9)]
    engine = wrap_query_engine(FakeQueryEngine(nodes), name="kb-bot")
    assert wrap_query_engine(engine) is engine
    assert engine.query("window?").response == "14 days [doc-1]"
    run = crashlab.load("last")
    assert run.name == "kb-bot"
    assert run.output == "14 days [doc-1]"
    assert run.input == {"query": "window?"}
    assert [d["id"] for _, d in rag.retrieved(run)] == ["doc-1"]
    assert len(run.tool_calls("search_docs")) == 1


def test_wrap_query_engine_with_handler_does_not_double_record_retrieval():
    nodes = [NodeWithScore(LINode("doc-1", "Refunds take 14 days."), 0.9)]
    h = llama_index_handler()

    class Engine(FakeQueryEngine):
        def query(self, q: Any) -> Any:
            h.handle(_event("RetrievalStartEvent", sid="r", str_or_query_bundle=q))
            h.handle(_event("RetrievalEndEvent", sid="r", str_or_query_bundle=q, nodes=nodes))
            return super().query(q)

    engine = wrap_query_engine(Engine(nodes))
    engine.query("window?")
    assert len(crashlab.load("last").tool_calls("search_docs")) == 1


def test_wrap_query_engine_async_error_and_existing_record():
    engine = wrap_query_engine(FakeQueryEngine([], boom=True))
    with pytest.raises(RuntimeError):
        engine.query("x")
    assert crashlab.load("last").status == "error"
    ok = wrap_query_engine(FakeQueryEngine([NodeWithScore(LINode("d", "t"), 0.1)]))

    async def go() -> None:
        with crashlab.record("outer-li") as run:
            await ok.aquery("q")
            assert run.output == "14 days [doc-1]"

    asyncio.run(go())
    assert crashlab.load("last").name == "outer-li"
    with pytest.raises(TypeError):
        wrap_query_engine(None)


# ---------------------------------------------------------------------------
# smolagents
# ---------------------------------------------------------------------------


class TokenUsage:
    def __init__(self, i: int, o: int) -> None:
        self.input_tokens, self.output_tokens, self.total_tokens = i, o, i + o


class Timing:
    def __init__(self) -> None:
        self.start_time = 1_700_000_000.0
        self.duration = 0.25


class SmolMessage:
    def __init__(self, role: str, text: str, tool_calls: list[Any] | None = None) -> None:
        self.role = role
        self.content = [{"type": "text", "text": text}]
        self.tool_calls = tool_calls
        self.token_usage = TokenUsage(11, 5)


class ToolCall:
    def __init__(self, name: str, arguments: Any, id: str = "t1") -> None:
        self.name, self.arguments, self.id = name, arguments, id


class AgentError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ActionStep:
    def __init__(self, n: int, **kw: Any) -> None:
        self.step_number = n
        self.timing = Timing()
        self.model_input_messages = [SmolMessage("system", "be brief"), SmolMessage("user", "task")]
        self.tool_calls: list[Any] = []
        self.error: Any = None
        self.model_output_message = None
        self.model_output = None
        self.observations: Any = None
        self.action_output: Any = None
        self.token_usage = TokenUsage(11, 5)
        for k, v in kw.items():
            setattr(self, k, v)


class PlanningStep:
    def __init__(self) -> None:
        self.timing = Timing()
        self.model_input_messages = [SmolMessage("user", "plan")]
        self.model_output_message = SmolMessage("assistant", "1. look up price")
        self.plan = "1. look up price"
        self.token_usage = TokenUsage(4, 4)


class FinalAnswerStep:
    def __init__(self, output: Any) -> None:
        self.output = output


class FakeSmolModel:
    model_id = "smol-model"


class FakeSmolAgent:
    """ToolCallingAgent-like: run() builds memory steps, calls step_callbacks."""

    def __init__(self, name: str | None = None, tool_error: bool = False, **kw: Any) -> None:
        self.name = name
        self.model = FakeSmolModel()
        self.memory = types.SimpleNamespace(steps=[])
        self.step_callbacks: Any = []
        self.managed_agents: dict[str, Any] = kw.get("managed_agents", {})
        self.tool_error = tool_error
        self.answer = "54"

    def _steps(self, task: str) -> list[Any]:
        calls = [ToolCall("get_price", {"item": "widget"})]
        s1 = ActionStep(
            1,
            tool_calls=calls,
            model_output_message=SmolMessage("assistant", "checking", calls),
            observations="4.5",
        )
        if self.tool_error:
            s1.error = AgentError("price service down")
            s1.observations = None
        s2 = ActionStep(
            2,
            tool_calls=[ToolCall("final_answer", {"answer": "54"})],
            model_output_message=SmolMessage("assistant", "done"),
            observations="54",
            action_output="54",
        )
        return [s1, s2]

    def _fire(self, step: Any) -> None:
        cbs = self.step_callbacks
        for cb in cbs if isinstance(cbs, list) else []:
            cb(step, agent=self)

    def _gen(self, task: str) -> Any:
        for s in self._steps(task):
            self.memory.steps.append(s)
            self._fire(s)
            yield s
        yield FinalAnswerStep("54")

    def run(self, task: str, stream: bool = False, reset: bool = True, **kw: Any) -> Any:
        if reset:
            self.memory.steps.clear()
        gen = self._gen(task)
        if stream:
            return gen
        for _ in gen:
            pass
        return self.answer


def test_wrap_smolagent_records_steps_tools_errors_and_usage():
    agent = wrap_smolagent(FakeSmolAgent(), name="smol-prices")
    assert wrap_smolagent(agent) is agent
    assert len(agent.step_callbacks) == 1  # installed once
    assert agent.run("12 widgets?") == "54"
    run = crashlab.load("last")
    assert run.name == "smol-prices" and run.output == "54"
    assert run.input["task"] == "12 widgets?"
    assert [s.operation for s in run.steps] == ["chat", "execute_tool", "chat", "execute_tool"]
    chat = run.steps[0]
    assert chat.name == "smol-model"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 11
    assert chat.response["tool_calls"][0]["name"] == "get_price"
    assert chat.request["system"] == "be brief"
    assert run.tool_calls("get_price")[0].result == "4.5"
    assert run.tool_calls("final_answer")[0].result == "54"
    assert run.steps[0].started_at.year == 2023


def test_wrap_smolagent_tool_error_is_failing_tool_step():
    agent = wrap_smolagent(FakeSmolAgent(tool_error=True))
    agent.run("x")
    run = crashlab.load("last")
    [step] = run.tool_calls("get_price")
    assert step.error and step.error.is_tool_error and "price service down" in step.error.message


def test_wrap_smolagent_parsing_error_without_tool_call_is_recorded():
    agent = FakeSmolAgent()
    agent._steps = lambda task: [  # type: ignore[method-assign]
        ActionStep(
            1, model_output_message=SmolMessage("assistant", "garbage"), error=AgentError("no code")
        )
    ]
    wrap_smolagent(agent).run("x")
    run = crashlab.load("last")
    [err] = [s for s in run.steps if s.operation == "custom"]
    assert "no code" in err.error.message


def test_wrap_smolagent_stream_records_steps_and_final_answer():
    agent = wrap_smolagent(FakeSmolAgent(), name="smol-stream")
    items = list(agent.run("q", stream=True))
    assert type(items[-1]).__name__ == "FinalAnswerStep"
    run = crashlab.load("last")
    assert run.name == "smol-stream" and run.output == "54"
    assert [s.operation for s in run.steps].count(
        "chat"
    ) == 2  # no double recording (callback + stream)
    assert len(run.tool_calls()) == 2


def test_wrap_smolagent_without_callback_support_reconciles_memory():
    agent = FakeSmolAgent()
    agent.step_callbacks = None  # nothing to hook: memory is read after run()
    wrap_smolagent(agent).run("q")
    run = crashlab.load("last")
    assert len(run.chat_steps()) == 2 and len(run.tool_calls()) == 2


def test_wrap_smolagent_registry_style_callbacks(monkeypatch):
    registered: list[tuple[Any, Any]] = []

    class Registry:
        def register(self, cls: Any, cb: Any) -> None:
            registered.append((cls, cb))

    mod = types.ModuleType("smolagents.memory")
    mod.ActionStep, mod.PlanningStep = ActionStep, PlanningStep  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "smolagents.memory", mod)
    agent = FakeSmolAgent()
    agent.step_callbacks = Registry()
    wrap_smolagent(agent)
    assert {c for c, _ in registered} == {ActionStep, PlanningStep}
    cb = registered[0][1]
    with crashlab.record("manual"):
        cb(ActionStep(1, model_output_message=SmolMessage("assistant", "hi")), agent=agent)
    assert len(crashlab.load("last").chat_steps()) == 1


def test_crashlab_step_callback_standalone_and_noop_outside_run():
    cb = crashlab_step_callback(name="solo")
    step = ActionStep(
        1,
        tool_calls=[ToolCall("t", {"a": 1})],
        model_output_message=SmolMessage("assistant", "x"),
        observations="ok",
    )
    cb(step)  # outside a record: nothing, no error
    with crashlab.record("cb"):
        cb(step, agent=None)
        cb(step, agent=None)  # same step twice: recorded once
        cb(PlanningStep())
    run = crashlab.load("last")
    assert len(run.tool_calls("t")) == 1
    assert run.agents == ["solo"]
    assert len(run.chat_steps()) == 2  # action + planning
    assert run.chat_steps()[1].attributes["crashlab.smolagents.step"] == "planning"


def test_wrap_smolagent_managed_agents_are_attributed():
    worker = FakeSmolAgent(name="catalog")
    worker.answer = "4.5 each"
    manager = FakeSmolAgent(name="manager", managed_agents={"catalog": worker})

    def manager_steps(task: str) -> list[Any]:
        worker.run(f"price for {task}")  # what the managed agent's tool call does
        return [
            ActionStep(
                1,
                tool_calls=[ToolCall("catalog", {"task": task})],
                model_output_message=SmolMessage("assistant", "delegate"),
                observations="54",
            )
        ]

    manager._steps = manager_steps  # type: ignore[method-assign]
    wrap_smolagent(manager, name="smol-managed")
    manager.run("12 widgets")
    run = crashlab.load("last")
    assert run.name == "smol-managed"
    assert run.output == "54"  # the managed agent's nested run must not overwrite it
    assert set(run.agents) == {"manager", "catalog"}
    assert {s.agent for s in run.chat_steps() if s.agent == "catalog"} == {"catalog"}
    assert run.tool_calls("get_price", agent="catalog")
    assert run.tool_calls("catalog", agent="manager")


def test_wrap_smolagent_no_double_recording_with_wrapped_client_and_tool():
    from tests.fakes import FakeOpenAI, text_response

    client = crashlab.wrap(FakeOpenAI([text_response(1, "from wrap")]))

    @crashlab.tool()
    def get_price(item: str) -> str:
        return "4.5-real"

    agent = FakeSmolAgent()

    def steps(task: str) -> list[Any]:
        client.chat.completions.create(
            model="fake-model", messages=[{"role": "user", "content": "q"}]
        )
        get_price(item="widget")
        calls = [ToolCall("get_price", {"item": "widget"})]
        return [
            ActionStep(
                1,
                tool_calls=calls,
                model_output_message=SmolMessage("assistant", "x", calls),
                observations="4.5",
            )
        ]

    agent._steps = steps  # type: ignore[method-assign]
    wrap_smolagent(agent).run("q")
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat", "execute_tool"]
    assert run.steps[1].result == "4.5-real"
    assert run.steps[0].response["messages"][0]["parts"][0]["content"] == "from wrap"


def test_wrap_smolagent_error_propagates_and_requires_agent():
    agent = FakeSmolAgent()

    def boom(task: str, **kw: Any) -> Any:
        raise RuntimeError("model down")

    agent.run = boom  # type: ignore[method-assign]
    wrap_smolagent(agent)
    with pytest.raises(RuntimeError):
        agent.run("x")
    assert crashlab.load("last").status == "error"
    with pytest.raises(TypeError):
        wrap_smolagent(None)


# ---------------------------------------------------------------------------
# Haystack
# ---------------------------------------------------------------------------


class ChatMessage:
    def __init__(self, role: str, text: str, tool_calls: Any = None, results: Any = None) -> None:
        self.role = Role(role)
        self.text = text
        self.tool_calls = tool_calls or []
        self.tool_call_results = results or []


class Document:
    def __init__(self, id: str, content: str, score: float | None = None, **meta: Any) -> None:
        self.id, self.content, self.score, self.meta = id, content, score, meta


class InMemoryBM25Retriever:
    def run(self, query: str, top_k: int = 2) -> dict[str, Any]:
        docs = [
            Document("d1", "Refunds take 14 days.", 1.5, file="policy.md"),
            Document("d2", "Shipping is free.", 0.7),
        ]
        return {"documents": docs[:top_k]}


class PromptBuilder:
    def run(self, documents: list[Any], question: str) -> dict[str, Any]:
        return {"prompt": f"Context: {len(documents)} docs. Q: {question}"}


class OpenAIChatGenerator:
    def __init__(self, reply: str = "Refunds take 14 days.", fail: bool = False) -> None:
        self.reply, self.fail = reply, fail

    def run(self, messages: list[Any]) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("rate limited")
        return {
            "replies": [ChatMessage("assistant", self.reply)],
            "meta": [
                {
                    "model": "gpt-hay",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26},
                }
            ],
        }


class Embedder:
    def run(self, text: str) -> dict[str, Any]:
        return {"embedding": [0.1] * 64}


class ToolInvoker:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def run(self, messages: list[Any]) -> dict[str, Any]:
        origin = types.SimpleNamespace(tool_name="get_price", arguments={"item": "tea"}, id="c1")
        res = types.SimpleNamespace(
            result="tool blew up" if self.fail else "3 USD", origin=origin, error=self.fail
        )
        return {"tool_messages": [ChatMessage("tool", "", results=[res])]}


class FakePipeline:
    def __init__(self, **components: Any) -> None:
        self.components = components

    def walk(self) -> Any:
        return iter(self.components.items())

    def run(self, data: dict[str, Any], include_outputs_from: Any = None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        docs = ret = None
        if "retriever" in self.components:
            ret = self.components["retriever"].run(**data["retriever"])
            docs = ret["documents"]
        if "embedder" in self.components:
            self.components["embedder"].run(text="x")
        if "prompt" in self.components:
            prompt = self.components["prompt"].run(documents=docs or [], question="q")
        else:
            prompt = {"prompt": "p"}
        msgs = [
            ChatMessage("system", "Answer from context."),
            ChatMessage("user", prompt["prompt"]),
        ]
        llm = self.components["llm"].run(messages=msgs)
        out["llm"] = llm
        if "invoker" in self.components:
            out["invoker"] = self.components["invoker"].run(messages=llm["replies"])
        return out

    async def run_async(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.run(data)


def _rag_pipeline(**extra: Any) -> FakePipeline:
    return FakePipeline(
        retriever=InMemoryBM25Retriever(),
        prompt=PromptBuilder(),
        llm=OpenAIChatGenerator(),
        **extra,
    )


def test_wrap_pipeline_records_components_retrieval_chat_and_output():
    pipe = wrap_pipeline(_rag_pipeline(embedder=Embedder()), name="hay-rag")
    assert wrap_pipeline(pipe) is pipe
    result = pipe.run({"retriever": {"query": "refund window?"}})
    assert result["llm"]["replies"][0].text == "Refunds take 14 days."
    run = crashlab.load("last")
    assert run.name == "hay-rag"
    assert run.output == "Refunds take 14 days."
    ops = [(s.operation, s.name) for s in run.steps]
    assert ops == [
        ("execute_tool", "search_docs"),
        ("custom", "embedder"),
        ("custom", "prompt"),
        ("chat", "gpt-hay"),
    ]
    docs = rag.retrieved(run)
    assert [d["id"] for _, d in docs] == ["d1", "d2"]
    assert docs[0][1]["metadata"] == {"file": "policy.md"}
    assert run.tool_calls("search_docs")[0].request["args"]["query"] == "refund window?"
    chat = run.chat_steps()[0]
    assert chat.attributes["gen_ai.usage.input_tokens"] == 20
    assert chat.attributes["gen_ai.usage.output_tokens"] == 6
    assert chat.request["system"] == "Answer from context."
    # embeddings are summarised, never stored as 64 floats
    emb = [s for s in run.steps if s.name == "embedder"][0]
    assert emb.response["output"] == {"embedding": "[64 numbers]"}


def test_wrap_pipeline_tool_invoker_steps_and_errors():
    pipe = wrap_pipeline(_rag_pipeline(invoker=ToolInvoker()))
    pipe.run({"retriever": {"query": "q"}})
    [ok] = crashlab.load("last").tool_calls("get_price")
    assert ok.request["args"] == {"item": "tea"} and ok.result == "3 USD"
    bad = wrap_pipeline(_rag_pipeline(invoker=ToolInvoker(fail=True)))
    bad.run({"retriever": {"query": "q"}})
    [step] = crashlab.load("last").tool_calls("get_price")
    assert step.error and step.error.is_tool_error and "tool blew up" in step.error.message


def test_wrap_pipeline_component_error_is_recorded_and_raised():
    pipe = wrap_pipeline(
        FakePipeline(retriever=InMemoryBM25Retriever(), llm=OpenAIChatGenerator(fail=True))
    )
    with pytest.raises(RuntimeError, match="rate limited"):
        pipe.run({"retriever": {"query": "q"}})
    run = crashlab.load("last")
    assert run.status == "error"
    chat = run.chat_steps()[0]
    assert chat.error and "rate limited" in chat.error.message


def test_wrap_pipeline_custom_retriever_names_and_tool_name():
    class VectorLookup:
        def run(self, query: str) -> dict[str, Any]:
            return {"documents": [Document("v1", "vec text")]}

    pipe = FakePipeline(retriever=VectorLookup(), llm=OpenAIChatGenerator())
    wrap_pipeline(pipe, retrievers=["retriever"], retrieval_tool="kb")
    pipe.run({"retriever": {"query": "q"}})
    run = crashlab.load("last")
    assert [d["id"] for _, d in rag.retrieved(run, "kb")] == ["v1"]


def test_wrap_pipeline_run_async_and_existing_record():
    pipe = wrap_pipeline(_rag_pipeline(), name="hay-async")

    async def go() -> None:
        with crashlab.record("outer-hay") as run:
            await pipe.run_async({"retriever": {"query": "q"}})
            assert run.output == "Refunds take 14 days."

    asyncio.run(go())
    run = crashlab.load("last")
    assert run.name == "outer-hay" and len(run.chat_steps()) == 1


def test_wrap_pipeline_no_double_recording_with_wrapped_client():
    from tests.fakes import FakeOpenAI, text_response

    client = crashlab.wrap(FakeOpenAI([text_response(1, "from wrap")]))

    class WrappedGenerator(OpenAIChatGenerator):
        def run(self, messages: list[Any]) -> dict[str, Any]:
            client.chat.completions.create(
                model="fake-model", messages=[{"role": "user", "content": "hi"}]
            )
            return super().run(messages)

    pipe = wrap_pipeline(FakePipeline(retriever=InMemoryBM25Retriever(), llm=WrappedGenerator()))
    pipe.run({"retriever": {"query": "q"}})
    run = crashlab.load("last")
    assert len(run.chat_steps()) == 1
    assert run.chat_steps()[0].response["messages"][0]["parts"][0]["content"] == "from wrap"


def test_wrap_pipeline_agent_component_is_attributed():
    class Agent:
        def run(self, messages: list[Any]) -> dict[str, Any]:
            return OpenAIChatGenerator().run(messages)

    pipe = wrap_pipeline(FakePipeline(retriever=InMemoryBM25Retriever(), llm=Agent()))
    pipe.run({"retriever": {"query": "q"}})
    assert crashlab.load("last").chat_steps()[0].agent == "llm"


def test_wrap_pipeline_requires_pipeline():
    with pytest.raises(TypeError):
        wrap_pipeline(None)


def test_components_outside_a_run_are_untouched():
    pipe = wrap_pipeline(_rag_pipeline(), name="no-run")
    # pipeline.run opens its own run; component calls alone (no run) must not fail
    comp = pipe.components["retriever"]
    assert comp.run(query="q")["documents"][0].id == "d1"


def _span_run(
    tracer: HaystackTracer, name: str, ctype: str, inp: Any, out: Any, fail: bool = False
):
    with tracer.trace(
        "haystack.component.run",
        tags={"haystack.component.name": name, "haystack.component.type": ctype},
    ) as span:
        span.set_content_tag("haystack.component.input", inp)
        if fail:
            raise RuntimeError("component failed")
        span.set_content_tag("haystack.component.output", out)


def test_haystack_tracer_records_component_spans():
    tracer = HaystackTracer(name="hay-traced")
    assert tracer.current_span() is None
    with tracer.trace("haystack.pipeline.run", tags={}) as pspan:
        assert tracer.current_span() is pspan
        pspan.set_tag("haystack.pipeline.input_data", {"retriever": {"query": "refunds?"}})
        _span_run(
            tracer, "retriever", "InMemoryBM25Retriever", {"query": "refunds?"},
            {"documents": [Document("d1", "Refunds take 14 days.", 1.0)]},
        )  # fmt: skip
        _span_run(
            tracer, "llm", "OpenAIChatGenerator", {"messages": [ChatMessage("user", "refunds?")]},
            {
                "replies": [ChatMessage("assistant", "14 days")],
                "meta": [
                    {"model": "gpt-hay", "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
                ],
            },
        )  # fmt: skip
        pspan.set_tag(
            "haystack.pipeline.output_data",
            {"llm": {"replies": [ChatMessage("assistant", "14 days")]}},
        )
    run = crashlab.load("last")
    assert run.name == "hay-traced" and run.output == "14 days"
    assert [d["id"] for _, d in rag.retrieved(run)] == ["d1"]
    assert run.chat_steps()[0].attributes["gen_ai.usage.input_tokens"] == 3


def test_haystack_tracer_error_span_and_no_double_record_with_wrap_pipeline():
    tracer = HaystackTracer()
    with crashlab.record("traced-err"):
        with pytest.raises(RuntimeError):
            _span_run(tracer, "llm", "OpenAIChatGenerator", {"messages": []}, None, fail=True)
    chat = crashlab.load("last").chat_steps()[0]
    assert chat.error and "component failed" in chat.error.message

    class TracedPipeline(FakePipeline):
        """Like haystack: every component run is inside a tracer span."""

        def run(self, data: dict[str, Any], include_outputs_from: Any = None) -> dict[str, Any]:
            comp = self.components["llm"]
            with tracer.trace(
                "haystack.component.run",
                tags={
                    "haystack.component.name": "llm",
                    "haystack.component.type": "OpenAIChatGenerator",
                },
            ) as span:
                msgs = [ChatMessage("user", "hi")]
                span.set_content_tag("haystack.component.input", {"messages": msgs})
                out = comp.run(messages=msgs)
                span.set_content_tag("haystack.component.output", out)
            return {"llm": out}

    pipe = wrap_pipeline(TracedPipeline(llm=OpenAIChatGenerator()))
    pipe.run({})
    assert len(crashlab.load("last").chat_steps()) == 1


def test_haystack_tracer_outside_run_is_noop():
    tracer = HaystackTracer()
    _span_run(tracer, "llm", "OpenAIChatGenerator", {"messages": []}, {"replies": []})
