"""crashlab.integrations.frameworks: OpenAI Agents SDK, PydanticAI, CrewAI.

Duck-typed fakes only. No module-level agents / pydantic_ai / crewai / openai
imports, so the file runs in the core-only (venv-nosdk) environment.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import crashlab
from crashlab.integrations import (
    CrewAICallback,
    OpenAIAgentsTracingProcessor,
    PydanticAIAdapter,
    crashlab_tracing_processor,
    wrap_crew,
    wrap_pydantic_ai_agent,
)
from crashlab.integrations.frameworks import CrewAICallback as FrameworksCrewAICallback
from crashlab.integrations.frameworks import (
    crashlab_tracing_processor as frameworks_processor,
)

# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_import_does_not_need_optional_sdks():
    banned = ("agents", "pydantic_ai", "crewai")
    before = {name: name in sys.modules for name in banned}
    import importlib

    import crashlab.integrations.frameworks as fw

    importlib.reload(fw)
    for name in banned:
        if not before[name]:
            assert name not in sys.modules, name
    proc = fw.crashlab_tracing_processor()
    assert callable(proc.on_trace_start)
    assert callable(proc.on_span_end)
    assert callable(proc.shutdown)
    assert callable(proc.force_flush)
    assert proc.__class__.__name__ == "OpenAIAgentsTracingProcessor"


def test_public_reexports_match_submodule():
    assert CrewAICallback is FrameworksCrewAICallback
    assert crashlab_tracing_processor is frameworks_processor
    assert callable(wrap_pydantic_ai_agent)
    assert callable(wrap_crew)
    adapter = PydanticAIAdapter()
    assert callable(adapter.wrap)


# ---------------------------------------------------------------------------
# OpenAI Agents SDK tracing processor
# ---------------------------------------------------------------------------


class FakeSpanData:
    def __init__(self, typ: str, **fields: Any) -> None:
        self.type = typ
        for key, value in fields.items():
            setattr(self, key, value)

    def export(self) -> dict[str, Any]:
        data = {"type": self.type}
        data.update({k: v for k, v in vars(self).items() if k != "type"})
        return data


class FakeSpan:
    def __init__(
        self,
        typ: str,
        *,
        span_id: str = "span-1",
        error: Any = None,
        **fields: Any,
    ) -> None:
        self.span_id = span_id
        self.trace_id = "trace-1"
        self.parent_id = None
        self.error = error
        self.started_at = "2026-11-20T00:00:00+00:00"
        self.ended_at = "2026-11-20T00:00:01+00:00"
        self.span_data = FakeSpanData(typ, **fields)


class FakeTrace:
    def __init__(self, name: str = "support-workflow", output: Any = None) -> None:
        self.name = name
        self.trace_id = "trace-1"
        self.output = output


def _drive_shop_trace(proc: OpenAIAgentsTracingProcessor) -> None:
    proc.on_span_start(FakeSpan("agent", span_id="a1", name="shopper"))
    proc.on_span_start(
        FakeSpan(
            "generation",
            span_id="g1",
            model="gpt-x",
            input=[
                {"role": "system", "content": "You sell tea."},
                {"role": "user", "content": "price of tea?"},
            ],
            output=[{"role": "assistant", "content": "let me look it up"}],
            usage={"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
            model_config={"temperature": 0},
        )
    )
    proc.on_span_end(
        FakeSpan(
            "generation",
            span_id="g1",
            model="gpt-x",
            input=[
                {"role": "system", "content": "You sell tea."},
                {"role": "user", "content": "price of tea?"},
            ],
            output=[{"role": "assistant", "content": "let me look it up"}],
            usage={"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
            model_config={"temperature": 0},
        )
    )
    proc.on_span_start(
        FakeSpan(
            "function",
            span_id="f1",
            name="get_price",
            input='{"item": "tea"}',
            output={"item": "tea", "price": 3},
        )
    )
    proc.on_span_end(
        FakeSpan(
            "function",
            span_id="f1",
            name="get_price",
            input='{"item": "tea"}',
            output={"item": "tea", "price": 3},
        )
    )
    proc.on_span_start(FakeSpan("handoff", span_id="h1", from_agent="shopper", to_agent="cashier"))
    proc.on_span_end(FakeSpan("handoff", span_id="h1", from_agent="shopper", to_agent="cashier"))
    proc.on_span_end(FakeSpan("agent", span_id="a1", name="shopper"))


def test_tracing_processor_records_agent_generation_tool_and_handoff():
    proc = crashlab_tracing_processor(name="shop")
    with crashlab.record("shop-run") as run:
        proc.on_trace_start(FakeTrace("shop-workflow"))
        _drive_shop_trace(proc)
        proc.on_trace_end(FakeTrace("shop-workflow", output="tea is $3"))
        run.output = "tea is $3"
    src = crashlab.load(run.run_id)
    ops = [(s.operation, s.name) for s in src.steps]
    assert ("invoke_agent", "shopper") in ops
    assert ("chat", "gpt-x") in ops
    assert ("execute_tool", "get_price") in ops
    assert ("custom", "shopper -> cashier") in ops
    chat = next(s for s in src.steps if s.operation == "chat")
    assert chat.request["system"] == "You sell tea."
    assert chat.request["messages"][0]["role"] == "user"
    assert chat.response["messages"][0]["parts"][0]["content"] == "let me look it up"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 4
    assert chat.attributes["gen_ai.usage.output_tokens"] == 6
    assert chat.attributes["crashlab.api"] == "openai.agents"
    assert chat.agent == "shopper"
    tool = next(s for s in src.steps if s.operation == "execute_tool")
    assert tool.request["args"] == {"item": "tea"}
    assert tool.result == {"item": "tea", "price": 3}
    assert tool.agent == "shopper"
    handoff = next(s for s in src.steps if s.name == "shopper -> cashier")
    assert handoff.attributes["crashlab.handoff.from"] == "shopper"
    assert handoff.attributes["crashlab.handoff.to"] == "cashier"
    assert src.usage.input_tokens == 4
    assert src.usage.output_tokens == 6
    assert src.usage.llm_calls == 1
    assert src.usage.tool_calls == 1
    assert src.output == "tea is $3"
    assert src.findings == [] or isinstance(src.findings, list)


def test_tracing_processor_auto_record_on_trace():
    proc = crashlab_tracing_processor(name="auto-shop")
    proc.on_trace_start(FakeTrace("auto-workflow"))
    try:
        _drive_shop_trace(proc)
    finally:
        proc.on_trace_end(FakeTrace("auto-workflow", output="done"))
    src = crashlab.load("last")
    assert src.name == "auto-shop"
    assert any(s.operation == "chat" for s in src.steps)
    assert any(s.name == "shopper -> cashier" for s in src.steps)
    assert src.output == "done"
    proc.shutdown()
    proc.force_flush()


def test_tracing_processor_records_function_error_and_findings():
    proc = OpenAIAgentsTracingProcessor(auto_record=False)
    with crashlab.record("tool-boom"):
        proc.on_span_start(FakeSpan("function", span_id="f", name="place_order", input='{"n": 1}'))
        proc.on_span_end(
            FakeSpan(
                "function",
                span_id="f",
                name="place_order",
                input='{"n": 1}',
                output=None,
                error={"message": "inventory empty", "data": {"sku": "tea"}},
            )
        )
    src = crashlab.load("last")
    step = src.steps[0]
    assert step.operation == "execute_tool"
    assert step.error is not None
    assert step.error.is_tool_error
    assert "inventory empty" in step.error.message
    assert src.status == "ok"
    assert isinstance(src.findings, list)


def test_tracing_processor_ignores_events_outside_a_run():
    proc = OpenAIAgentsTracingProcessor(auto_record=False)
    proc.on_trace_start(FakeTrace())
    proc.on_span_start(FakeSpan("generation", span_id="g", model="m", input="hi", output="yo"))
    proc.on_span_end(FakeSpan("generation", span_id="g", model="m", input="hi", output="yo"))
    proc.on_span_end(FakeSpan("handoff", span_id="h", from_agent="a", to_agent="b"))
    proc.on_trace_end(FakeTrace())
    proc.shutdown()


def test_tracing_processor_skips_already_wrapped_chat_and_tool():
    from tests.fakes import FakeOpenAI, text_response

    proc = crashlab_tracing_processor()
    client = crashlab.wrap(FakeOpenAI([text_response(1, "from wrap")]))

    @crashlab.tool()
    def get_price(item: str) -> dict:
        return {"item": item, "price": 3}

    with crashlab.record("already-wrapped"):
        proc.on_span_start(FakeSpan("generation", span_id="g", model="fake-model"))
        client.chat.completions.create(
            model="fake-model", messages=[{"role": "user", "content": "hi"}]
        )
        proc.on_span_end(
            FakeSpan(
                "generation",
                span_id="g",
                model="fake-model",
                input=[{"role": "user", "content": "hi"}],
                output=[{"role": "assistant", "content": "from callback"}],
            )
        )
        proc.on_span_start(
            FakeSpan("function", span_id="f", name="get_price", input='{"item": "tea"}')
        )
        get_price(item="tea")
        proc.on_span_end(
            FakeSpan(
                "function",
                span_id="f",
                name="get_price",
                input='{"item": "tea"}',
                output={"item": "tea", "price": 99},
            )
        )
    src = crashlab.load("last")
    assert [s.operation for s in src.steps] == ["chat", "execute_tool"]
    assert src.steps[0].response["messages"][0]["parts"][0]["content"] == "from wrap"
    assert src.steps[0].attributes.get("crashlab.api") != "openai.agents"
    assert src.steps[1].result == {"item": "tea", "price": 3}


def test_tracing_processor_response_span_and_dict_span_data():
    proc = crashlab_tracing_processor()

    class DictSpan:
        def __init__(self) -> None:
            self.span_id = "r1"
            self.error = None
            self.span_data = {
                "type": "response",
                "model": "o-resp",
                "input": [{"role": "user", "content": "hi"}],
                "output": "hello",
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }

    with crashlab.record("resp-span"):
        proc.on_span_start(DictSpan())
        proc.on_span_end(DictSpan())
    src = crashlab.load("last")
    assert src.steps[0].operation == "chat"
    assert src.steps[0].name == "o-resp"
    assert src.steps[0].attributes["gen_ai.usage.input_tokens"] == 2
    assert src.steps[0].response["messages"][0]["parts"][0]["content"] == "hello"


# ---------------------------------------------------------------------------
# PydanticAI
# ---------------------------------------------------------------------------


class FakeUsage:
    def __init__(self, inp: int = 3, out: int = 5) -> None:
        self.input_tokens = inp
        self.output_tokens = out
        self.total_tokens = inp + out
        self.request_tokens = inp
        self.response_tokens = out


class FakePAIResult:
    def __init__(self, output: Any, usage: FakeUsage | None = None) -> None:
        self.output = output
        self._usage = usage or FakeUsage()

    def usage(self) -> FakeUsage:
        return self._usage


class FakePAITool:
    def __init__(self, name: str, fn: Any) -> None:
        self.name = name
        self.function = fn


class FakePAIModel:
    def __init__(self) -> None:
        self.model_name = "pai-model"
        self.calls: list[Any] = []

    def request(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append(messages)
        return types.SimpleNamespace(content="looked up", usage=FakeUsage(2, 2))


class FakePAIStream:
    def __init__(self, agent: FakePAIAgent, prompt: str) -> None:
        self._agent = agent
        self._prompt = prompt
        self._result: FakePAIResult | None = None
        self.output: Any = None

    async def __aenter__(self) -> FakePAIStream:
        self._result = self._agent.run_sync(self._prompt)
        self.output = self._result.output
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def usage(self) -> FakeUsage:
        assert self._result is not None
        return self._result.usage()


class FakePAIAgent:
    def __init__(self) -> None:
        self.name = "shop"
        self.model = FakePAIModel()

        def get_price(item: str) -> dict:
            return {"item": item, "price": 3}

        self._function_tools = {"get_price": FakePAITool("get_price", get_price)}

    def run_sync(self, prompt: str) -> FakePAIResult:
        price = self._function_tools["get_price"].function(item="tea")
        self.model.request([{"role": "user", "content": prompt}])
        return FakePAIResult(f"tea costs {price['price']}")

    async def run(self, prompt: str) -> FakePAIResult:
        return self.run_sync(prompt)

    def run_stream(self, prompt: str) -> FakePAIStream:
        return FakePAIStream(self, prompt)


def test_wrap_pydantic_ai_agent_run_sync_records_tools_usage_and_output():
    agent = wrap_pydantic_ai_agent(FakePAIAgent(), name="pai-shop")
    result = agent.run_sync("price of tea")
    assert "3" in result.output
    src = crashlab.load("last")
    assert src.name == "pai-shop"
    assert src.output == result.output
    ops = [s.operation for s in src.steps]
    assert "execute_tool" in ops
    assert "chat" in ops
    tool = next(s for s in src.steps if s.operation == "execute_tool")
    assert tool.name == "get_price"
    assert tool.request["args"]["item"] == "tea"
    assert tool.result == {"item": "tea", "price": 3}
    chat = next(s for s in src.steps if s.operation == "chat")
    assert chat.name == "pai-model"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 2
    assert src.usage.tool_calls == 1
    assert src.usage.llm_calls == 1
    assert src.findings == [] or isinstance(src.findings, list)
    # idempotent
    assert wrap_pydantic_ai_agent(agent) is agent


def test_wrap_pydantic_ai_agent_async_run_and_existing_record():
    agent = PydanticAIAdapter(FakePAIAgent(), name="pai-async").wrap()

    async def go() -> None:
        with crashlab.record("outer-pai") as run:
            result = await agent.run("price of tea")
            run.output = result.output

    asyncio.run(go())
    src = crashlab.load("last")
    assert src.name == "outer-pai"
    assert any(s.operation == "execute_tool" for s in src.steps)
    assert any(s.operation == "chat" for s in src.steps)
    assert "3" in str(src.output)


def test_wrap_pydantic_ai_agent_run_stream():
    agent = wrap_pydantic_ai_agent(FakePAIAgent(), name="pai-stream")

    async def go() -> Any:
        async with agent.run_stream("price of tea") as streamed:
            return streamed.output

    output = asyncio.run(go())
    src = crashlab.load("last")
    assert src.name == "pai-stream"
    assert output == src.output
    assert any(s.name == "get_price" for s in src.steps)


def test_wrap_pydantic_ai_agent_tool_error():
    class BoomAgent(FakePAIAgent):
        def __init__(self) -> None:
            self.name = "boom"
            self.model = FakePAIModel()

            def place_order(item: str) -> str:
                raise RuntimeError("sold out")

            self._function_tools = {"place_order": FakePAITool("place_order", place_order)}

        def run_sync(self, prompt: str) -> FakePAIResult:
            self._function_tools["place_order"].function(item="tea")
            return FakePAIResult("nope")

    agent = wrap_pydantic_ai_agent(BoomAgent(), name="pai-boom")
    try:
        agent.run_sync("order tea")
    except RuntimeError as exc:
        assert "sold out" in str(exc)
    else:
        raise AssertionError("expected tool error")
    src = crashlab.load("last")
    assert src.status == "error"
    tool = src.steps[0]
    assert tool.error is not None and tool.error.is_tool_error
    assert "sold out" in tool.error.message
    assert isinstance(src.findings, list)


def test_wrap_pydantic_ai_agent_requires_agent():
    try:
        wrap_pydantic_ai_agent(None)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------


class FakeAction:
    def __init__(self, tool: str, tool_input: Any, result: Any, thought: str = "use tool") -> None:
        self.thought = thought
        self.tool = tool
        self.tool_input = tool_input
        self.text = f"Action: {tool}"
        self.result = result


class FakeFinish:
    def __init__(self, output: str, thought: str = "done") -> None:
        self.thought = thought
        self.output = output
        self.text = output


class FakeTaskOutput:
    def __init__(self, raw: str, agent: str, name: str, description: str = "do the work") -> None:
        self.raw = raw
        self.agent = agent
        self.name = name
        self.description = description
        self.expected_output = "a short answer"


class FakeCrewAgent:
    def __init__(self, role: str) -> None:
        self.role = role
        self.step_callback = None

    def execute_task(self, task: Any = None) -> str:
        return f"{self.role}-notes"


class FakeCrew:
    def __init__(self) -> None:
        self.name = "research-crew"
        self.agents = [FakeCrewAgent("researcher"), FakeCrewAgent("writer")]
        self.step_callback = None
        self.task_callback = None

    def kickoff(self, inputs: dict[str, Any] | None = None) -> str:
        outs = []
        for agent, action in zip(
            self.agents,
            [
                FakeAction("search", '{"topic": "AI"}', "notes on AI"),
                FakeFinish("a polished brief"),
            ],
            strict=True,
        ):
            out = agent.execute_task()
            cb = agent.step_callback or self.step_callback
            if cb is not None:
                cb(action)
            if self.task_callback is not None:
                self.task_callback(
                    FakeTaskOutput(
                        raw=out if agent.role == "researcher" else "a polished brief",
                        agent=agent.role,
                        name=f"{agent.role}-task",
                    )
                )
            outs.append(out)
        return "a polished brief"


def test_wrap_crew_records_tools_tasks_and_multi_agent_attribution():
    crew = wrap_crew(FakeCrew(), name="research-crew")
    result = crew.kickoff(inputs={"topic": "AI"})
    assert result == "a polished brief"
    src = crashlab.load("last")
    assert src.name == "research-crew"
    assert src.output == "a polished brief"
    tool = next(s for s in src.steps if s.operation == "execute_tool")
    assert tool.name == "search"
    assert tool.request["args"] == {"topic": "AI"}
    assert tool.result == "notes on AI"
    assert tool.agent == "researcher"
    tasks = [s for s in src.steps if s.operation == "custom"]
    names = {s.name: s for s in tasks}
    assert "researcher-task" in names
    assert names["researcher-task"].agent == "researcher"
    assert names["researcher-task"].result == "researcher-notes"
    assert "writer-task" in names
    assert names["writer-task"].agent == "writer"
    finish = next(
        s
        for s in src.steps
        if s.name == "writer"
        or (s.operation == "custom" and s.result == "a polished brief" and s.agent == "writer")
    )
    assert finish.agent == "writer"
    assert src.agents == ["researcher", "writer"] or set(src.agents) >= {"researcher", "writer"}
    assert isinstance(src.findings, list)
    assert wrap_crew(crew) is crew


def test_crewai_callback_standalone_inside_record():
    handler = CrewAICallback()
    with crashlab.record("manual-crew") as run:
        with crashlab.agent_scope("researcher"):
            handler.step_callback(FakeAction("lookup", {"topic": "sqlite"}, "an in-process DB"))
        handler.task_callback(
            FakeTaskOutput("an in-process DB", agent="researcher", name="lookup-task")
        )
        err_action = FakeAction("boom", '{"x": 1}', None)
        err_action.error = RuntimeError("nope")
        handler.step_callback(err_action)
        run.output = "ok"
    src = crashlab.load(run.run_id)
    lookup = next(s for s in src.steps if s.name == "lookup")
    assert lookup.agent == "researcher"
    assert lookup.result == "an in-process DB"
    task = next(s for s in src.steps if s.name == "lookup-task")
    assert task.agent == "researcher"
    boom = next(s for s in src.steps if s.name == "boom")
    assert boom.error is not None and boom.error.is_tool_error
    assert isinstance(src.findings, list)


def test_crewai_callback_and_wrap_crew_ignore_events_outside_a_run():
    handler = CrewAICallback()
    handler.step_callback(FakeAction("search", "{}", "x"))
    handler.task_callback(FakeTaskOutput("x", "researcher", "t"))
    # wrap_crew without calling kickoff must not start a run
    crew = wrap_crew(FakeCrew(), name="idle")
    assert crew.name == "research-crew"


def test_wrap_crew_inside_existing_record_and_async_kickoff():
    class AsyncCrew(FakeCrew):
        async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> str:
            return self.kickoff(inputs)

    crew = wrap_crew(AsyncCrew(), name="async-crew")

    async def go() -> str:
        with crashlab.record("outer-crew") as run:
            result = await crew.kickoff_async(inputs={"topic": "AI"})
            run.output = result
            return result

    assert asyncio.run(go()) == "a polished brief"
    src = crashlab.load("last")
    assert src.name == "outer-crew"
    assert any(s.agent == "researcher" for s in src.steps)
    assert any(s.agent == "writer" for s in src.steps)


def test_wrap_crew_requires_crew():
    try:
        wrap_crew(None)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")
