# Framework and protocol integrations

crashlab records whatever your agent already does. For a plain OpenAI or Anthropic
client that is `crashlab.wrap(client)`. The helpers here cover two other cases, plus
a command that isolates what changed between a passing run and a failing one.

- [MCP](#mcp) — record `call_tool` on an MCP client session
- [LangChain / LangGraph](#langchain--langgraph) — a callback handler that labels nodes and fills gaps
- [OpenAI Agents SDK](#openai-agents-sdk) — a duck-typed `TracingProcessor` for agent, generation, tool and handoff spans
- [PydanticAI](#pydanticai) — wrap `Agent.run` / `run_sync` / `run_stream` inside `crashlab.record`
- [CrewAI](#crewai) — `step_callback` / `task_callback` hooks and a `wrap_crew()` helper
- [LlamaIndex](#llamaindex) — an instrumentation event + span handler (LLM, retrieval, tools, agents) and `wrap_query_engine()`
- [smolagents](#smolagents) — `wrap_smolagent()` and a step callback for memory steps, tools and `managed_agents`
- [Haystack](#haystack) — `wrap_pipeline()` records every component; a duck-typed `HaystackTracer`
- [`crashlab bisect`](#crashlab-bisect) — compare two runs and isolate the minimal causal change

None of these import `mcp`, `langchain`, `langchain_core`, `agents`, `pydantic_ai`,
`crewai`, `llama_index`, `smolagents` or `haystack` until you use them (the OpenAI
Agents, PydanticAI, CrewAI, LlamaIndex, smolagents and Haystack adapters are purely
duck-typed and never import those packages), so a missing optional package
never breaks `import crashlab`.

## MCP

Wrap an MCP `ClientSession` (or any duck-typed client with `call_tool(name, arguments=...)`,
sync or async). The `mcp` package is not required.

```python
import crashlab
from crashlab.integrations.mcp import wrap_mcp_session

session = wrap_mcp_session(session, server_name="github", side_effects=False)

@crashlab.agent(name="mcp-bot")
async def run(query: str):
    result = await session.call_tool("search", arguments={"q": query})
    return result.content[0].text
```

Each `call_tool` is recorded as an `execute_tool` step. With `server_name="github"` the
tool name is `mcp:github:search`; without a server name it is just `search`. Arguments,
the result, duration and errors go through the usual redaction and runtime checks
(`suspicious_tool_result`, `tool_error_streak`, …).

`side_effects=True` marks every call as a side-effect tool (writes, sends, payments). A
replay or fork will not hit the real server for a new call unless you pass
`allow_live_side_effects=True`.

MCP `CallToolResult`-like values (`content`, `isError`) are stored as JSON and rebuilt
on replay, so `result.content[0].text` and `result.isError` still work offline.

```python
import crashlab

run = crashlab.load("last")
crashlab.replay(run)  # serves the recorded MCP result; the server is not contacted
```

Wrapping is idempotent. `list_tools`, `initialize` and context-manager methods are
forwarded and not recorded. Resources, prompts and notifications are not wrapped yet.

## LangChain / LangGraph

The most reliable path is still to wrap the underlying provider client and pass it into
LangChain, as in [examples/integrations/langchain.md](../examples/integrations/langchain.md)
and [langgraph.md](../examples/integrations/langgraph.md). Offline replay needs that wrap.

`CrashlabCallbackHandler` is the extra hook for graphs and executors: it labels steps
with the LangGraph node (or chain name) and records LLM / tool calls that `wrap()` did
not already see.

```python
import crashlab
from crashlab.integrations.langchain import CrashlabCallbackHandler

handler = CrashlabCallbackHandler()

@crashlab.agent(name="support")
def run(msg: str) -> str:
    oc = crashlab.wrap(OpenAI(base_url=BASE_URL, api_key=KEY))
    llm = ChatOpenAI(model=MODEL, client=oc.chat.completions, root_client=oc)
    graph = create_react_agent(llm, tools)
    out = graph.invoke(
        {"messages": [("user", msg)]},
        config={"callbacks": [handler]},
    )
    return out["messages"][-1].content
```

When a chain starts with `metadata["langgraph_node"]` (LangGraph does this) or a
non-generic chain name, the handler opens `crashlab.agent_scope(...)`, so every step
recorded inside that node — including by `crashlab.wrap()` — gets `step.agent`. Generic
LangChain internals (`RunnableSequence`, `LLMChain`, …) are not used as names.

If the client is already wrapped, `on_llm_end` / `on_tool_end` see the step `wrap()`
just wrote and do not record a second copy.

**Limits.** A callback cannot cancel a provider call. For offline `crashlab.replay`,
wrap the OpenAI / Anthropic client as well; the handler alone is enough to inspect a
live `crashlab.record()` run.

## OpenAI Agents SDK

The most reliable path for offline replay is still to wrap the underlying OpenAI
client and pass it into the SDK (see [examples/integrations/openai-agents.md](../examples/integrations/openai-agents.md)).

`OpenAIAgentsTracingProcessor` is the extra hook: it implements the SDK's
`TracingProcessor` interface (`on_trace_start`, `on_trace_end`, `on_span_start`,
`on_span_end`, `shutdown`, `force_flush`) without importing `agents`. Register it
with `add_trace_processor`. Agent spans, generation spans (model, usage, input /
output), function/tool spans and handoff spans (`from_agent -> to_agent`) become
crashlab steps with `step.agent` attribution.

```python
import crashlab
from agents import Runner, add_trace_processor
from crashlab.integrations.frameworks import crashlab_tracing_processor

add_trace_processor(crashlab_tracing_processor())

@crashlab.agent(name="support")
def run(msg: str) -> str:
    result = Runner.run_sync(agent, msg)
    return str(result.final_output)
```

If no `crashlab.record()` / `@crashlab.agent` run is already active, the processor
starts one on `on_trace_start` (named after the trace, or `name=` on
`crashlab_tracing_processor(name="support")`) and closes it on `on_trace_end`.
Already-recorded LLM or tool calls from `crashlab.wrap()` / `@crashlab.tool` are
not duplicated.

A callback cannot cancel a provider call. For offline `crashlab.replay`, wrap the
OpenAI client as well; the processor alone is enough to inspect a live
`crashlab.record()` run.

## PydanticAI

`wrap_pydantic_ai_agent` wraps a PydanticAI-like `Agent` (`run`, `run_sync`, and
`run_stream` when present) so each call runs inside `crashlab.record(name=...)`
when no run is already active. Function tools on `agent._function_tools` (or
`tools`) and a model `request` method are wrapped in place so usage, tool calls
and output are captured even without wrapping the provider client.

```python
import crashlab
from crashlab.integrations.frameworks import wrap_pydantic_ai_agent
from pydantic_ai import Agent

agent = wrap_pydantic_ai_agent(
    Agent(model, system_prompt=PROMPT, tools=[get_price, place_order]),
    name="shop",
)
result = agent.run_sync("price of tea")
print(result.output)
```

Wrapping is idempotent. Passing the wrapped functions through `tools=[...]` still
works; already-decorated `@crashlab.tool` callables are left alone. For offline
replay, also wrap the OpenAI / Anthropic client you pass into the model (see
[examples/integrations/pydantic-ai.md](../examples/integrations/pydantic-ai.md)).

`PydanticAIAdapter(agent, name="shop").wrap()` is the same helper in class form.

## CrewAI

`CrewAICallback` provides the `step_callback` and `task_callback` hooks CrewAI
invokes after each agent action and each task. `wrap_crew(crew, name=...)`
installs those hooks (attributed to each agent's `role`), wraps
`crew.kickoff(...)` (and `kickoff_async` when present) inside
`crashlab.record(name=...)`, and records tool calls and task outputs.

```python
import crashlab
from crashlab.integrations.frameworks import CrewAICallback, wrap_crew
from crewai import Crew

crew = wrap_crew(
    Crew(agents=[researcher, writer], tasks=[research, write]),
    name="research-crew",
)
result = crew.kickoff(inputs={"topic": "AI"})
```

You can also attach the hooks yourself:

```python
import crashlab
from crashlab.integrations.frameworks import CrewAICallback

handler = CrewAICallback()
crew = Crew(
    agents=[researcher],
    tasks=[task],
    step_callback=handler.step_callback,
    task_callback=handler.task_callback,
)

with crashlab.record("research-crew") as run:
    run.output = str(crew.kickoff())
```

Wrapping is idempotent. For offline replay, wrap the LLM client as well (CrewAI
calls `with_raw_response.create`; see
[examples/integrations/crewai.md](../examples/integrations/crewai.md)).

## LlamaIndex

`crashlab.integrations.llama_index_handler()` returns a duck-typed handler for
LlamaIndex's instrumentation dispatcher. It implements both interfaces (`handle(event)`
for events, `span_enter` / `span_exit` / `span_drop` for spans), so one object records:

- **LLM calls** (`LLMChatStartEvent` / `LLMChatEndEvent`, and the completion events) as
  `chat` steps with the model, the messages, token usage and any tool calls;
- **retrieval** (`RetrievalStartEvent` / `RetrievalEndEvent`) as a tool step named
  `search_docs` whose result is `[{id, text, score, metadata}]`, exactly what
  `crashlab.rag.retrieved` reads (`retrieval_tool="kb"` renames it);
- **tools**: a `FunctionTool` / `QueryEngineTool` `call` / `acall` span becomes a tool
  step with its arguments and `ToolOutput` (an `is_error` output is a failing tool step);
  an `AgentToolCallEvent` that never sees a tool span is recorded with
  `crashlab.llama_index.result_unavailable`;
- **agents**: spans of an object named `*Agent` with a `name` attribute attribute every
  step inside them to that agent (`gen_ai.agent.name`), so an `AgentWorkflow` handoff
  shows up in `run.agents`;
- **errors**: `SpanDropEvent`, `ExceptionEvent` and dropped spans become failing steps.

```python
import crashlab
from crashlab import rag
from crashlab.integrations import llama_index_handler, wrap_query_engine

handler = llama_index_handler()
handler.attach()  # root dispatcher: add_event_handler + add_span_handler

engine = wrap_query_engine(index.as_query_engine(), name="kb-bot")
engine.query("What is the refund window?")  # one recorded run

run = crashlab.load("last")
assert rag.retrieval_non_empty(run) is None
```

`wrap_query_engine(engine, name=...)` wraps `query` / `aquery` / `chat` / `achat` in
`crashlab.record` (unless a run is already active) and sets the run output to the
response text. With no handler attached it still records the response's `source_nodes`
as the retrieval step, so `crashlab.rag` oracles work; LLM calls then need the handler,
`crashlab.wrap(client)` or `crashlab.instrument()`. Without an active run the handler
does nothing. A chat or tool step that a wrapped client or `@crashlab.tool` recorded
inside the same LlamaIndex call is not recorded a second time.

Limits: checked locally against `llama-index-core` 0.14.26 (query engine, tool spans,
LLM events with a mock LLM), not against every LlamaIndex release or a live provider.
Streaming chat events (`LLMChatInProgressEvent`) are ignored, and a completion event
nested inside a chat event (a completion-only LLM used as a chat model) is recorded once.
A document's `id` is the node id; the source document id is in `metadata.ref_doc_id`.
Replay needs a wrapped client as for any other framework.

## smolagents

`crashlab.integrations.wrap_smolagent(agent, name=...)` wraps `agent.run` in
`crashlab.record` (unless a run is already active), both plain and `stream=True`
(the generator is passed through and each step recorded as it is yielded):

```python
import crashlab
from crashlab.integrations import crashlab_step_callback, wrap_smolagent

agent = wrap_smolagent(
    ToolCallingAgent(tools=[PriceTool()], model=model, name="manager", managed_agents=[catalog]),
    name="smol-prices",
)
answer = agent.run("What do 12 widgets cost?")
```

Each memory step is recorded: an `ActionStep` / `PlanningStep` model output is a `chat`
step (model id, messages, token usage, tool calls), every tool call is an `execute_tool`
step with its `observations`, and a tool error is a failing tool step (an error with no
tool call, such as a parse error, is a failing `custom` step named `smolagents.error`).
`managed_agents` are wrapped recursively and their steps carry the managed agent's own
name, so `run.agents` is `["manager", "catalog"]`. With several tool calls in one step
the combined observations are attached to each call (smolagents does not split them).

`crashlab_step_callback()` is the same recorder as an `agent.step_callbacks` entry
(works with the list and the registry styles), for agents you construct yourself:

```python
agent = ToolCallingAgent(
    tools=[PriceTool()], model=model, step_callbacks=[crashlab_step_callback()]
)
```

`wrap_smolagent` installs it once, and a step is never recorded twice; a step that
`crashlab.wrap(model.client)` or `@crashlab.tool` already recorded live is not
duplicated. Without a callback hook, steps are read from `agent.memory` after `run`.
Checked locally against `smolagents` 1.26.0 (a `ToolCallingAgent` and a `CodeAgent` with
fake models, managed agents, streaming and a failing tool). A `CodeAgent` step's code is
recorded as a `python_interpreter` tool call.

## Haystack

`crashlab.integrations.wrap_pipeline(pipeline, name=...)` wraps a `Pipeline` /
`AsyncPipeline`: `run` (and `run_async`) execute inside `crashlab.record`, and every
component's `run` / `run_async` becomes a step:

| component | recorded as |
|---|---|
| `*Generator`, `Agent` | `chat` step; model, finish reason and token usage from `meta`; tool calls in the reply |
| `*Retriever` | tool step `search_docs` with `[{id, text, score, metadata}]` (visible to `crashlab.rag`) |
| `ToolInvoker` | one tool step per `ToolCallResult`; `error=True` results are failing tool steps |
| other components that return `documents` (rankers, joiners) | tool step named after the component |
| everything else | `custom` step with bounded inputs / outputs (embedding vectors are summarised) |

```python
import crashlab
from crashlab import rag
from crashlab.integrations import wrap_pipeline

pipe = wrap_pipeline(rag_pipeline, name="hay-rag")
pipe.run({"retriever": {"query": "What is the refund window?"}})

run = crashlab.load("last")
assert rag.retrieval_non_empty(run) is None
```

`wrap_pipeline(pipe, retrievers=["my_lookup"], retrieval_tool="kb")` treats other
components as retrievers and renames the tool. Instead of wrapping, enable Haystack's
own tracing with the duck-typed `HaystackTracer`
(`haystack.tracing.enable_tracing(HaystackTracer())`): each `haystack.component.run`
span becomes the same step, and a `haystack.pipeline.run` span starts a run when none is
active. Content tracing is not needed (the tracer reads the content tags itself). A
component already recorded by `wrap_pipeline` is not recorded again, and neither adapter
duplicates a chat or tool step recorded live by a wrapped client or `@crashlab.tool`.

Limits: checked locally against `haystack-ai` 3.5.0 (a real `Pipeline` with
`InMemoryBM25Retriever`, `ChatPromptBuilder` and a fake generator); component inputs are
recorded as Haystack hands them over, with embeddings and long strings truncated, not
stored in full.

## `crashlab bisect`


Given a passing run and a failing run, `crashlab bisect` reports what changed: git /
code version, model and sampling params, system prompts, declared tool schemas and
descriptions, the agent roster, recorded inputs, and the step-by-step trajectory
(the same comparison as `crashlab.diff`). Factors are ranked by how close they sit to
the first diverging step.

```bash
crashlab bisect GOOD BAD
crashlab bisect GOOD BAD --agent researcher --model gpt-4o
crashlab bisect GOOD BAD --isolate
crashlab bisect GOOD BAD --isolate --json
```

In Python:

```python
from crashlab.bisect import bisect_runs

report = bisect_runs(good, bad, isolate=True)
print(report.summary())
if report.causal:
    print(report.causal.patch_description)
```

`--isolate` / `isolate=True` builds a candidate patch from each reversible difference
(revert the failing run's system prompt, model, params, or a tool description / schema
to the passing run's value) and forks the failing run with that one patch. The first
single revert that restores `outcome == "pass"` is the **minimal causal change**.

Isolation forks serve matching tool calls from the cassette and only go live (or hit a
mock) for the patched request. They need a replayable entry point (`@crashlab.agent` or
a `crashlab.record(...)` block passed to `fork`). A config change that cannot be
expressed as a single patch, or a trajectory-only difference (same request, different
model reply), is reported as a factor but is not a candidate. `--agent` / `--model`
restrict both the comparison and the patches.
