# Framework and protocol integrations

crashlab records whatever your agent already does. For a plain OpenAI or Anthropic
client that is `crashlab.wrap(client)`. The helpers here cover two other cases, plus
a command that isolates what changed between a passing run and a failing one.

- [MCP](#mcp) — record `call_tool` on an MCP client session
- [LangChain / LangGraph](#langchain--langgraph) — a callback handler that labels nodes and fills gaps
- [OpenAI Agents SDK](#openai-agents-sdk) — a duck-typed `TracingProcessor` for agent, generation, tool and handoff spans
- [PydanticAI](#pydanticai) — wrap `Agent.run` / `run_sync` / `run_stream` inside `crashlab.record`
- [CrewAI](#crewai) — `step_callback` / `task_callback` hooks and a `wrap_crew()` helper
- [`crashlab bisect`](#crashlab-bisect) — compare two runs and isolate the minimal causal change

None of these import `mcp`, `langchain`, `langchain_core`, `agents`, `pydantic_ai`, or
`crewai` until you use them (the OpenAI Agents, PydanticAI and CrewAI adapters are
purely duck-typed and never import those packages), so a missing optional package
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
