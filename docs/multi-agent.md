# Multi-agent runs

A multi-agent app (a triage agent handing off to a specialist, a supervisor with
workers, sub-agents fanned out with `asyncio.gather`) is recorded as **one run** when
it runs inside one `@crashlab.agent` / `crashlab.record(...)`. Two things make such a
run debuggable:

1. **Which agent made each step.** Steps carry the OpenTelemetry GenAI attribute
   `gen_ai.agent.name`, read back as `step.agent`.
2. **Which chat step asked for each tool call**, even when sub-agents interleave.

## Labelling steps with the agent

One client per agent (OpenAI Agents SDK `OpenAIChatCompletionsModel`, and other
frameworks that take a client per agent, or your own loop):

```python
raw = AsyncOpenAI(...)
triage_model = OpenAIChatCompletionsModel(model=m, openai_client=crashlab.wrap(raw, agent="triage"))
refunds_model = OpenAIChatCompletionsModel(model=m, openai_client=crashlab.wrap(raw, agent="refunds"))
```

One client shared by several agents (LangGraph nodes, hand-written supervisors):

```python
def researcher_node(state):
    with crashlab.agent_scope("researcher"):
        return {"notes": llm.invoke(state["messages"])}
```

Rules:

- `wrap(client, agent=...)` labels every call that client makes; it wins over a scope.
  Re-wrapping an already wrapped client with another name wraps the original client
  again; `with_options()` / `copy()` keep the name.
- `agent_scope(name)` labels every step recorded inside the block (LLM calls, tools,
  custom spans). It nests (innermost wins) and works around `await` and in
  `asyncio.gather` tasks started inside it.
- A tool step with no scope takes the agent of the chat step that requested it.
- A nested `@crashlab.agent` (recorded as an `invoke_agent` step) labels the steps
  made inside it with its own name.
- The litellm module is patched process-wide, so `wrap(litellm, agent=...)` is refused;
  use `agent_scope`.

Using the labels:

```python
@crashlab.check("support")
def only_refunds_agent_refunds(run):
    assert not run.tool_calls("issue_refund", agent="triage"), "triage issued a refund"

run.agents                      # ["triage", "refunds"], in order of first appearance
run.chat_steps(agent="refunds")
```

`crashlab show` prints an `agents  triage → refunds` row and `@agent` next to each
labelled step; `crashlab show last --agent refunds` lists only that agent's steps
(combines with `--tools`, `--llm`, `--errors`). The diagnoser's view of the run names
the agent of each labelled step, so a diagnosis can say which agent went wrong.

Labels are metadata only: request hashes, replay, fork and export are unchanged, and
runs recorded by 0.0.3 (no labels) load and replay as before (`step.agent` is `None`).

## Concurrent sub-agents

When sub-agents run concurrently in one run, their chat and tool steps interleave.
Each `@crashlab.tool` step is linked to the chat step whose response requested it
through `gen_ai.tool.call.id`, and `run.caller_chat(idx)`, `run.describe_step(idx)`
and the heuristic diagnoser use that link (falling back to the nearest earlier chat
step when a recording has no call id). Before this, a slower sub-agent's tool step lost
its call id and was attributed to the other sub-agent's chat step.

## Not covered

- Separate processes or services (an orchestrator calling a worker over HTTP) record
  separate runs; crashlab does not link runs across processes.
- Framework-internal handoff events are not recorded as their own steps; a handoff is
  visible as the handoff tool call (e.g. `transfer_to_refunds`) followed by steps of
  the next agent.
- Fork patches still target steps by index, not by agent.
