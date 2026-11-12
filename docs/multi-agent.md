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

## Runtime checks per agent

In a labelled run, `step_repetition` compares replies and requests within one agent:
a writer that repeats the researcher's answer is a hand-over, not "the same reply 2
times in a row". Repeated tool calls are still counted across agents (two agents
placing the same order is a duplicate). Unlabelled runs behave as before.

## Agents used as tools

A sub-agent called from inside a tool (PydanticAI agent delegation, OpenAI Agents SDK
`agent.as_tool()`, your own `delegate()` tool) records its steps nested under that tool
step. A strict replay serves the tool from the cassette without running the sub-agent,
and copies the sub-agent's recorded steps into the replay (`source="cassette"`,
attribute `crashlab.replay.served_inside` = the tool step), so checks, usage and
exported tests see the same steps as the recording.

A served tool result keeps its types: pydantic models and dataclasses, also inside a
plain dict / list (Haystack's `AgentTool` returns `{"messages": [ChatMessage, ...]}`),
come back as those types, so the framework's code that reads `messages[-1].text`
works the same on replay. Each is rebuilt from its recorded fields with pydantic; a
type that cannot be imported or rebuilt (or a recording made before this) is served
as the recorded JSON. To record the delegation itself as a tool step, wrap the
framework tool's function, e.g. Haystack: `tool.function =
crashlab.tool(name="research")(tool.function)`.

Tools that take a framework context (PydanticAI `ctx: RunContext[...]`) work: the
context is not recorded as a tool argument. Leave out other per-run arguments with
`@crashlab.tool(ignore_args=("request_id",))`.

## Budgets per agent

`within_budget(run, agent="researcher", max_llm_calls=4)` checks one agent's steps
only; see [budgets-and-flakes.md](budgets-and-flakes.md).

## Handoffs

When one agent hands the conversation to another (OpenAI Agents SDK handoffs,
`transfer_to_refunds`; LlamaIndex AgentWorkflow `handoff`), the next agent's first
request answers the previous agent's handoff tool call. crashlab reads that from the
recording: `run.handoffs()` returns
`Handoff(from_agent="triage", to_agent="refunds", step=1, via_step=0,
tool="transfer_to_refunds", call_id=...)`, `crashlab show` prints a `handoffs` row
(`triage → refunds at step 1 (transfer_to_refunds, step 0)`) and marks the step
`⇢ from triage`, `crashlab show --json` adds a `handoffs` list when there are any, and
the diagnoser's view notes `(handoff from triage (transfer_to_refunds at step 0))`.
Agents called as tools (`as_tool`, smolagents managed agents, Haystack sub-agents,
PydanticAI delegation) start a fresh conversation and are not handoffs: they show as
the tool call and the sub-agent's labelled steps. Needs agent labels; works on runs
recorded before it existed.

## Fork patches for one agent

Prompt-level patches (system prompt, model, temperature, tools, tool choice, any patch
that applies to every LLM call from the fork step on) take `agent=` to edit only that
agent's calls:

```python
branch = crashlab.fork(run, at_step=0, patch=crashlab.SystemPromptPatch(
    op="append", agent="orchestrator",
    text="Copy dates exactly as the research tool gives them.",
))
```

or `{"kind": "system_prompt", "op": "append", "agent": "orchestrator", "text": "..."}`
in a `--patch` file. The agent of a forked call is the one it is recorded under
(`wrap(client, agent=...)`, `agent_scope`, `@crashlab.agent`). The fork is refused
before anything runs when the run has no LLM call of that agent from the fork step on
(a typo, or an agent that only ran earlier); the error lists the run's agents. Patch
descriptions end with `[agent orchestrator]`. Use the agent name, not an instance
label (`researcher#2`); step-bound patches (tool results, messages) already name their
step and do not take `agent=`. Two agents that share one wrapped client and no
`agent_scope` cannot be told apart, as when recording.

## The same agent running twice at once

When one agent runs twice in parallel (a manager calls `researcher.as_tool()` for two
topics at once), both runs carry the label `researcher`. crashlab tells them apart from
their conversations: each chat request that extends an earlier request of that agent
continues it (the one whose tool calls it answers). If two conversations of one agent
interleave, `crashlab show`, the diagnoser's view and `run.agent_labels` call them
`researcher#1` / `researcher#2` (numbered by first step); tool steps follow the chat
that requested them. Filter with `run.chat_steps(agent="researcher#2")`,
`run.tool_calls(agent="researcher#2")` or `crashlab show --agent "researcher#2"`; the
plain name still matches every instance. `run.agents` lists names only. An agent called
again later (not at the same time) keeps its plain name. This is inferred after the
fact, so it also works on runs recorded before it existed; it needs requests that carry
the conversation (concurrent agents that send only the latest message are not split).

## Diagnosing a multi-agent run

The diagnoser's view tags each step with its agent (`[agent: critic]`). The
`--heuristic` diagnoser uses the labels too: a failing check whose message quotes a
reply (`assert ..., out[:200]`) blames the chat step that first wrote that text and names
its agent, so a sub-agent that invented a fact is blamed rather than the orchestrator
that repeated it, and an orchestrator that changed a correct fact is blamed rather than
the lookup that returned it. Quotes name the field they came from (a critic's reply
"APPROVE" is `response.text`, not its system prompt that also says "reply APPROVE").

## Runs in other processes

A worker started as a subprocess, or a service the orchestrator calls, records its own
run. Link it to the run that started it:

```python
import os, subprocess
import crashlab

with crashlab.record("orchestrator", input={"task": task}) as run:
    # CRASHLAB_PARENT_RUN, CRASHLAB_TRACE_ID and CRASHLAB_DIR for the child
    subprocess.run(["python", "worker.py"], env={**os.environ, **crashlab.child_env()})
```

Any run the worker records (`crashlab.record`, `@crashlab.agent`) reads
`$CRASHLAB_PARENT_RUN` / `$CRASHLAB_TRACE_ID` and links itself. Across HTTP or a queue,
send the two ids yourself and pass them explicitly:

```python
with crashlab.record("worker", parent=msg["parent_run"], trace=msg["trace"]) as run: ...
```

`record(parent=run)` also takes a `Run`. A linked run has `run.parent_run` and
`run.trace_id` (the root run's id; an unlinked run is its own trace).
`crashlab show PARENT` lists the child runs; `crashlab show CHILD` shows its parent and
trace; `crashlab ls --trace RUN` and `crashlab search --trace RUN` list every run of the
trace of any run in it. `Store().child_runs(run_id)` and
`Store().list_runs(trace=...)` do the same in Python.

- The link is stored in `run.env["link"]`, so the run format is unchanged (0.0.3 reads
  linked runs) and runs recorded before have no link.
- `child_env()` sets `CRASHLAB_DIR` to the parent's store so `show` finds the children;
  drop it if the worker should record elsewhere (the link is kept, `show` just cannot
  list children from another store).
- Every run recorded by a process started with these variables links to that parent
  (a long-lived worker handling many tasks links them all). Replays and forks are not
  linked; they point to their source run (`replay of` / `forked from`).
- Bad values in the variables are ignored with a warning; the worker is still recorded.

## Not covered

- Runs in other processes are separate runs, linked rather than merged (see "Runs in
  other processes"): no step of the worker shows in the parent's step table, and a
  replay or fork of the parent does not re-run or replay the workers.
- Handoffs are shown, not recorded as extra steps (see "Handoffs"), so step numbers and
  replay are unchanged. A handoff that passes no conversation (the next agent starts
  from scratch) looks like any other agent change.
