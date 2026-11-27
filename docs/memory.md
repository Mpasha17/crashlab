# Memory and state debugging

Long-running and multi-turn agents fail in ways a single-call test never sees: they
**forget** what the user said three turns ago, **contradict** their own earlier answer,
quietly **lose context** when the history is trimmed, **reuse a stale tool result**, or
carry a prompt that grows without bound. crashlab catches these from the recording alone,
with turn-level evidence, and lets you record your agent's own state to debug it.

```bash
crashlab memory last           # what did the agent forget, contradict or lose?
crashlab state last --diff     # how did its state change turn by turn?
```

## The five `memory.*` checks

They are ordinary runtime checks: they run while recording, on `crashlab check`, and on
`crashlab memory`, and their findings land on the run like any other. All are heuristic and
English-centric; every finding quotes the steps it is based on so you can judge it.

| Finding | Fires when |
| --- | --- |
| `memory.context_truncation` | messages that an earlier LLM request of the conversation had (beyond the system prompt) are silently missing from a later one: a sliding window, a bad trim, a history not passed on. Lists the lost turns and the facts in them. A turn of `crashlab.simulate` that carries nothing of the previous turn is reported as a reset. A request that contains a summary of the history ("summary of the conversation so far") counts as deliberate compaction and is skipped. |
| `memory.forgotten_fact` | a fact the user gave (name, email, phone, ids such as `order A1234`, dates, a budget, a party size) is **asked for again**, or **contradicted** by a later reply with a value that appears nowhere in the conversation or tool results. |
| `memory.self_contradiction` | the agent states different values for the same thing in different replies (`The total is $120` then `$150`; a date; eligible / not eligible; "I can cancel" / "I cannot cancel"). Skipped when a tool result, a side-effect tool call or a new user message in between explains the change, and for hedged statements ("if", "would", "about"). |
| `memory.stale_tool_result` | a reply still uses a value from a tool result that a newer call of the same tool with the same arguments replaced (the order was `pending`, was shipped, and the agent still says `pending`). Timestamps and similar volatile fields are ignored. |
| `memory.unbounded_growth` | the prompt (or a state snapshot) grows faster every turn: per-turn growth in the second half is well above the first half. Linear growth and compacted histories do not fire. |

A normal conversation that passes its history on every turn produces none of them.

## Settings

Under `[tool.crashlab.checks.memory]` in `pyproject.toml`, or `crashlab.record(..., checks={"memory": ...})`:

```toml
[tool.crashlab.checks.memory]
enabled = true

[tool.crashlab.checks.memory.context_truncation]
min_dropped = 1        # dropped messages needed to report
reset = true           # also report a turn that carries nothing of the previous one

[tool.crashlab.checks.memory.forgotten_fact]
ask_again = true
contradicted = true
ignore_kinds = ["date"]   # email, phone, id, name, date, money, count

[tool.crashlab.checks.memory.stale_tool_result]
ignore_tools = ["poll_job"]

[tool.crashlab.checks.memory.unbounded_growth]
min_steps = 6
min_growth_tokens = 1000
acceleration = 1.5

[tool.crashlab.checks.memory.self_contradiction]
enabled = false
```

`memory = false` turns all five off; a section set to `false` turns that one off.

## `crashlab memory`

```bash
crashlab memory last
crashlab memory 01J... --json
crashlab memory last --judge --model gpt-4o-mini
```

Prints each finding with its steps and verbatim evidence (the user's message that held the
fact, the reply that asked again). Exit 0 when clean, 1 when there are findings, 2 when the
run does not exist. `--json` prints the findings (`kind`, `severity`, `steps`, `message`,
`evidence`) for CI.

`--judge` asks a model to confirm or dismiss each finding (a custom judge from
[`crashlab.judges`](judges.md); same `CRASHLAB_JUDGE_MODEL` / `_BASE_URL` / `_API_KEY`
settings, and the same caveat that run data is sent to the model). Findings it scores as
false alarms are marked `dismissed` and do not count towards the exit code.

From Python:

```python
import crashlab
from crashlab import memory

run = crashlab.load("last")
for f in memory.memory_findings(run):
    print(f.kind, f.steps, f.message)

confirmed = memory.confirm_with_judge(run, memory.memory_findings(run), model="gpt-4o-mini")
```

Because they are checks, the same findings appear in `crashlab show`, `crashlab check`,
the diagnoser's evidence (forgotten and contradicted facts map to `TASK_SPEC_VIOLATION`, a
stale tool result to `TOOL_ERROR_MISHANDLED`) and `crashlab.simulate` runs:

```python
import crashlab

scenario = crashlab.Scenario(
    name="remembers-the-order",
    turns=[
        crashlab.ScenarioTurn(user="My name is Priya and my order is A1234."),
        crashlab.ScenarioTurn(user="Where is my order?"),
    ],
)
result = crashlab.simulate(my_agent, scenario)   # my_agent(user_msg, history)
print(crashlab.load("last").findings_of("memory.context_truncation"))
```

## State snapshots

Record whatever your agent calls memory (a scratchpad, a plan, a cart, a vector-store
summary) with `crashlab.snapshot(name, state)` at the points that matter:

```python
import crashlab


@crashlab.agent(name="shopper")
def run_agent(user_msg: str) -> str:
    notes = {"user": None, "cart": []}
    crashlab.snapshot("memory", notes)
    notes["cart"].append("book")
    crashlab.snapshot("memory", notes)
    return "done"
```

The state must be JSON-serialisable. It is redacted like every other step, long strings,
lists and dicts are cut, and a snapshot over 20,000 characters keeps only a preview. A
snapshot is a `custom` step (never counted as an LLM call), replays serve the recorded
state so they stay offline, and outside a recorded run `crashlab.snapshot` does nothing and
returns `False`.

```bash
crashlab state last                      # timeline: step, turn, name, size
crashlab state last --diff               # key-level changes between snapshots
crashlab state last --diff --name memory --json
```

`--diff` compares each snapshot with the previous one of the same name and lists keys
`added`, `removed` and `changed` (`cart[1]`, `user.email`). Growing snapshots feed
`memory.unbounded_growth`. From Python: `memory.snapshots(run)` and
`memory.state_diff(old, new)`.
