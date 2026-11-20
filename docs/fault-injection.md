# Fault injection and chaos testing

Inject realistic tool and LLM failures into a live recorded run, then see whether
the agent recovered. Every injection is recorded on the step as
`crashlab.fault`, so `crashlab show`, `crashlab diff` and `crashlab diagnose`
can see it. Strict `crashlab.replay` never re-injects: it serves the cassette,
including the recorded fault.

## `crashlab.faults(...)`

A context manager. Rules are stackable and nest via contextvars (thread- and
asyncio-safe). When no policy is active the hook is a single context-var read.

```python
import crashlab
from crashlab.faults import tool_error, llm_error

with crashlab.faults(tool_error(tool="get_order", status=502, on_calls=[1])):
    run_agent("Refund order A17")

run = crashlab.load("last")
step = next(s for s in run.steps if s.operation == "execute_tool")
print(step.attributes["crashlab.fault"])
```

Constructors also hang off the context manager, so this is the same rule:

```python
import crashlab

with crashlab.faults(crashlab.faults.tool_error(tool="get_order", on_calls=[1])):
    run_agent("Refund order A17")
```

### When a rule fires

Call numbers are 1-based and count only matching calls (the same tool, or any
LLM call for an LLM fault; and, if you pass `agent=`, that agent, including
`researcher#1` when the rule says `agent="researcher"`):

| argument | meaning |
|---|---|
| `on_calls=[1, 3]` | only those call numbers |
| `after_calls=2` | starting at call 3 |
| `every=2` | every second matching call (2, 4, 6, …) |
| `max_triggers=1` | stop after this many injections |
| `probability=0.5`, `seed=0` | after the filters above, fire with this probability (pass `seed` for a deterministic stream) |
| `agent="researcher"` | only that agent's calls |

The first matching rule of the innermost `faults()` block wins.

### Tool faults

```python
from crashlab.faults import tool_delay, tool_error, tool_result

tool_error(tool="get_order", exc=TimeoutError("upstream hung"))
tool_error(tool="get_order", status=502)          # HTTP-style error, status_code=502
tool_result(tool="get_order", mode="html_502")    # nginx 502 HTML page
tool_result(tool="get_order", mode="empty")       # {} / [] / ""
tool_result(tool="get_order", mode="truncated")   # cut-off JSON (runs the tool first)
tool_result(tool="get_order", mode="schema_drift")  # renamed / dropped keys
tool_result(tool="get_order", result={"ok": False}, mode="replace")
tool_delay(tool="get_order", seconds=0.05)        # live record / fork only
```

`tool="*"` (the default) matches every tool. `html_502`, `empty` and `replace`
do not call the tool; `truncated` and `schema_drift` call it, then corrupt the
result. `tool_delay` sleeps, then calls the tool as usual. A strict replay of a
delayed step does not sleep again.

### LLM faults

```python
from crashlab.faults import llm_bad_tool_call, llm_empty, llm_error, llm_truncate

llm_error(status=429, message="rate limited")  # raised instead of calling the model
llm_empty()                                   # no text, no tool calls
llm_truncate(max_chars=20)                    # finish_reason="length"
llm_bad_tool_call(name="not_a_real_tool", arguments="{not json")
```

Mutation faults (`llm_empty`, `llm_truncate`, `llm_bad_tool_call`) apply to
non-streaming completions: the live call still happens, then the response
object is edited before the agent sees it. Streaming calls are not mutated
(`llm_error` still raises before the request).

### Replay vs fork

```python
import crashlab
from crashlab.faults import tool_result

# A recording that already contains an injected fault replays offline, unchanged:
replayed = crashlab.replay(run)

# A fork with an active faults() context injects on live steps only
# (steps before at_step are served from the cassette and are not re-injected):
with crashlab.faults(tool_result(tool="get_order", mode="html_502", on_calls=[1])):
    branch = crashlab.fork(run, at_step=0, tools="live")
```

Use `tools="live"` on the fork if you want a tool fault to fire: the default
`tools="cassette_if_match"` would otherwise serve an identical recorded tool
call from the cassette.

## `crashlab stress`

Takes a passing (or any) run and tries the built-in chaos scenarios against the
tools and models that run actually used:

| scenario | what it injects |
|---|---|
| `tool_502` | HTML 502 page from that tool |
| `tool_timeout` | `TimeoutError` from that tool |
| `tool_empty` | empty tool result |
| `tool_schema_drift` | renamed / dropped keys |
| `llm_rate_limit` | HTTP 429 on the first LLM call |
| `llm_empty_reply` | empty assistant message |
| `llm_bad_tool_args` | cut-off JSON tool-call arguments |

```python
from crashlab.stress import stress_run

report = stress_run(run, scenarios=["tool_502", "llm_rate_limit"], at_step=0)
print(report.survived)
for row in report.results:
    print(row.scenario, row.target, row.outcome, row.survived)
```

`stress_run` (and `crashlab stress`) fork the recorded entry point with the
fault active. When the entry point cannot be imported or no live client is
available, the row is `needs_live_agent` or `skipped`, not a crash. A
scenario *survived* when the forked run ended `ok` and every `@crashlab.check`
still passed.

```bash
crashlab stress last
crashlab stress last --scenario tool_502 --scenario llm_rate_limit
crashlab stress last --at-step 0 --agent researcher --json
crashlab stress last --fail-on-crash
```

`--fail-on-crash` exits 1 when a scenario crashed or failed its checks.
Skipped / `needs_live_agent` rows do not fail the command.

## Limits

- Faults apply to live recorded calls and to live steps of a fork. Strict
  replay always serves the cassette.
- `tool_delay` is skipped on replay (the original duration is already on the
  step).
- Streaming LLM responses are not mutated; use a non-streaming call, or
  `llm_error`.
- Side-effect tools (`side_effects=True`) are not run live in a fork unless
  you pass `allow_live_side_effects=True` or register a `fake_tool`. A tool
  scenario that would need that is reported rather than executed against the
  real world.
