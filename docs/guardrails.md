# Live guardrails: `crashlab.guard`

crashlab's detectors tell you *after* a run that the agent looped, overspent or called
the wrong tool. `crashlab.guard(...)` enforces the same rules **while the agent runs**:
every LLM call and every `@crashlab.tool` call inside the block is checked *before* it
is made, and the guard can block it, warn, retry it, switch model, repair it, or redact
what comes back.

```python
import crashlab

with crashlab.guard(
    max_cost_usd=0.50,            # circuit breakers
    max_steps=40,
    max_seconds=120,
    loop_breaker=3,               # same tool + args 3 times, or 3 identical LLM replies
    tool_denylist=["delete_*", "drop_table"],
    validate_tool_args="repair",  # bad args go back to the model as a tool error
    retry_policy=True,            # 429 / 5xx / timeouts: exponential backoff + jitter
    fallback_model="gpt-4o-mini",
    block_secrets_in_output=True,
):
    run_agent("Refund order A17")
```

It also works as a decorator (each call gets a fresh scope with its own counters), on
`async def` functions, and as `async with`:

```python
import crashlab


@crashlab.guard(max_llm_calls=20, on_violation="stop")
@crashlab.agent(name="refund-bot")
def run_agent(user_msg: str) -> str: ...


async def main():
    async with crashlab.guard(max_tool_calls=10) as scope:
        await run_async_agent("hi")
    print(scope.tool_calls, scope.interventions)
```

Guards run inside a recorded run (`@crashlab.agent`, `crashlab.record`,
`crashlab.instrument()` ...): that is where crashlab sees the calls. They are
stackable: nested `with crashlab.guard(...)` blocks each count the calls made inside
their own scope, and every active guard is checked. They use `contextvars`, so they are
thread- and asyncio-safe (concurrent tasks each get their own scope). When no guard is
active the cost is one context-variable read per call.

## Policies

| Policy | What it does |
| --- | --- |
| `max_cost_usd` | Blocks the next call once the calls made under the guard cost this much (steps whose cost is unknown count as 0; `detail.cost_unknown` says so). |
| `max_tokens` | Blocks the next call once this many tokens were used (chat + embeddings). |
| `max_steps` | Blocks the next LLM / tool call after this many. |
| `max_llm_calls` / `max_tool_calls` | Separate limits for LLM calls and tool calls. |
| `max_seconds` | Blocks the next call once the guard has been open this long (wall clock). |
| `loop_breaker` | `N` (or `True` = 3): blocks a tool call that already ran `N` times with the same arguments, and the next LLM call after `N` identical replies (text and tool calls, ids ignored). |
| `tool_allowlist` / `tool_denylist` | Tool names or globs (`"delete_*"`). A tool not on the allowlist, or on the denylist, is never run. |
| `validate_tool_args` | Checks the call's arguments against the JSON schema the model was given for that tool (the latest LLM request's `tools`), or the Python signature when there is none: `type`, `required`, `properties`, `additionalProperties: false`, `enum`, `minimum` / `maximum`, `minLength` / `maxLength`, `pattern`, `items`. |
| `validate_tool_args="repair"` | Instead of raising, returns `{"error": "invalid_tool_arguments", "problems": [...], "message": ...}` as the tool result, so the model can fix its call. The tool does not run. (Same as `repair_tool_args=True`.) |
| `retry_policy` | `True`, an int (max attempts), a dict, or `RetryPolicy(...)`: retries transient LLM errors (HTTP 408 / 409 / 429 / 5xx / 529, timeouts, connection errors) with exponential backoff and jitter, honouring a numeric `Retry-After`. |
| `fallback_model` | After `fallback_after` failed attempts (default: the retry policy's `max_attempts`), the call is retried on this model. |
| `block_secrets_in_output` | Redacts API keys, tokens, JWTs and private keys (the patterns `crashlab audit-secrets` uses) from LLM replies and tool results before your code sees them. `redact_pii=True` also redacts emails and phone numbers. Streamed replies are not redacted. |

```python
from crashlab.guard import RetryPolicy

policy = RetryPolicy(max_attempts=4, base_delay=0.5, max_delay=8.0, jitter=True, seed=7)
```

Retries happen inside one recorded step: the step's `crashlab.guard.attempts` says how
many attempts were made and `crashlab.guard.fallback_model` which model finally
answered. Each retry is a new call for `crashlab.faults`, so
`crashlab.faults(llm_error(429, on_calls=[1]))` plus `retry_policy=True` tests that
your agent survives a rate limit.

## What happens on a violation

`on_violation` decides:

- `"raise"` (default): the call is not made; `GuardViolation` is raised from it. The
  agent can catch it (for example to tell the model a tool is not allowed) and go on.
- `"warn"`: the violation is recorded and a `GuardWarning` is emitted; the call runs.
- `"stop"`: like `"raise"`, and the recorded run is stopped: every later LLM / tool
  call raises `CircuitBreakerError`, even if the agent caught the violation.

```python
import crashlab
from crashlab.guard import GuardViolation

try:
    with crashlab.guard(max_steps=30):
        run_agent("Refund order A17")
except GuardViolation as exc:
    print(exc.policy, exc.step_idx, exc.detail)
```

Every intervention (block, warning, retry, fallback, repair, redaction) is recorded:

- on the step, as a list under the `crashlab.guard` attribute
  (`{"policy", "action", "message", "detail"}`); a blocked or repaired call is a step
  with `crashlab.guard.blocked = true` and does not count toward the limits;
- on the run, as a `guard.<policy>` finding (`error` for a block, `warn` for a
  warning, `info` for retries, fallbacks, repairs and redactions; `stopped_run` when
  `"stop"` ended the run);
- optionally, violations go to alert sinks: `alerts=` takes a sink from
  `crashlab.alerts` (`WebhookSink`, `JsonlSink`, `CallbackSink`), any callable taking
  an `AlertEvent`, or a list of them.

```python
import crashlab
from crashlab.alerts import JsonlSink, WebhookSink

slack = WebhookSink("https://example.com/my-webhook", format="slack")
with crashlab.guard(max_cost_usd=1.0, alerts=[slack, JsonlSink("guard-alerts.jsonl")]):
    run_agent("Refund order A17")
```

List what a guard did in a run:

```bash
crashlab guard-report last
crashlab guard-report last --json
```

## Configuration

Settings come from `[tool.crashlab.guard]` in `pyproject.toml`, then the `config`
dict, then keyword arguments (later wins). Pass `pyproject=False` to ignore the file.

```toml
[tool.crashlab.guard]
max_cost_usd = 0.50
max_steps = 40
loop_breaker = 3
tool_denylist = ["delete_*"]
validate_tool_args = "repair"
retry_policy = { max_attempts = 4, base_delay = 0.5, max_delay = 8.0 }
fallback_model = "gpt-4o-mini"
block_secrets_in_output = true
on_violation = "raise"
```

```python
import crashlab

with crashlab.guard({"max_steps": 10, "tool_allowlist": ["lookup", "search"]}, max_seconds=60):
    run_agent("hi")
```

## Replay and fork

Strict `crashlab.replay` serves every call from the cassette, so a guard never re-checks,
retries or falls back during a replay, even if one is active: the replayed steps carry
the recorded `crashlab.guard` metadata, and a call the guard blocked while recording
raises the recorded `GuardViolation` again at the same step. `crashlab guard-report` on
the replay shows the same interventions. In a `crashlab.fork`, steps before the fork
step come from the cassette; calls made live after it are guarded as usual.
