# refund-bot

A small customer-support agent: a plain tool-calling loop (OpenAI-compatible or Anthropic)
with two tools and a mock in-memory order DB, instrumented with crashlab.

- `get_order(order_id)`: recorded tool, safe to re-run.
- `issue_refund(order_id, amount)`: recorded with `side_effects=True` (it "moves money").
- The refund policy lives in the system prompt (look up first, only delivered orders
  within 30 days, never more than the order total, one refund per order).
- Mock orders: `A17` (delivered 5 days ago, $49), `B22` (shipped, not delivered),
  `C03` (delivered 45 days ago, $120).

## Run it

From the repo root, with crashlab installed (`pip install -e ".[openai]"`):

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
MODEL=gpt-4o-mini python -m examples.refund_bot "Refund my order A17, it arrived broken"

crashlab ls
crashlab show last
```

Any OpenAI-compatible endpoint works through `OPENAI_BASE_URL`, `OPENAI_API_KEY` and
`MODEL` (vLLM, LM Studio, LiteLLM proxy, OpenRouter ...). Tested live with:

| provider | `OPENAI_BASE_URL` | `MODEL` |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-flash-latest` |

Gemini 3 models attach a `thought_signature` to each tool call and reject the next turn
if it is not sent back. The agent echoes tool calls with `model_dump()` so the signature is
kept; do the same in your own agents.

### Free, local, with Ollama

```bash
ollama pull qwen2.5:3b            # any model with tool calling: llama3.2:3b, qwen2.5:7b, ...
ollama serve                      # if it is not already running

export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama      # Ollama ignores the key, but the client needs one
export MODEL=qwen2.5:3b
python -m examples.refund_bot "Refund my order A17, it arrived broken"
crashlab show last                # est. $0: local servers are free
```

Or through LiteLLM (`pip install -e ".[litellm]"`), with the same loop and tools:

```bash
python -m examples.refund_bot --provider litellm --model ollama_chat/qwen2.5:3b "Refund A17"
PROVIDER=litellm MODEL=ollama_chat/llama3.2:3b python -m examples.refund_bot "Refund A17"
```

Use `ollama_chat/...` (Ollama's chat API with native tool calling). LiteLLM's older
`ollama/...` route emulates tools through JSON prompting: in our runs qwen2.5:3b on it
once repeated `get_order` three times (flagged by `step_repetition`, failed by the step
budget oracle) and once answered `{}` without calling a tool. `--provider litellm`
also takes any other LiteLLM model (`groq/...`, `gemini/...`) with that provider's key.

Small local models are less reliable at tool calling, which makes them good at producing
failures to look at: llama3.2:3b called `issue_refund` with `amount: "0"` (a string where
the schema says number, flagged by `bad_tool_args`), and both 3B models sometimes promise
a refund in the reply without calling `issue_refund`.

### Anthropic

```bash
pip install -e ".[anthropic]"
export PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...
MODEL=claude-haiku-4-5 python -m examples.refund_bot "Refund my order A17, it arrived broken"
crashlab show last
```

The Anthropic loop uses `messages.create` with the same tools and system prompt; crashlab
records it in the same common shape, so `crashlab show` looks the same for both providers.
`--provider anthropic` works too.

## Inject a fault

| fault | env var / flag | what changes |
|---|---|---|
| `html_502` | `FAULT_HTML_502=1` / `--fault-html-502` | `get_order` returns an nginx `502 Bad Gateway` HTML page instead of JSON |
| `schema_drift` | `FAULT_SCHEMA_DRIFT=1` / `--fault-schema-drift` | `get_order` returns valid JSON, but `total` is renamed `amount_total` |
| `policy_drop` | `FAULT_POLICY_DROP=1` / `--fault-policy-drop` | policy line 2 (only delivered orders, at most 30 days ago) is removed from the system prompt |

```bash
FAULT_HTML_502=1 python -m examples.refund_bot "Refund my order A17"
FAULT_SCHEMA_DRIFT=1 python -m examples.refund_bot "Refund my order A17"
# policy_drop only bites on an order outside the policy: C03 was delivered 45 days ago
FAULT_POLICY_DROP=1 python -m examples.refund_bot "Please refund order C03, the shoes fell apart"
crashlab show last
```

The active faults are recorded in the run input (`faults: [...]`), together with the
resolved `model` and `provider`, so `crashlab replay` reproduces a faulty run without any
env vars. The tool result itself is served from the cassette during replay.

What we saw in one live pass (2026-10-18, one run per case, so treat it as anecdote):
`openai/gpt-oss-120b` (Groq) and `gemini-3.7-flash` both refused C03 with the full policy
and both **refunded it ($120) under `policy_drop`**, which `refund_within_policy` flagged.
Neither was fooled by `schema_drift` (both refunded the correct $49 for A17), and both
stopped and apologised on `html_502`.

## Checks: pass / fail instead of "unknown"

`agent.py` registers three `@crashlab.check("refund-bot")` oracles. They grade each run
against the real order database, not against what the (possibly faulty) tools returned:

- `refund_amount_equals_order_total`: every refund issued is for exactly the order total.
- `refund_within_policy`: refunds only after a `get_order` lookup, only for delivered orders
  inside the 30-day window, at most one per order, never above the total.
- `within_step_budget`: at most 8 steps (the happy path takes 5), which catches lookup loops.

They run when the run ends; `crashlab show` lists them and the run gets `outcome: pass` or
`fail` with the first failing check as the reason. Refund attempts that the tool rejected
(it raises for amounts above the total) are not counted as refunds.
`crashlab check <run_id>` re-runs the checks on an older run and stores the new outcome.

## Runtime checks

Besides the oracles, the built-in runtime checks watch every run (no setup needed):

```bash
FAULT_HTML_502=1 python -m examples.refund_bot "Refund my order A17"
crashlab show last        # runtime findings: e.g. tool_error_streak (error) at steps 1, 3, 5:
                          # "get_order failed 3 times in a row ... HTML page (502 Bad Gateway)"
crashlab ls --flagged     # runs with findings
crashlab check last --findings-only --max-steps 4 --fail-on error   # stricter, after the fact
```

To stop a run instead of only flagging it, turn on the circuit breaker, e.g. in a
`pyproject.toml` next to where you run the agent:

```toml
[tool.crashlab.checks]
circuit_breaker = true
max_steps = 6
```

A critical finding (a budget exceeded, 5 identical tool calls, 5 tool failures in a row)
then raises `crashlab.CircuitBreakerError` right after the step that caused it. The run is
stored with `status=error` and `outcome=fail` ("stopped by crashlab: [budget] ..."), and
`crashlab replay` stops at the same step.

## Replay offline

```bash
unset OPENAI_API_KEY ANTHROPIC_API_KEY      # not needed
crashlab replay last                        # identical run, nothing sent, no refund issued
```

Every LLM call and every tool call (including `issue_refund`, which has side effects) is
served from the cassette; the replay is stored as a new run with `replay_of` set. If the
code now sends a different request (edit the system prompt and try again), the replay stops
with a `DivergenceError` that shows the step and a diff of what changed.

## Fork with a patch: "what if the rule were back?"

`FAULT_POLICY_DROP=1` removes the 30-day refund-window rule from the system prompt, so the
bot refunds C03 (delivered 45 days ago) and `refund_within_policy` fails. Fork that run at
its first LLM call with the rule put back and run it live:

```bash
FAULT_POLICY_DROP=1 python -m examples.refund_bot "Please refund order C03, it fell apart"
crashlab fork last --at 0 --patch examples/refund_bot/restore_policy.yaml
# or with flags:
# crashlab fork last --at 0 --system-insert-after "Always look up the order" \
#   '- Only refund orders whose status is "delivered" and that were delivered at most 30 days ago.'
crashlab diff last          # outcome fail -> pass, refund_within_policy changed, steps 5 -> 3
```

The fork re-sends the conversation with the patched prompt; `get_order("C03")` is served
from the parent's cassette, and `issue_refund` never runs live (an identical refund gets the
recorded result, a different one is blocked and still counts as an attempt for the checks).

To count flips over several tries, with a control (unpatched forks of the same run):

```bash
python -m examples.refund_bot.fork_demo last -k 3 --control
```

Measured on 2026-10-18 (one recorded policy-drop run per model, forked at step 0):

| model | patched forks | unpatched control forks |
|---|---|---|
| Groq `openai/gpt-oss-120b` | 4 / 4 pass (3 via `fork_demo`, 1 via the CLI), refund refused | 3 / 3 fail (refund issued again) |
| Gemini `gemini-3.7-flash` | 3 / 3 pass, refund refused | 2 fail (refund issued); 1 hit the free-tier rate limit |

This is a small sample, not a guarantee; the point is that the counterfactual is one
command and each try is a stored run you can `crashlab show` and `crashlab diff`.
