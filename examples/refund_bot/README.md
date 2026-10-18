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
ollama pull qwen2.5:7b            # any model with tool-calling support, e.g. llama3.1:8b
ollama serve                      # if it is not already running

export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama      # Ollama ignores the key, but the client needs one
export MODEL=qwen2.5:7b
python -m examples.refund_bot "Refund my order A17, it arrived broken"
crashlab show last
```

Small local models are less reliable at tool calling, so expect more interesting failures.

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

## Replay offline

```bash
unset OPENAI_API_KEY ANTHROPIC_API_KEY      # not needed
crashlab replay last                        # identical run, nothing sent, no refund issued
```

Every LLM call and every tool call (including `issue_refund`, which has side effects) is
served from the cassette; the replay is stored as a new run with `replay_of` set. If the
code now sends a different request (edit the system prompt and try again), the replay stops
with a `DivergenceError` that shows the step and a diff of what changed.
