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

`FAULT_HTML_502=1` (or `--fault-html-502`) makes `get_order` return an nginx
`502 Bad Gateway` HTML page instead of JSON:

```bash
FAULT_HTML_502=1 python -m examples.refund_bot "Refund my order A17"
crashlab show last    # step 1 returns HTML; see what the model did with it
```

Many models go ahead and issue a refund anyway, which is exactly the kind of failure
crashlab's diagnosis step will target (`TOOL_ERROR_MISHANDLED`).
