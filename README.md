# crashlab

**Crash-test your AI agents.**

crashlab is an open-source, framework-agnostic Python library that turns agent failures into fixes you can trust. The goal for v0.1:

- **Record** every LLM and tool call from your existing agent loop (OpenAI, Anthropic, LiteLLM), with no rewrite.
- **Replay** any failed run exactly, offline, and fork it at any step.
- **Diagnose** the failing step and its root cause, backed by verbatim evidence from the trace.
- **Verify** a proposed fix by rerunning the failure and your past good runs.
- **Export** the failure as a pytest regression test.

> **Status: pre-alpha (0.0.1).** Today crashlab **records** runs and lets you inspect
> them. Replay, fork, diagnose, verify and test export are in progress (see
> [ARCHITECTURE.md](ARCHITECTURE.md)) and are not available yet. Not on PyPI yet.

## What works today

- `crashlab.wrap(client)` records `chat.completions.create` calls on OpenAI clients
  (sync and async) and any OpenAI-compatible endpoint (Ollama, vLLM, LiteLLM proxy ...):
  normalized request/response, tool calls, token usage, finish reason, errors.
- `@crashlab.tool(side_effects=...)` records tool arguments, results and exceptions.
- `@crashlab.agent(name=...)` and `with crashlab.record(name, input=...)` mark run
  boundaries (input, output, status, usage totals, entry point, git sha).
- Everything goes to a local store, `.crashlab/` (or `$CRASHLAB_DIR`): a SQLite index plus
  one JSONL cassette per run. API keys and auth headers are redacted before writing.
- `crashlab ls` and `crashlab show <run_id|last>` in the terminal; `crashlab.load("last")`
  in Python.

Not yet: Anthropic and LiteLLM wrappers (they raise `NotImplementedError`), streaming
capture (`stream=True` calls pass through but the streamed response is not recorded),
`responses.create`, replay, runtime checks, diagnosis, verification, export.

## Quickstart

```bash
pip install "crashlab[openai] @ git+https://github.com/Mpasha17/crashlab"
```

```python
import crashlab
from openai import OpenAI

client = crashlab.wrap(OpenAI())          # or OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

@crashlab.tool()
def get_order(order_id: str) -> dict: ...

@crashlab.tool(side_effects=True)
def issue_refund(order_id: str, amount: float) -> dict: ...

@crashlab.agent(name="refund-bot")
def run_agent(user_msg: str) -> str:
    ...                                    # your existing tool-calling loop, unchanged

run_agent("Refund my order A17, it arrived broken")

run = crashlab.load("last")                # Run with .steps (chat / execute_tool)
```

```bash
crashlab ls          # runs: id, name, status, outcome, steps, tokens, started
crashlab show last   # step table: idx, op, name, args / result, error
```

Try the bundled demo agent, free with a local model through Ollama:

```bash
git clone https://github.com/Mpasha17/crashlab && cd crashlab
python -m venv .venv && . .venv/bin/activate && pip install -e ".[openai]"
ollama pull qwen2.5:7b
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama MODEL=qwen2.5:7b \
  python -m examples.refund_bot "Refund my order A17, it arrived broken"
FAULT_HTML_502=1 OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama MODEL=qwen2.5:7b \
  python -m examples.refund_bot "Refund my order A17"     # get_order returns an HTML 502 page
crashlab ls && crashlab show last
```

See [examples/refund_bot](examples/refund_bot/README.md) for details.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,openai]"
ruff check . && ruff format --check . && pytest
```

The tests use a fake OpenAI-like client and a mocked HTTP transport, so they need no
network access and no API key.

## License

Apache-2.0
