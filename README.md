# crashlab

**Crash-test your AI agents.**

crashlab is an open-source, framework-agnostic Python library that turns agent failures into fixes you can trust. The goal for v0.1:

- **Record** every LLM and tool call from your existing agent loop (OpenAI, Anthropic, LiteLLM), with no rewrite.
- **Replay** any failed run exactly, offline, and fork it at any step.
- **Diagnose** the failing step and its root cause, backed by verbatim evidence from the trace.
- **Verify** a proposed fix by rerunning the failure and your past good runs.
- **Export** the failure as a pytest regression test.

> **Status: pre-alpha (0.0.1).** Today crashlab **records** runs, lets you inspect them,
> **replays** them strictly offline and grades them with **`@crashlab.check`** oracles.
> Fork, diagnose, verify and test export are in progress (see
> [ARCHITECTURE.md](ARCHITECTURE.md)) and are not available yet. Not on PyPI yet.

## What works today

- **`crashlab.wrap(client)`** records LLM calls, sync and async:
  - OpenAI `OpenAI` / `AsyncOpenAI` and any OpenAI-compatible endpoint (Ollama, vLLM,
    LiteLLM proxy ...): `chat.completions.create` and `responses.create`.
  - Anthropic `Anthropic` / `AsyncAnthropic`: `messages.create` and `messages.stream(...)`.
- **LiteLLM:** `crashlab.integrations.litellm.patch()` (or `crashlab.wrap(litellm)`) records
  `litellm.completion` / `litellm.acompletion`; `unpatch()` restores them.
- **Streaming:** with `stream=True` (and Anthropic's `messages.stream()`), chunks reach your
  code unchanged; crashlab assembles and records the final response when the stream ends,
  errors, or is closed early (marked incomplete).
- **One common shape** for every provider: system, messages as typed parts (text,
  tool calls, tool results, images/audio/files as placeholders, reasoning), tools, params,
  plus finish reason and token usage. The same conversation on OpenAI and Anthropic
  normalizes to the same messages, and each request gets a stable sha256 `request_hash`.
- **`@crashlab.tool(side_effects=...)`** records tool arguments, results and exceptions;
  **`@crashlab.agent(name=...)`** / **`with crashlab.record(name, input=...)`** mark runs
  (input, output, status, usage totals, entry point, git sha).
- **Threads:** `crashlab.bind_context(fn)`, `crashlab.ContextThreadPoolExecutor` and
  `crashlab.run_in_thread(fn, ...)` keep recording tool/LLM calls made in worker threads
  (asyncio tasks and `asyncio.to_thread` work out of the box).
- **Local store** `.crashlab/` (or `$CRASHLAB_DIR`): a SQLite index plus one JSONL cassette
  per run. API keys, auth headers (`Authorization`, `x-api-key` ...) and token-like strings
  are redacted before anything is written; add your own rules with
  `crashlab.redaction.add_pattern(...)`, `add_key(...)`, `add_hook(fn)`.
- **Strict offline replay:** `crashlab.replay(run)` / `crashlab replay <run_id|last>`
  re-runs the recorded entry point with every LLM and tool call served from the cassette:
  no API key, no network (outbound connections are blocked), and tools never run, so
  `side_effects=True` tools cannot fire. Calls are matched by step order plus the normalized
  request hash; anything else raises `DivergenceError` with the step, the differing fields
  and a diff. Works for OpenAI chat / Responses, Anthropic (`messages.stream()` included) and
  LiteLLM, plain and streamed, sync and async. The replay is stored as a new run with
  `replay_of` pointing at the source.
- **Outcome oracles:** `@crashlab.check("agent-name")` functions get the finished run and
  pass (return `True` / `None`) or fail (`False`, a message string, or a failed `assert`).
  They run when a run ends and set `outcome` to `pass` / `fail`;
  `crashlab check <run_id>` re-runs them on a stored run.
- **CLI:** `crashlab ls`, `crashlab show <run_id|last>`, `crashlab replay`, `crashlab check`;
  in Python, `crashlab.load("last")`.

Not yet: fork, runtime checks (loop / error-streak flags), diagnosis, verification, test
export, HTML report. Streamed calls replay as a synthetic stream rebuilt from the recorded
final response (same content, different chunk boundaries).
Also not intercepted yet: Anthropic `client.beta.messages`, OpenAI
`chat.completions.stream()` / `responses.stream()` helpers, `with_raw_response` calls.
OpenAI streams only report token usage if you pass `stream_options={"include_usage": True}`.

## Quickstart

```bash
pip install "crashlab[openai] @ git+https://github.com/Mpasha17/crashlab"
# or crashlab[anthropic], crashlab[litellm]
```

```python
import crashlab
from openai import OpenAI

client = crashlab.wrap(OpenAI())          # or OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# client = crashlab.wrap(anthropic.Anthropic())

@crashlab.tool()
def get_order(order_id: str) -> dict: ...

@crashlab.tool(side_effects=True)
def issue_refund(order_id: str, amount: float) -> dict: ...

@crashlab.agent(name="refund-bot")
def run_agent(user_msg: str) -> str:
    ...                                    # your existing tool-calling loop, unchanged

@crashlab.check("refund-bot")              # outcome oracle: pass / fail
def refund_not_above_total(run) -> None:
    for step in run.tool_calls("issue_refund", ok_only=True):
        assert step.args["amount"] <= 49.0, f"refunded {step.args['amount']}"

run_agent("Refund my order A17, it arrived broken")

run = crashlab.load("last")                # Run with .steps (chat / execute_tool), .outcome
replayed = crashlab.replay(run)            # offline; raises DivergenceError if it differs
```

```bash
crashlab ls            # runs: id, name, status, outcome, steps, tokens, started
crashlab show last     # checks + step table: idx, op, name, args / result, error
crashlab replay last   # offline re-run from the cassette; exit 1 + diff on divergence
crashlab check last    # re-run the @crashlab.check oracles, store pass / fail
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
crashlab replay last                                      # no model server needed

# the same agent on Anthropic
PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... MODEL=claude-haiku-4-5 \
  python -m examples.refund_bot "Refund my order A17, it arrived broken"
```

See [examples/refund_bot](examples/refund_bot/README.md) for details.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,openai,anthropic,litellm]"
ruff check . && ruff format --check . && pytest
```

The tests use fake clients and the real SDKs over mocked HTTP transports (LiteLLM via
`mock_response`), so they need no network access and no API key. Tests for SDKs that are
not installed are skipped.

## License

Apache-2.0
