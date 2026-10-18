# crashlab

**Crash-test your AI agents.**

crashlab is an open-source, framework-agnostic Python library that turns agent failures into fixes you can trust. The goal for v0.1:

- **Record** every LLM and tool call from your existing agent loop (OpenAI, Anthropic, LiteLLM), with no rewrite.
- **Replay** any failed run exactly, offline, and fork it at any step.
- **Diagnose** the failing step and its root cause, backed by verbatim evidence from the trace.
- **Verify** a proposed fix by rerunning the failure and your past good runs.
- **Export** the failure as a pytest regression test.

> **Status: pre-alpha (0.0.1).** Today crashlab **records** runs, lets you inspect them,
> **replays** them strictly offline, grades them with **`@crashlab.check`** oracles,
> **forks** them at any step with typed **patches** and **diffs** two runs. Diagnose,
> verify and test export are in progress (see [ARCHITECTURE.md](ARCHITECTURE.md)) and are
> not available yet. Not on PyPI yet.

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
- **Fork at step N with a patch:** `crashlab.fork(run, at_step=N, patch=...)` /
  `crashlab fork <run> --at N ...` re-runs the agent with steps before N served strictly from
  the cassette and step N onwards live (or against a client you pass), with the patch
  applied to every LLM request from N on. Tool calls identical to the parent's are served
  from the cassette; `side_effects=True` tools never run live (an identical call gets the
  recorded result, anything else is blocked and recorded as an attempt). The fork is a new
  run linked by `forked_from`, `fork_step` and the patch.
- **Typed patches** (pydantic, JSON/YAML-serializable, composable with `+`, each renders a
  human-readable diff): `SystemPromptPatch` (replace / append / prepend / insert, replace
  or delete a line / substitute), `MessagePatch`, `InjectMessagePatch`, `ToolResultPatch`
  (counterfactual tool result or error), `ToolDescriptionPatch`, `ToolSchemaPatch`,
  `ModelParamPatch` (model and params). Work for OpenAI chat / Responses, Anthropic and
  LiteLLM.
- **`crashlab diff A B`** / `crashlab.diff(a, b)`: how the runs relate (fork / replay),
  outcome and check changes, tokens / steps / time side by side, a step-by-step comparison
  (same, different request, different response, only in A / B) and the first divergence
  with a field diff. `B` defaults to `A`'s parent.
- **Runtime checks** (built in, on by default): cheap deterministic detectors that run
  while the agent is recorded and on stored runs (`crashlab check`). They emit typed
  findings (kind, severity `info` / `warn` / `error` / `critical`, steps, a message and
  verbatim evidence quotes):
  `step_repetition` (same tool + arguments again and again, near-identical LLM requests),
  `tool_error_streak`, `bad_tool_args` (validated against the tool's JSON schema),
  `suspicious_tool_result` (HTML / error pages, empty or truncated results, a result whose
  shape drifted from earlier results of the same tool), `budget` (steps, LLM / tool calls,
  tokens, USD, seconds) and `length_cutoff`. Findings are stored with the run and shown by
  `show`, `diff` and `ls --flagged`. They describe *how* a run behaved; `@crashlab.check`
  oracles still decide whether it passed. Opt-in **circuit breaker**: a critical finding
  stops the agent (`CircuitBreakerError`), and strict replay stops at the same step.
- **Cost estimates** per step and per run in USD, from a small built-in price table
  (OpenAI, Anthropic, Groq, Gemini; dated prices) plus LiteLLM's cost map when LiteLLM is
  installed. Local servers (Ollama, localhost) cost $0. Unknown models show no cost; prices
  are never guessed. Your own prices: `crashlab.pricing.set_price(...)` or
  `[tool.crashlab.prices]`.
- **Share runs:** `crashlab export RUN -o run.jsonl` / `crashlab import run.jsonl`
  (redacted again on the way out).
- **CLI:** `crashlab ls`, `crashlab show <run_id|last>`, `crashlab replay`, `crashlab check`,
  `crashlab fork`, `crashlab diff`, `crashlab export`, `crashlab import`; in Python,
  `crashlab.load("last")`.

Not yet: diagnosis, verification, test export, HTML report. Streamed calls replay as a synthetic stream rebuilt from the recorded
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

# what if the prompt had said so? fork at the first LLM call with a patch, run live
branch = crashlab.fork(run, at_step=0, patch=crashlab.SystemPromptPatch(
    op="append", text="Never refund more than the order total."))
print(crashlab.diff(run, branch).summary())
```

```bash
crashlab ls            # runs: id, name, status, outcome, steps, tokens, cost, findings
crashlab ls --flagged  # only runs with runtime-check findings (--min-severity error)
crashlab show last     # checks, findings + step table: idx, op, name, args / result, error
crashlab replay last   # offline re-run from the cassette; exit 1 + diff on divergence
crashlab check last    # re-run the oracles and the runtime checks (--save to store them)
crashlab check --all --findings-only --max-cost 0.05 --fail-on error   # e.g. in CI
crashlab fork last --at 0 --system-append "Never refund more than the order total."
crashlab fork last --at 2 --patch fix.yaml --dry-run   # show the patch diff, run nothing
crashlab diff last     # the fork vs its parent: checks, usage, steps, first divergence
```

Runtime checks are configured per run or in `pyproject.toml`, and `record(checks=...)`
wins over pyproject:

```python
@crashlab.agent(name="refund-bot", checks={"max_steps": 20, "max_cost_usd": 0.10,
                                           "circuit_breaker": True})
```

```toml
[tool.crashlab.checks]
circuit_breaker = true          # stop the agent on a critical finding (default: off)
max_steps = 20                  # budget: also max_llm_calls, max_tool_calls, max_tokens,
max_cost_usd = 0.10             #   max_seconds; warn at 80% (budget.warn_at)
[tool.crashlab.checks.step_repetition]
threshold = 3                   # warn at 3 identical tool calls, critical at 5
```

### Try it with no API key and no model

The repo ships five real recorded runs of the demo agent (secrets and local paths scrubbed):
a clean refund, a policy violation the oracles fail, a 502 retry loop the runtime checks
flag, a run the circuit breaker stopped, and a small local model passing a bad argument.

```bash
git clone https://github.com/Mpasha17/crashlab && cd crashlab
python -m venv .venv && . .venv/bin/activate && pip install -e ".[openai]"
export CRASHLAB_DIR=/tmp/crashlab-demo
for f in examples/refund_bot/demo/*.jsonl; do crashlab import "$f"; done
crashlab ls                        # 5 runs, with cost and findings
crashlab show 01M57GM9Q5           # Groq gpt-oss-120b retrying an HTML 502 page 3 times
crashlab check --all               # oracles + runtime checks, offline (exit 1: C03 fails)
crashlab replay 01M57GM7H9         # strict offline replay of the policy violation
crashlab fork 01M57GM7H9 --at 0 --dry-run \
  --system-append "Never refund an order delivered more than 30 days ago."
```

Running the fork for real (without `--dry-run`) needs a model, for example a local one:

### Free, local, with Ollama

```bash
ollama pull qwen2.5:3b             # or llama3.2:3b / qwen2.5:7b: any tool-calling model
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama MODEL=qwen2.5:3b \
  python -m examples.refund_bot "Refund my order A17, it arrived broken"
FAULT_HTML_502=1 OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama MODEL=qwen2.5:3b \
  python -m examples.refund_bot "Refund my order A17"     # get_order returns an HTML 502 page
# or through LiteLLM (pip install -e ".[litellm]"): ollama_chat/ uses Ollama's native tools
python -m examples.refund_bot --provider litellm --model ollama_chat/qwen2.5:3b "Refund A17"
crashlab ls && crashlab show last  # cost: $0
crashlab replay last               # no model server needed

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
