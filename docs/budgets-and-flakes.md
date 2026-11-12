# Budget oracles and flake detection

Two deterministic tools for debugging agents in production-like runs. Both read only
what crashlab recorded; neither calls a model.

## Budget oracles: `crashlab.budgets.within_budget`

An outcome oracle (pass / fail) for cost, tokens, latency and step counts. Use it inside
a `@crashlab.check`; it returns `None` when every limit holds and a failure message
otherwise (a check returning a non-empty string fails with it):

```python
import crashlab
from crashlab.budgets import within_budget

@crashlab.check("refund-bot")
def cheap_and_fast(run):
    return within_budget(
        run,
        max_cost_usd=0.02,           # summed estimated cost of the chat calls
        max_tokens=20_000,           # also max_input_tokens, max_output_tokens
        max_call_input_tokens=8_000, # the largest single prompt (context size)
        max_seconds=30,              # the run's wall time
        max_call_seconds=10,         # the slowest single live LLM or tool call
        max_llm_calls=8,             # also max_steps, max_tool_calls
    )
```

A failing check reads e.g.
`budget: 1,040 tokens > max_tokens 500; 900 input tokens at step 2, grew from 100 > max_call_input_tokens 500`.
Because it is a check, it decides the run's outcome, works with `crashlab check`, the
pytest plugin, exported tests and `crashlab verify` (a fix must also stay in budget).

`crashlab.budgets.summarize_usage(run)` returns the numbers it uses (tokens, cost, wall
time, slowest call, largest prompt, and which steps are missing data).

### One agent of a multi-agent run

`within_budget(run, agent="researcher", max_llm_calls=4, max_cost_usd=0.01)` counts only
that agent's steps (its label, see [multi-agent.md](multi-agent.md); `"researcher#2"` for
one of two concurrent runs of it). The message then starts `budget (agent researcher):`.
An agent with no steps fails (naming the agents the run has). `max_seconds` is the run's
wall time and is refused with `agent=`; `max_call_seconds` works.
`summarize_usage(run, agent=...)` gives the numbers.

### Where the numbers come from

- Tokens: the usage each chat step's provider reported (`gen_ai.usage.*` attributes).
  `max_tokens` uses the provider's total when it reports one (it can exceed input +
  output, e.g. Gemini thinking tokens).
- Cost: the cost stored on the step, else `crashlab.pricing` (built-in dated price table,
  your `[tool.crashlab.prices]`, LiteLLM's cost map when installed). An estimate, not your
  invoice.
- Wall time: the run's measured wall time (`run.usage.wall_time_ms`), else
  `ended_at - started_at`.
- Per-call latency: `duration_ms` of live chat and tool steps.
- Embeddings calls are not counted in tokens or cost.

### Missing data is not a pass

If a limit needs a number the recording does not have, the check fails by default and
says so (`cannot check: ...`):

- a chat call reported no token usage (typically a stream without
  `stream_options={"include_usage": True}`) — failed requests that report nothing are
  treated as using nothing, not as missing;
- the model is not in the price table (the message names the model);
- a run that has not ended (no wall time), or no live call to time.

Pass `unknown="pass"` to skip such limits instead. A limit already exceeded by the known
part still fails either way.

### Replays, forks, exported tests and verify

A strict replay (`crashlab replay`, exported regression tests) and a fork (`crashlab
fork`, `crashlab verify` trials) serve recorded steps from the cassette, which takes no
real time. By default (`replay_time="pass"`) `max_seconds` is not evaluated on them and
`max_call_seconds` only times their live steps (the steps after a fork point; none in a
strict replay): time is judged on the live recording. That keeps an exported test of a
run that met its time budget passing on replay. `replay_time="fail"` reports "cannot
check" instead. Tokens and cost are still checked on replays and forks, from the usage
recorded on each step.

## Runtime budget detector: per-call limits

The built-in `budget` runtime check (findings, not pass / fail) gains two opt-in limits:

```toml
[tool.crashlab.checks.budget]
max_call_input_tokens = 8000   # one LLM call's prompt; growing history shows up here
max_call_seconds = 20          # one live LLM or tool call
```

or `checks={"max_call_input_tokens": 8000}`. Both are off by default, critical when
exceeded (so the opt-in circuit breaker can stop the run) and have no 80% early warning.
When `max_tokens` or `max_call_input_tokens` is set and some LLM call reported no usage,
the detector adds an `info` finding (`max_tokens:missing`): those calls count as 0, so
the limit is only a lower bound. Existing limits and their behaviour are unchanged.

## Flake detection: `crashlab flakes`

Finds inputs that the same agent sometimes handles and sometimes does not:

```bash
crashlab flakes                       # flaky inputs among the newest 500 runs
crashlab flakes --name refund-bot --min-runs 3
crashlab flakes --json                # machine-readable
crashlab flakes --fail-on-flaky       # exit 1 in CI when something is flaky
crashlab flakes --all                 # also list inputs whose runs all agreed
```

```text
FLAKY  refund-bot  input {"msg":"Refund A17"}  code 3f2a9c1e0b7d  (5 runs, 2 distinct outputs)
  raised: 1/5 runs (20%, 95% CI 4%-62%)  [openai.APITimeoutError]
    compare: crashlab diff 01K... 01K...
  check:refund_not_above_total: 2/5 runs (40%, 95% CI 12%-77%)  [fail]
    compare: crashlab diff 01K... 01K...      (printed again only when the pair changes)
  models: gpt-4o-mini
1 flaky of 3 repeated input(s) (11 runs). Counts are of these runs; ...
```

Library: `crashlab.flakes.find_flakes(store=None, *, name=None, limit=500, min_runs=2,
min_severity="warn", across_versions=False, include_stable=False, runs=None)` returns
`FlakeGroup`s (flaky first), each with `signals` (`name`, `bad`, `n`, `ci95`,
`bad_runs`, `good_runs`, `detail`), `models`, `distinct_outputs` and `notes`.

### What "same input" means

Runs are compared when they have all of:

- the same agent `name`;
- the same recorded `run.input` (the agent's bound arguments as stored, after
  redaction), compared as canonical JSON — key order does not matter, values do;
- the same `code_version` (git commit, `-dirty` kept). Runs of different commits are
  different code, not a flake; `--across-versions` groups them anyway and says so.

Strict replays, forks and runs still recording are excluded. Not part of the key: the
model, prompt files, external state the tools read, the environment. When runs in a group
called different models, the report notes it; `-dirty` versions get a note too, since
uncommitted code may differ between runs.

### Signals

A signal is reported only when it differs inside a group (some runs bad, some good):

- `raised`: the run ended with an exception (types listed);
- `outcome`: `fail` vs `pass` (runs with outcome `unknown` are not counted);
- `check:<name>`: a `@crashlab.check` failed or errored, counted over runs that
  evaluated that check;
- `finding:<kind>`: a runtime-check finding of at least `--min-severity` (default
  `warn`), counted over runs where runtime checks ran with that detector enabled.

`distinct_outputs` (how many different final outputs the successful runs gave) is shown
but not flagged: different wording is not a failure.

### Statistics, honestly

The numbers are counts of the runs in your store. The 95% interval is a Wilson score
interval for the bad-run rate and assumes the runs are independent repetitions of the
same thing; with a handful of runs it is wide (1 of 5 is 4%–62%). An input that never
varied in 3 runs can still be flaky: the report does not claim otherwise. To get more
evidence, run the input more times, or use `crashlab verify`, which runs controlled
trials. Use the printed `crashlab diff` command to see where a good and a bad run first
diverge.
