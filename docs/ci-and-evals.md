# Regression evals and patch sweeps in CI

Grade a set of recorded runs (or exported cassettes) with the same oracles, runtime
checks and budget gates you use in development, and try one candidate patch across
those runs before you merge it. Nothing here calls a model unless you ask `crashlab
sweep` to fork live; `crashlab eval` is offline.

## `crashlab eval`

```bash
crashlab eval
crashlab eval --name refund-bot --since 7d --limit 50
crashlab eval recordings/*.jsonl --checks tests/oracles.py --replay
crashlab eval recordings/ --max-cost 0.05 --max-tokens 20000 --max-steps 30
crashlab eval --outcome fail --junit eval-results.xml --markdown eval-summary.md
crashlab eval --json
```

With no files, the command scans the store (`$CRASHLAB_DIR` or `./.crashlab`). Arguments
are exported `.jsonl` cassettes from `crashlab export`, directories of them, or glob
patterns. `--checks` imports a Python file (repeatable) so the `@crashlab.check`
oracles in it are registered before grading. `--replay` re-runs each cassette offline
first; a `DivergenceError` or `ReplayError` is a failure, not a crash.

The process exits `0` if every run passes, `1` if any fail or error, and `2` if there
are no matching runs or the inputs are invalid.

### What counts as a failure

- a `@crashlab.check` oracle returns a failure or errors
- a built-in runtime finding at or above `--min-finding-severity` (default `error`)
- a budget gate (`--max-cost`, `--max-tokens`, `--max-seconds`, `--max-steps`)
- a stored `outcome=fail` when no oracles were registered
- a run that ended with `status=error`

Warnings do not fail the suite unless you pass `--min-finding-severity warn`.

### JUnit XML, Markdown and GitHub Actions

`--junit PATH` writes a JUnit XML file (`<testsuites><testsuite><testcase>…`) that
GitHub, pytest and most CI UIs can display. `--markdown PATH` writes a summary table.
When `GITHUB_STEP_SUMMARY` is set (GitHub Actions), the same Markdown is also appended
there, even without `--markdown`.

```bash
crashlab eval recordings/*.jsonl --checks tests/oracles.py \
  --junit eval-results.xml --markdown eval-summary.md
```

```yaml
# .github/workflows/crashlab-eval.yml
name: crashlab eval
on:
  pull_request:
  push:
    branches: [main]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install crashlab
      - name: Grade exported cassettes
        run: |
          crashlab eval recordings/*.jsonl --checks tests/oracles.py \
            --max-cost 0.05 --junit eval-results.xml --markdown eval-summary.md
```

The Python API is the same report the CLI prints:

```python
from pathlib import Path
from crashlab.suite import eval_runs

report = eval_runs(
    files="recordings/*.jsonl",
    check_modules=["tests/oracles.py"],
    replay=False,
    max_cost_usd=0.05,
    max_tokens=20_000,
    max_seconds=30,
    max_steps=40,
    min_finding_severity="error",
)
Path("eval-results.xml").write_text(report.to_junit_xml())
Path("eval-summary.md").write_text(report.to_markdown())
assert report.failed == 0 and report.errored == 0
print(report.passed, "of", report.total, "passed;", report.pass_rate)
print("cost", report.total_cost_usd, "tokens", report.total_tokens)
```

`eval_runs` also accepts a `Store`, and the filters `name=`, `tag=`, `outcome=`,
`since=` (`"7d"` or a datetime) and `limit=`.

## `crashlab sweep`

Try one candidate patch on many recorded runs and see which previously failing runs it
fixes, and which previously passing runs it breaks. Each run is forked with
`crashlab.fork` (steps before `--at-step` come from the cassette; from that step on
the agent continues live with the patch applied).

```bash
crashlab sweep --patch fix.json --name refund-bot --outcome fail
crashlab sweep --system-prompt "Never refund more than the order total." --at-step 0
crashlab sweep 01K1ABC 01K1DEF --model gpt-4o-mini --json
crashlab sweep --patch fix.json --markdown sweep-summary.md --fail-on-regression
```

Give `--patch FILE` and / or `--system-prompt TEXT` and / or `--model`. `--agent NAME`
scopes request patches to that agent's LLM calls. `--fail-on-regression` exits `1` if
any previously passing run fails after the patch; without it the command exits `0`
even when some runs still fail (the report still lists them). Missing runs or a missing
patch exit `2`.

Each run is classified as `fixed` (bad → good), `regressed` (good → bad),
`still_passing`, `still_failing`, or `errored` (the fork itself failed). The report
also has the cost and token deltas.

```python
from crashlab import patch
from crashlab.suite import sweep_runs

report = sweep_runs(
    runs,
    patch.SystemPrompt(op="append", text="Never refund more than the order total."),
    at_step=0,
    max_cost_usd=0.05,
    max_tokens=20_000,
)
print(report.to_markdown())
print(report.fixes, "fixed;", report.regressions, "regressed")
print("cost delta", report.cost_delta_usd, "token delta", report.token_delta)
```

`sweep_runs` also accepts a path to a JSON or YAML patch file, and `model=` / `agent=`
to apply on top of it. Forks are stored in `store` (the default store when omitted).

`--system-prompt` appends to the recorded system prompt, the same as
`SystemPrompt(op="append", …)`. Use a patch file when you need replace / substitute or
a tool-description change.

YAML patch files need `pip install "crashlab[yaml]"`.
