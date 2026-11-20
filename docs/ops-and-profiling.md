# Operational diagnostics, project scaffolding and run profiling

Check that the local store is healthy, scaffold a CI eval workflow into a new
project, and break down a recorded run by time, tokens, cost and prompt-cache
opportunity. Nothing here calls a model.

## `crashlab doctor`

```bash
crashlab doctor
crashlab doctor --fix
crashlab doctor --json
```

Inspects `$CRASHLAB_DIR` (or `./.crashlab`):

- the store directory and SQLite index (`PRAGMA integrity_check`, schema
  version, run count, database and cassette disk size)
- orphaned child runs (`parent_run_id` / `forked_from` / `replay_of` pointing
  at a missing run) and a dangling `last` run whose files are gone
- optional packages (`openai`, `anthropic`, `litellm`, `langchain`, `mcp`,
  `pydantic_ai`, `pytest`, `pyyaml`) — missing extras are reported, never
  imported, and never crash the doctor
- redaction status and crashlab environment variables (`CRASHLAB_DIR`,
  `CRASHLAB_MODE`, `CRASHLAB_REDACT`, the diagnoser settings, …). Secret
  values are reported as `set` / `unset`, never printed. Writes are always
  redacted; setting `CRASHLAB_REDACT=0` does not turn that off
- pricing-table coverage for models seen in recent runs (unpriced models are
  flagged so you can add `[tool.crashlab.prices]` or call
  `crashlab.pricing.set_price`)

`--fix` only applies safe repairs: it initializes a missing or empty store,
and runs SQLite `VACUUM` / `REINDEX` on an existing index. It never deletes
recorded runs. The process exits `0` when there are no errors (warnings are
fine) and `1` when a check failed.

```python
from crashlab.doctor import run_doctor
from crashlab.pricing import set_price

report = run_doctor(store_dir=None, fix=False)
print(report.ok, report.summary)
for check in report.checks:
    print(check.name, check.status, check.message)
if report.warnings:
    set_price("my-model", input=0.15, output=0.60)
```

## `crashlab init`

```bash
crashlab init
crashlab init --ci github
crashlab init --ci none --no-checks
crashlab init ./my-agent --force
```

Scaffolds a project directory (the current directory by default):

- `.github/workflows/crashlab-eval.yml` when `--ci github` (the default): an
  offline `crashlab eval` workflow that writes JUnit XML and a Markdown
  summary. When `GITHUB_STEP_SUMMARY` is set, `crashlab eval` also appends
  that Markdown to the job summary
- `crashlab_checks.py` when `--checks` is on (the default): a starter
  `@crashlab.check` suite you can point `crashlab eval --checks` at
- `.gitignore` is created or updated so `.crashlab/` is ignored

Existing files are left alone unless you pass `--force`. `--ci none` skips the
workflow; `--no-checks` skips the starter suite.

```python
from crashlab.scaffold import init_project

result = init_project(".", ci="github", checks=True, force=False)
print(result.created, result.updated, result.skipped, result.notes)
```

## `crashlab profile`

```bash
crashlab profile last
crashlab profile last --compare-models gpt-4o-mini,gpt-5
crashlab profile last --json
```

Breaks one recorded run down by:

- LLM time vs tool-execution time vs other overhead, as a percentage of the
  run's total duration (wall time when known, otherwise the sum of step
  durations)
- tokens in / out and estimated cost per step (the same list-price figures
  `crashlab show` uses)
- a prompt-redundancy / cache-opportunity estimate: the unchanged prompt
  prefix shared by consecutive LLM steps in the same run, converted into
  redundant re-sent tokens
- the slowest and costliest steps, and a tool leaderboard (calls and latency)
- what-if model-swap costs for `--compare-models` (comma-separated). When
  omitted, a small set of known cheap / mid / flagship models is used
  (`gpt-4o-mini`, `gpt-4o`, `gpt-5`, `claude-haiku-4-5`, `claude-sonnet-4-5`)

```python
from crashlab.profiler import profile_run
from crashlab.store import load

run = load("last")
profile = profile_run(run)
print(profile.llm_pct, profile.tool_pct, profile.other_pct)
print(profile.cache.estimated_redundant_tokens)
profile = profile_run(run, compare_models=["gpt-4o-mini", "gpt-5"])
for swap in profile.what_if:
    print(swap.model, swap.estimated_cost_usd)
```
