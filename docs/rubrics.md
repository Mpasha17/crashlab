# Weighted rubric scoring

Grade a recorded run 0–100 across weighted criteria (correctness, safety,
tool efficiency, cost, latency) instead of a single pass/fail. Nothing here
calls a model. The same rubric works from Python (`crashlab.score_run`,
`crashlab.score_runs`) and from the CLI (`crashlab score`,
`crashlab eval --rubric`).

## Built-in default

`DEFAULT_RUBRIC` (also used when `crashlab score` is given no `--rubric`)
weights:

- outcome / correctness: 0.35
- safety / findings: 0.25
- tool efficiency: 0.15
- cost: 0.15
- latency: 0.10

The passing score is 70 (a C). Letter grades are A ≥ 90, B ≥ 80, C ≥ 70,
D ≥ 60, otherwise F. Cost and latency score 1.0 when no budget is set on the
criterion; add `target` / `max_value` (or `target_cost_usd` / `max_cost_usd`,
`target_ms` / `max_ms`) to turn those dimensions into a linear or threshold
gate.

```python
import crashlab
from crashlab import Rubric, RubricCriterion, score_run, score_runs
from crashlab.rubrics import DEFAULT_RUBRIC, load_rubric

run = crashlab.load("last")
result = score_run(run)
print(result.score, result.grade, result.passed)

report = score_runs([run], DEFAULT_RUBRIC)
print(report.mean, report.median, report.weakest)
```

## A rubric file

JSON, TOML or YAML. Each criterion has a `name`, a `kind`, and a `weight`.
Optional `target` and `max_value` are interpreted by the kind.

```yaml
# rubric.yaml
name: refund
min_score: 70
criteria:
  - name: outcome
    kind: outcome_pass
    weight: 0.35
  - name: safety
    kind: no_findings
    weight: 0.25
    target: error
    partial_credit: 0.5
  - name: tools
    kind: tool_efficiency
    weight: 0.15
    target: [get_order, issue_refund]
  - name: cost
    kind: cost_budget
    weight: 0.15
    target: 0.01
    max_value: 0.10
  - name: latency
    kind: latency_budget
    weight: 0.10
    target: 2000
    max_value: 30000
  - name: must_refund
    kind: rule
    weight: 0.20
    rule:
      type: must_call
      tools: [issue_refund]
```

The same rubric as TOML:

```toml
name = "refund"
min_score = 70.0

[[criteria]]
name = "outcome"
kind = "outcome_pass"
weight = 0.35

[[criteria]]
name = "cost"
kind = "cost_budget"
weight = 0.15
target = 0.01
max_value = 0.10
```

```python
from crashlab.rubrics import load_rubric, score_run

rubric = load_rubric("rubric.yaml")
result = score_run(run, rubric)
```

## Criterion kinds

### `outcome_pass`

1.0 if the run is not in an error status, the stored outcome is not `fail`,
and no `@crashlab.check` result failed; otherwise 0.0.

### `no_findings`

1.0 if the run has no findings at or above `target` severity (`warn` or
`error`, default `error`). Set `partial_credit: 0.5` (or `true`) to award
half credit when the only findings are warnings and the floor is `error`.

### `rule` / `rules`

Runs an assertion rule (the same dicts as `crashlab.assertions` /
`docs/rules.md`). 1.0 if the rule passes, otherwise 0.0. `rule` may be one
mapping or a list (every rule must pass).

### `cost_budget`

Linear or threshold score against a cost budget. At or below `target`
(`target_cost_usd`) is 1.0; at or above `max_value` (`max_cost_usd`) is 0.0;
in between, a linear interpolation. With only one bound, the score is a
threshold (1.0 at or under, 0.0 over). With neither bound, the score is 1.0.

### `latency_budget`

The same shape as `cost_budget`, against wall time in milliseconds
(`target` / `target_ms`, `max_value` / `max_ms`).

### `tool_efficiency`

When `target` (or `expected_tools`) lists tools that should be called, the
score is expected tools hit, divided by expected tools plus extra and
redundant calls. With no expected list, the score is unique tools over total
tool calls (1.0 when there are no calls or no repeats).

### `check`

A Python callable `fn(run) -> float | bool | CheckResult`. Only available
from the Python API (a file cannot name a function). A float in 0–1 is used
as-is; a float above 1 is treated as a 0–100 score; `True` / a passing
`CheckResult` is 1.0; anything else is 0.0.

```python
from crashlab import Rubric, RubricCriterion, score_run

rubric = Rubric(
    name="refund",
    min_score=70.0,
    criteria=[
        RubricCriterion(name="outcome", kind="outcome_pass", weight=0.35),
        RubricCriterion(name="safety", kind="no_findings", weight=0.25, target="error"),
        RubricCriterion(
            name="tools",
            kind="tool_efficiency",
            weight=0.15,
            target=["get_order", "issue_refund"],
        ),
        RubricCriterion(name="cost", kind="cost_budget", weight=0.15, target=0.01, max_value=0.10),
        RubricCriterion(
            name="latency", kind="latency_budget", weight=0.10, target=2000, max_value=30000
        ),
        RubricCriterion(
            name="custom",
            kind="check",
            weight=0.10,
            check=lambda run: run.outcome == "pass",
        ),
    ],
)
result = score_run(run, rubric)
```

## CLI

```bash
crashlab score last
crashlab score last --json
crashlab score --all --name refund-bot --min-score 70
crashlab score --all --rubric rubric.yaml --fail-under 70
crashlab eval --rubric rubric.yaml --min-score 70
crashlab eval --rubric rubric.yaml --min-score 70 --json
```

`crashlab score` defaults to the last stored run. `--all` scores every stored
run (optionally filtered with `--name`). `--min-score` overrides the rubric's
passing threshold (default 70). `--fail-under N` exits 1 if any run scores
below N, even if it still meets `--min-score`. The process exits 0 if every
run meets the passing score, 1 if any fall short, and 2 if there are no runs
or the rubric file is invalid.

`crashlab eval --rubric` is optional. Without it, `crashlab eval` is unchanged
(binary pass/fail only). With `--rubric` (and optional `--min-score`), each
run is also graded 0–100 and fails the suite if it scores below the
threshold.

Pass `checks=` to `score_run` to run `@crashlab.check` oracles first; the
outcome criterion then sees the updated result.
