# Declarative policy rules

Write policy as a JSON, TOML or YAML file instead of (or as well as) a
`@crashlab.check` function. The same file grades a run through `crashlab check`,
`crashlab eval`, `crashlab rules test`, and the Python API
(`crashlab.assertions`). Nothing here calls a model.

There are no new runtime dependencies. TOML uses the standard library on
Python 3.11+ (`tomli` on 3.10, already a crashlab dependency). YAML uses
PyYAML when it is installed (`pip install 'crashlab[yaml]'`, already in the
`yaml` extra for patch files); a small built-in parser still reads simple
rule lists when PyYAML is not installed.

## A rule file

```yaml
# .crashlab/rules.yaml
rules:
  - type: must_call
    tools: [get_order, issue_refund]
    name: refund-bot
  - type: must_not_call
    tools: [delete_user]
  - type: tool_order
    tools: [get_order, issue_refund]
  - type: max_tool_calls
    max: 8
  - type: max_cost_usd
    max: 0.05
  - type: max_tokens
    max: 20000
  - type: max_duration_ms
    max: 30000
  - type: max_steps
    max: 20
  - type: reply_contains
    text: "refund"
  - type: reply_not_matches
    pattern: "(?i)i have deleted"
  - type: tool_arg_matches
    tool: issue_refund
    constraint: "amount <= 500"
  - type: json_valid_reply
    required_keys: [status]
    properties:
      status: string
      amount: number
  - type: no_findings_above
    threshold: error
```

The same rules as JSON (`{ "rules": [ ... ] }`) or a bare list. TOML:

```toml
[[rules]]
type = "must_call"
tools = ["get_order", "issue_refund"]
name = "refund-bot"

[[rules]]
type = "tool_arg_matches"
tool = "issue_refund"
constraint = "amount <= 500"

[[rules]]
type = "max_cost_usd"
max = 0.05
```

Or in `pyproject.toml`:

```toml
[tool.crashlab.rules]
rules = [
  { type = "must_call", tools = ["get_order"] },
  { type = "max_cost_usd", max = 0.05 },
]
```

`[[tool.crashlab.rules]]` arrays of tables are also accepted.

## Discovery

When `--rules` is omitted or set to `auto`, crashlab looks at or above the
current directory, in this order, and uses the first hit:

- `.crashlab/rules.json`
- `.crashlab/rules.toml`
- `.crashlab/rules.yaml` / `.crashlab/rules.yml`
- `pyproject.toml` if it has a `[tool.crashlab.rules]` section

Pass `--rules off` (or `load_rules("off")`) to skip discovery. A missing
explicit path is an error.

## Common fields

Every rule accepts:

- `type` (required)
- `id` — used as the check name (default: `type:tool` or `type#N`)
- `severity` — `info`, `warn`, `error` or `critical` (informational; a failing
  rule still fails the check)
- `message` — custom failure message
- `name` — only apply when `run.name` matches (the agent / run name)
- `agent` — only inspect steps labelled with that agent (multi-agent runs;
  `"researcher#2"` for one concurrent instance)

## Rule types

### `must_call` / `must_not_call`

`tools` (or `tool`) names a tool or a list. `must_call` requires each tool to
be called at least `min_calls` times (default 1). `must_not_call` fails if any
listed tool was called.

### `tool_order`

An ordered list `[a, b, ...]`: the first call of `a` must appear before the
first call of `b`. Tools that were never called fail the rule.

### `max_tool_calls`

A total cap (`max: 8`) and / or a per-tool cap:

```yaml
- type: max_tool_calls
  max: 10
  tools:
    search: 3
    browse: 2
```

`{type: max_tool_calls, tool: search, max: 3}` caps one tool.

### `max_cost_usd`, `max_tokens`, `max_duration_ms`, `max_steps`

Cost and tokens come from `crashlab.budgets.summarize_usage` (the same numbers
as the budget oracles). `max_duration_ms` is the run's wall time in
milliseconds. `max_steps` is the number of recorded steps. With `agent:`, cost
and tokens are scoped to that agent.

### `reply_matches` / `reply_not_matches`

A regex (`pattern` or `regex`) on the final assistant reply (`run.output` when
it is a string, otherwise the last chat text with no tool call). Optional
`ignore_case: true` or `flags: IGNORECASE`.

### `reply_contains` / `reply_not_contains`

Substring match. Case-insensitive by default; set `ignore_case: false` for a
case-sensitive check. The haystack is `text` or `contains`.

### `tool_arg_matches` / `tool_arg_forbidden`

Check arguments of one or more tools. `tool_arg_matches` fails if the tool was
not called, or if any call breaks the constraint. `tool_arg_forbidden` fails
when a call *satisfies* the constraint (a banned pattern or bound).

```yaml
- type: tool_arg_matches
  tool: issue_refund
  constraint: "amount <= 500"
- type: tool_arg_matches
  tool: issue_refund
  arg: amount
  max: 500
- type: tool_arg_matches
  tool: search
  args:
    query: {pattern: "^[A-Za-z0-9 -]+$"}
    limit: {min: 1, max: 20}
- type: tool_arg_forbidden
  tool: issue_refund
  arg: note
  pattern: "(?i)ignore previous"
```

Constraints can also use `op` / `value` (`op: "<="`, `value: 500`), `equals`,
`contains`, `in`, or `exists`.

### `json_valid_reply`

The final reply must parse as JSON (a fenced ` ```json ` block or a leading
`{...}` / `[...]` slice is accepted). If `run.output` is already a dict or
list, that value is used. Optional `required_keys` and a lightweight
`properties` type check (`string`, `number`, `integer`, `boolean`, `object`,
`array`, `null`) — no external JSON Schema library.

### `no_findings_above`

Fail if a runtime finding is at or above `threshold` (`warn`, `error` or
`critical`). Use this to turn stored detectors into a policy gate, the same
idea as `crashlab eval --min-finding-severity`.

## CLI

```bash
crashlab rules validate .crashlab/rules.yaml
crashlab rules validate .crashlab/rules.yaml --json
crashlab rules test .crashlab/rules.yaml
crashlab rules test .crashlab/rules.yaml last --json
crashlab rules test .crashlab/rules.yaml --all --name refund-bot
crashlab check last --rules .crashlab/rules.yaml
crashlab check --all --rules auto
crashlab eval --rules auto --name refund-bot
crashlab eval --rules rules.toml --json
crashlab eval --rules off
```

`crashlab rules validate` checks syntax and the rule schema (exit 1 on
error). `crashlab rules test` grades one or more stored runs (default: `last`;
`--all` / `--name` to scan the store) and exits 1 if any rule fails. `--json`
prints a machine-readable report.

`crashlab check --rules` and `crashlab eval --rules` attach the compiled rules
as extra outcome oracles. `--rules auto` is the default (discover, or no-op
when nothing is found). `--rules off` skips them.

## Python

```python
from crashlab.assertions import (
    evaluate_rules,
    load_rules,
    register_rules,
    rules_to_checks,
)
from crashlab.suite import eval_runs
from crashlab.verify.oracles import apply_outcome, run_checks

rules = load_rules(".crashlab/rules.yaml")
results = evaluate_rules(run, rules)
apply_outcome(run, results)

specs = rules_to_checks(rules)
results = run_checks(run, specs)

register_rules(rules)  # so crashlab.check / crashlab.verify see them

report = eval_runs(rules=".crashlab/rules.yaml")
```

`load_rules()` / `load_rules("auto")` auto-discovers a file.
`load_rules("off")` returns no rules. `parse_rules(data)` validates an
in-memory list or mapping. Each wrapper is a
`crashlab.verify.oracles.CheckSpec` whose function returns `True` or a
failure message, the same convention as `@crashlab.check`.
