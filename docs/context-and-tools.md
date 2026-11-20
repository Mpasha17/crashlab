# Prompt context and tool-schema audits

When a long agent run gets slow, expensive, or a provider like Groq or Gemini
rejects a tool call, the cause is often sitting in the prompt itself: a system
prompt that grew, tool results that were resent every turn, or a tool schema
with `required` keys that are not in `properties`. These commands read a
recorded run and do not call a model.

## `crashlab context`

```bash
crashlab context last
crashlab context last --step 2
crashlab context last --json
```

For every LLM chat step in the run (or just `--step N`), this breaks the prompt
into sections:

- `system_prompt`
- `tool_definitions`
- `user_messages`
- `assistant_messages`
- `tool_results`
- `other` (uncategorized roles or payloads)

Each section reports a character count and an estimated token share. The
estimate starts from `chars // 4` and, when the step recorded `input_tokens`,
is scaled so the sections sum to that provider count. The report also includes:

- turn-over-turn growth (`Δ` characters and tokens against the previous chat
  step)
- a per-tool output size leaderboard: which tool result is flooding the
  context, including results carried forward in later turns
- repeated or re-appended messages (the same body appearing twice in one
  step, or more times than the growing history would explain)

`--json` prints the structured report. `--step` must name a chat step; a
missing run or a non-chat step exits `2`.

```python
from crashlab.context_audit import analyze_context
from crashlab.store import load

run = load("last")
report = analyze_context(run)
for step in report.steps:
    print(step.idx, step.total_chars, step.estimated_tokens, step.delta_chars)
    for section in step.sections:
        print(section.name, section.chars, section.estimated_tokens, section.share_pct)
for item in report.tool_outputs:
    print(item.name, item.chars, item.occurrences)
report = analyze_context(run, step=2)
```

## `crashlab audit-tools`

```bash
crashlab audit-tools last
crashlab audit-tools last --json
crashlab audit-tools last --fail-on-error
```

Inspects every tool definition on the run's LLM steps (`request.tools`) and
compares it with the tools that were actually called (`execute_tool` steps and
chat tool calls). It flags real-world schema issues that break providers:

- `required` referencing keys that are missing from `properties`, or
  `required` present when `properties` is missing or empty (the Groq
  `transfer_to_refunds`-style rejection)
- a missing top-level `description`, or an empty / vague one
- parameters in `properties` missing `type` or `description`
- duplicate tool names in the same request, or high-similarity name /
  description pairs that confuse tool selection
- massive tool schemas (more than 2000 characters per tool) resent on every
  request
- tools defined on every LLM step but never called, versus tools that were
  called

`--fail-on-error` exits `1` when any issue has severity `error` (schema bugs
such as `required` without `properties`). Warnings and unused-tool notes do
not fail the command unless you add that flag. A missing run exits `2`.

```python
from crashlab.context_audit import audit_tool_schemas
from crashlab.store import load

run = load("last")
audit = audit_tool_schemas(run)
print(audit.ok, audit.unused, audit.called)
for issue in audit.issues:
    print(issue.severity, issue.kind, issue.tool, issue.message)
```
