# Golden datasets and adversarial fuzzing

Turn recorded runs into a portable JSONL dataset of golden test cases, and
probe an agent with a seeded library of prompt injections, poisoned tool
results and edge-case inputs. Nothing here calls a model unless you pass a
live entry point (or the recorded run has one that can be imported).

## `crashlab dataset`

```bash
crashlab dataset build --name refund-bot --outcome pass --out golden.jsonl
crashlab dataset build --tag golden --since 7d --limit 50
crashlab dataset build last --out one-case.jsonl --no-redact
crashlab dataset inspect golden.jsonl
crashlab dataset inspect golden.jsonl --json
```

With no run ids, `dataset build` scans the store (`$CRASHLAB_DIR` or
`./.crashlab`) and keeps runs matching `--name`, `--tag`, `--outcome` and
`--since`. Each line of the JSONL file is one test case:

```json
{
  "id": "01JF...",
  "name": "refund-bot",
  "input": {"user": "Refund order A17"},
  "system_prompt": "You are a refund assistant...",
  "expected_tools": ["get_order", "refund"],
  "expected_outcome": "pass",
  "reference_reply": "Refunded $49.00 to order A17.",
  "tags": ["golden"],
  "metadata": {"run_id": "01JF...", "entrypoint": "refund_bot.agent:run_agent"}
}
```

`--redact` (the default) runs the same store redactor that strips API keys and
secrets, then scrubs email addresses, so a dataset is safe to share. Pass
`--no-redact` only for a private local copy. `--no-tools` leaves
`expected_tools` empty.

`crashlab dataset inspect PATH` counts cases, agents, tags, tool coverage and
malformed lines (a line that is not JSON, is not an object, or has no `id`).

The Python API writes the same file:

```python
from crashlab.dataset import build_dataset, inspect_dataset

report = build_dataset(runs, out_path="golden.jsonl", redact=True, include_tools=True)
summary = inspect_dataset("golden.jsonl")
print(summary.cases, summary.malformed, summary.tool_coverage)
```

`build_dataset` also accepts a single run, a run id (with `store=`), or an
iterable of either. When `out_path` is omitted the cases stay on
`report.records`.

## `crashlab fuzz`

Built-in mutation libraries are deterministic for a given `--seed` (default
`0`):

| library | what it does |
|---|---|
| `prompt_injection` | classic override, exfiltration and delimiter-escape payloads, appended or substituted into the user input |
| `tool_poison` | adversarial tool returns: injection in the payload, oversized strings, unicode / control / null bytes, empty / null, malformed JSON, extreme numbers |
| `edge_inputs` | empty input, a huge prompt, special characters, JSON / SQL / shell metacharacters |

```bash
crashlab fuzz last --mutation tool_poison --max-cases 10 --seed 0 --json
crashlab fuzz last --mutation prompt_injection --mutation edge_inputs --fail-on-finding
crashlab fuzz last --agent mypkg.agent:run_agent --checks tests/oracles.py --at-step 0
crashlab fuzz mypkg.agent:run_agent --mutation prompt_injection --max-cases 8 --seed 1
```

`crashlab fuzz RUN` forks the recorded run from step 0 (or `--at-step`) and
applies `ToolResultPatch` / `InjectMessagePatch` (and `crashlab.faults` for
poisoned tool output) so the agent sees the mutated prompt or tool result.
The same requirements as `crashlab fork` apply: the entry point must be
importable, or you pass `--agent module:fn`. Cases that cannot run are
reported as `needs_live_agent` or `skipped`, not as crashes.

`crashlab fuzz module:fn` calls that function directly on mutated inputs
(`fuzz_agent`). `--fail-on-finding` exits `1` if any case crashed, failed a
`@crashlab.check` oracle, or raised a runtime finding (especially
`system_prompt_leak`, `bad_tool_args`, `suspicious_tool_result`, `loop`).

```python
from crashlab.fuzz import fuzz_agent, fuzz_run

report = fuzz_agent(
    run_agent,
    inputs=["Refund order A17"],
    mutations=("prompt_injection", "edge_inputs"),
    max_cases=10,
    seed=0,
)
branch = fuzz_run(run, mutations=("tool_poison",), max_cases=10, seed=0)
print(report.survived, branch.survived, branch.findings_triggered)
```

Each result has an `outcome` of `survived`, `crashed`, `failed_checks`,
`finding`, `skipped` or `needs_live_agent`. `report.survived` is true only
when every executed case recovered; skipped and `needs_live_agent` cases are
ignored, matching `crashlab stress`.
