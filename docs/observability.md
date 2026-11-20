# Observability and visual debugging

Export a recorded run as OpenTelemetry OTLP/JSON or a Chrome trace, summarise a
fleet of runs, print a terminal waterfall, or open a local web explorer. All of
this reads the store only; nothing calls a model.

## Export: OTLP and Chrome traces

```bash
crashlab export last --format otel -o run.otel.json
crashlab export last --format chrome-trace -o run.trace.json
crashlab export last --format jsonl
```

`jsonl` is the portable cassette (the default, unchanged). `otel` is an
[OTLP/JSON](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding)
document (`resourceSpans` → `scopeSpans` → `spans`) with OpenTelemetry GenAI
semantic conventions (`gen_ai.system`, `gen_ai.request.model`,
`gen_ai.usage.input_tokens`, `gen_ai.agent.name`, `gen_ai.tool.name`, …) plus
`crashlab.run_id`, `crashlab.outcome` and `crashlab.cost_usd`. Each run has a
root span; tool steps are parented to the chat call that requested them;
linked child runs (`crashlab.child_env()` / `record(parent=...)`) hang off the
parent run, or off a `parent_step` / `parent_span` in `run.env["link"]` when
one is set. Failing steps and runtime findings become span status errors and
events. `chrome-trace` is a Chrome `traceEvents` log (`displayTimeUnit: ms`);
open it in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`. Both
formats include linked child runs that are in the same store.

In Python:

```python
from crashlab.otel import to_chrome_trace, to_otlp
from crashlab.store import Store, load

run = load("last")
otlp = to_otlp(run, store=Store(), include_children=True)
trace = to_chrome_trace(run, store=Store(), include_children=True)
```

Cost on a span is the same list-price estimate `crashlab show` uses, not the
provider invoice. Step timestamps and durations are the wall-clock times
crashlab measured around each call.

## Fleet stats

```bash
crashlab stats
crashlab stats --since 7d --name refund-bot --limit 100
crashlab stats --since 7d --json
crashlab stats --html report.html
```

`crashlab stats` counts pass / fail / error / unknown rates, totals and
percentiles for cost, tokens and duration, then breaks the same numbers down
by agent and model, with a tool leaderboard (calls, errors, error rate, p95
duration), the most common runtime findings, failing `@crashlab.check` names,
and the slowest / most expensive runs. Missing cost or duration is skipped in
means and percentiles, not treated as zero; the JSON includes how many runs
had no figure. `--since` uses the same grammar as `crashlab search`
(`30m` / `2h` / `7d` / `1w` or an ISO date). `--limit 0` (the default) scans
every matching run. `--html PATH` writes a self-contained HTML report (inline
CSS, no scripts or external assets) with summary cards, agent and model
tables, findings, failing checks, and the slowest / costliest runs.

```python
from pathlib import Path

from crashlab.stats import compute_stats, render_fleet_html
from crashlab.store import Store

stats = compute_stats(Store(), since="7d", name="refund-bot", limit=100)
print(stats.n, stats.fail_rate, stats.total_cost_usd, stats.p95_duration_s)
for tool in stats.tools:
    print(tool.name, tool.calls, tool.error_rate, tool.p95_duration_s)
Path("report.html").write_text(render_fleet_html(stats), encoding="utf-8")
```

## Trajectory diffs

`crashlab diff A B` already compares two runs step by step. `--json` (and
`RunDiff.to_dict()`) also include the tool-call sequences, their Levenshtein
edit distance, and a normalized trajectory similarity
(`1 - distance / max(len(A), len(B))`, or `1.0` when both agents called no
tools). Existing `RunDiff` fields and `.summary()` are unchanged.

```bash
crashlab diff last prev --json
crashlab diff last prev --html compare.html
crashlab diff last prev --markdown compare.md
crashlab diff last prev --markdown - --fail-on-regression
```

`--html PATH` writes a self-contained, interactive side-by-side report (inline
CSS and vanilla JS, no external assets): an executive header (status, duration,
cost, tokens, findings, trajectory edit distance / similarity), regression and
improvement pills, and an aligned step table that highlights diverged steps,
new or resolved findings, tool-call differences, and per-step cost / latency
deltas. `--markdown PATH` writes a GitHub-flavored Markdown summary table ready
for a PR comment (`gh pr comment --body-file`); use `-` to print it to stdout.
`--fail-on-regression` exits `1` when Run B regressed versus Run A (status or
outcome became a failure, a new error/warn finding appeared, or a check flipped
from pass to fail). Existing flags (`--json`, `--full`) are unchanged.

```python
from pathlib import Path

import crashlab
from crashlab.diff_report import has_regression, render_diff_html, render_diff_markdown

d = crashlab.diff(crashlab.load("last"), crashlab.load("prev"))
print(d.summary(), d.tool_edit_distance, d.trajectory_similarity)
print(d.to_dict()["tool_alignment"])
Path("compare.html").write_text(render_diff_html(d, d.a, d.b), encoding="utf-8")
print(render_diff_markdown(d, d.a, d.b))
print(has_regression(d))
```

## CI regression baselines

Snapshot fleet metrics from a known-good set of runs, then fail CI if the
pass rate drops or p95 cost / latency regresses.

```bash
crashlab baseline save
crashlab baseline save --out .crashlab/baseline.json --since 7d --name refund-bot
crashlab baseline check
crashlab baseline check --fail-on-regression --json
crashlab baseline check .crashlab/baseline.json --min-pass-rate-delta -0.05 --max-cost-ratio 1.25 --max-latency-ratio 1.5
```

`crashlab baseline save` writes pass rate, error rate, p95 cost, p95
latency, per-agent pass rates and the tool distribution to JSON (default
`<store>/baseline.json`). `crashlab baseline check` compares the current
store against that snapshot. By default a regression is a pass-rate drop of
more than 5 points, p95 cost growing by more than 1.25×, or p95 latency
growing by more than 1.5×; the same pass-rate budget is also applied
per agent. `--fail-on-regression` exits `1` when any threshold is crossed
(otherwise the command prints the comparison and exits `0`). Missing cost
or latency on either side is skipped, not treated as zero.

```python
from crashlab.baseline import compare_baseline, save_baseline

save_baseline(".crashlab", ".crashlab/baseline.json", since="7d")
result = compare_baseline(
    ".crashlab/baseline.json",
    ".crashlab",
    min_pass_rate_delta=-0.05,
    max_cost_ratio=1.25,
    max_latency_ratio=1.5,
)
print(result.ok, result.summary())
```

## Terminal waterfall

```bash
crashlab timeline last
crashlab timeline last --no-children
crashlab timeline last --json
```

Each step is a row: index, agent, operation, a proportional timing bar,
duration, tokens, cost, and markers for errors, findings and handoffs. Linked
child runs are indented under the parent unless you pass `--no-children`.

```python
from crashlab.store import Store, load
from crashlab.timeline import render_timeline

run = load("last")
print(render_timeline(run, store=Store(), include_children=True, width=100))
```

Overlapping bars are concurrent calls. Durations of cassette-served,
patched, blocked or faked steps are the time of this run (a replay or fork),
not of the original live call.

## Local web explorer

```bash
crashlab ui --port 8765 --no-open
```

`crashlab ui` serves a self-contained HTML explorer on `127.0.0.1` only (no
extra packages, no external assets). The page lists and filters runs; opening
one shows the step waterfall, chat, tool arguments and results, findings,
diagnosis and child runs. Tick two runs and use Diff for a side-by-side
comparison. `--no-open` skips launching a browser. Stop with Ctrl-C.

The same process exposes a JSON API: `GET /api/runs`, `GET /api/runs/<id>`,
`GET /api/diff?a=...&b=...`, `GET /api/stats`, `GET /api/runs/<id>/otel`.
Unknown ids return 404. In tests, bind an ephemeral port with
`create_server(store, port=0)`:

```python
from crashlab.store import Store
from crashlab.ui import create_server

server = create_server(Store(), host="127.0.0.1", port=0)
# server.server_address[1] is the bound port
```

The viewer is for local debugging. It is not an authenticated multi-user
service; do not proxy it onto a public address.
