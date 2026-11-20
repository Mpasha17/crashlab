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
```

`crashlab stats` counts pass / fail / error / unknown rates, totals and
percentiles for cost, tokens and duration, then breaks the same numbers down
by agent and model, with a tool leaderboard (calls, errors, error rate, p95
duration), the most common runtime findings, failing `@crashlab.check` names,
and the slowest / most expensive runs. Missing cost or duration is skipped in
means and percentiles, not treated as zero; the JSON includes how many runs
had no figure. `--since` uses the same grammar as `crashlab search`
(`30m` / `2h` / `7d` / `1w` or an ISO date). `--limit 0` (the default) scans
every matching run.

```python
from crashlab.stats import compute_stats
from crashlab.store import Store

stats = compute_stats(Store(), since="7d", name="refund-bot", limit=100)
print(stats.n, stats.fail_rate, stats.total_cost_usd, stats.p95_duration_s)
for tool in stats.tools:
    print(tool.name, tool.calls, tool.error_rate, tool.p95_duration_s)
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
