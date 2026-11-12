# Debugging a store full of runs

Tools for working through many recorded runs: find the ones you care about, group the
failures, and jump straight to the step that went wrong. Everything here is offline and
deterministic: it reads the local store (`.crashlab/` or `$CRASHLAB_DIR`) and calls no
model.

```bash
crashlab triage                         # failures grouped, diagnosed offline, ranked
crashlab cluster                        # which failures keep happening?
crashlab show 01M7... --failing         # the steps around the first failure
crashlab show 01M7... --step 3          # everything recorded for that step
crashlab search "timed out" --since 1d  # every run where it happened today
crashlab report 01M7... --html r.html   # share it, with a timing / cost waterfall
crashlab watch --failed                 # print failures as new runs finish
```

## `crashlab search`

```bash
crashlab search "502 Bad Gateway"              # text anywhere in the run
crashlab search refund --in tool --in error    # only tool args / results and errors
crashlab search --tool issue_refund --errors   # filters alone: list matching runs
crashlab search --check refund_within_policy --since 7d
crashlab search --label BAD_TOOL_ARGUMENTS --model gpt-4o
crashlab search 'order \d{5}' -E --json        # regular expression, JSON output
```

The query is a case-insensitive substring (`-s` for case-sensitive, `-E` for a Python
regular expression). It looks in:

| `--in`      | what                                                                  |
|-------------|-----------------------------------------------------------------------|
| `run`       | run id, agent name, tags, input, output, outcome reason               |
| `error`     | the run's exception and every step error (type and message)           |
| `message`   | chat messages: system prompt, user / assistant / tool turns, replies  |
| `tool`      | tool names, arguments and results                                     |
| `model`     | requested and responded model names                                   |
| `check`     | `@crashlab.check` names, status and messages                          |
| `finding`   | runtime-check findings (kind, severity, message)                      |
| `diagnosis` | the stored diagnosis's labels and root cause                          |

Each hit shows the step it came from and a short snippet. Chat requests repeat the whole
conversation on every call, so a message is reported once per run, at the step that
first sent it; a tool result echoed back to the model is reported once, at the tool step.

Filters select runs and are combined with AND:

| option | keeps runs where |
|---|---|
| `--tool NAME` | a tool with exactly this name ran |
| `--model TEXT` | a model name contains TEXT (case-insensitive) |
| `--check NAME` | this check did not pass (status `fail` or `error`) |
| `--label LABEL` | the diagnosis has this primary or secondary label |
| `--errors` | the run raised, ended with status `error`, or any step recorded an error |
| `--status`, `--outcome`, `--name` | exact run status, outcome, agent name |
| `--since 30m / 2h / 7d / 1w / 2026-11-12` | the run started at or after then (dates without a timezone are local time) |
| `--run PREFIX` | the run id starts with PREFIX |

Runs are scanned newest first and the scan stops after `--limit` matching runs (default
50; `--limit 0` for all); the footer says when it stopped early. Like `grep`, the command
exits 1 when nothing matched (also with `--json`), 2 on a bad option or regex.

In Python:

```python
from crashlab.search import search_runs

res = search_runs("timed out", tool="get_order", since=None, limit=None)
for m in res.matches:
    print(m.run_id, [(h.field, h.step, h.snippet) for h in m.hits])
```

**Limits.** Search reads each candidate run's cassette (the index only narrows by name,
run id prefix and start time first), so a scan of thousands of long runs takes as long
as loading them. It searches the redacted recording: values crashlab redacted cannot be
found. Raw provider payloads (`raw_request` / `raw_response`) are not searched; their
normalized form is. A run whose files cannot be read is skipped and named in the output.

## `crashlab cluster`

```bash
crashlab cluster                          # all failed runs, largest groups first
crashlab cluster --since 1d --name refund-bot
crashlab cluster --by name,checks         # group more loosely: same agent + failing checks
crashlab cluster --flagged                # also passing runs with error/critical findings
crashlab cluster --json --min-size 2
```

A run counts as failed when it raised (or ended with status `error`), its outcome is
`fail`, or a `@crashlab.check` did not pass. Each failed run gets a signature built only
from recorded facts:

| component  | value |
|------------|-------|
| `name`     | the agent name: runs of different agents never share a cluster |
| `error`    | the run's exception type, HTTP status if recorded, and normalized message |
| `step`     | the first step with an error: operation, name, exception and normalized message (not its index, which shifts between runs); without a step error, the first error/critical finding's step and kind |
| `checks`   | the names of the checks that did not pass |
| `label`    | the stored diagnosis's primary label, if the run was diagnosed |
| `findings` | kinds of error/critical runtime findings |

Two runs share a cluster only when every selected component is equal (`--by` picks the
components; the default is all of them). Normalization replaces timestamps, UUIDs,
crashlab run ids, hex strings of 8+ characters, e-mail addresses, ids with 4+ digits
(`call_abc1234`), decimals and numbers with 4+ digits, and quoted strings over 40
characters. It deliberately keeps words, numbers up to 3 digits (HTTP status codes,
retry counts) and short quoted values (`KeyError('sku')`), so `order A17` and `order C3`
end up in different clusters: the default splits rather than merges. Use `--by` to merge
more.

Each cluster lists its size, a stable id (a hash of the signature: the same failure keeps
the same id as new runs arrive), the newest run as the representative, and a few run
ids. `crashlab show <representative> --failing` is the next step.

**Limits.** Clustering does not read meaning: two different bugs that raise the same
exception with the same message at the same tool share a cluster, and one bug that
produces differently worded errors is split. A diagnosis label only takes part for runs
you diagnosed, so a diagnosed and an undiagnosed run of the same failure land in
different clusters by default (`--by name,error,step,checks` leaves the label out).

## `crashlab triage`

One command for a store full of failures: `crashlab cluster`, then a diagnosis of each
cluster's newest run, ranked largest cluster first.

```bash
crashlab triage                           # top 10 clusters
crashlab triage --since 1d --name refund-bot --limit 0   # all clusters of today's runs
crashlab triage --rediagnose              # replace stored diagnoses with new heuristic ones
crashlab triage --no-save --json          # read-only, machine-readable
```

```
┏━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ # ┃ runs ┃ failure                             ┃ diagnosis                                ┃
┡━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 2 │    1 │ hs-orchestrator                     │ EXTERNAL_FAILURE (heuristic, new)        │
│   │      │ first failure: execute_tool research│ step 4 (chat qwen/qwen3.8-27b) @research │
│   │      │ → openai.RateLimitError [429] ...   │ Heuristic: the LLM call failed: ...      │
│   │      │ latest 01M78TDG4H · 11-13 02:55     │ step 4 error: openai.RateLimitError ...  │
│   │      │                                     │ crashlab show 01M78TDG4H --failing       │
│   │      │                                     │ crashlab fix 01M78TDG4H                  │
│   │      │                                     │ crashlab export-test 01M78TDG4H          │
└───┴──────┴─────────────────────────────────────┴──────────────────────────────────────────┘
```

For each cluster: its size and signature, the diagnosis label, the step it blames (and
that step's agent in a multi-agent run), the root cause, one line of evidence (`step N
field: quote`), and the next commands: `show --failing` to read the steps, `fix` to get
proposed patches (an LLM call), `export-test` to turn the run into a regression test.

- **Offline.** New diagnoses are `crashlab diagnose --heuristic`: rules over checks,
  findings and errors, no LLM and no keys. They are labelled `heuristic`; a diagnosis
  already stored with the run (e.g. from `crashlab diagnose` with a model) is reused and
  labelled with its model and `stored`. `--rediagnose` replaces stored diagnoses with new
  heuristic ones. For a second opinion on one cluster, run `crashlab diagnose RUN` with a
  model.
- **Only representatives are diagnosed** (the top `--limit` clusters, default 10), and
  the diagnosis is saved with that run (`--no-save` keeps the store untouched).
- **Signature.** Same as `crashlab cluster` but without `label` by default, so a
  diagnosis saved by triage does not split its cluster next time; `--by` overrides.
- In Python: `from crashlab.triage import triage_runs; triage_runs(since=..., name=...)`
  returns ranked `TriageItem`s (`label`, `failing_step`, `agent`, `evidence`,
  `diagnoser`, `diagnosis_source`, `commands`, ...).

**Limits.** Heuristic labels are rule-based guesses: right on the bundled diagnose
benchmark, but they read checks, findings and errors, not intent. A cluster's diagnosis
is its newest run's; other runs in it may have failed for a different reason the
signature does not see.

## `crashlab show` filters

```bash
crashlab show last --errors        # only steps that recorded an error
crashlab show last --tools         # only tool steps (--llm: only LLM calls; both: union)
crashlab show last --failing       # 2 steps either side of the first failing step
crashlab show last --step 3        # full request / response, error with traceback,
                                   #   model, tokens, cost, findings on that step
crashlab show last --step 3 --json # that step as JSON
crashlab show last --tools --json  # the run as JSON with only the selected steps
```

A failed run's `show` ends with a line pointing at the failure, for example:

```
→ first error: step 3 (tool issue_refund, called by chat step 2) · checks failed: refund_within_policy · diagnosis: step 2 BAD_TOOL_ARGUMENTS
  open it: crashlab show 01M7ABCDEFGHJKMNPQRSTVWXYZ --step 3
```

The line names whichever of these exist: the first failing step, the run's exception
type (`raised TimeoutError`), the checks that did not pass and the diagnosis. "First
failing step" means the first step that recorded an error; if none did, the first step
an error/critical runtime finding fired on (`first budget finding: step 2 ...`); `--failing` falls back to the
diagnosis's step when neither exists. A passing run's output is unchanged.

## `crashlab watch`

```bash
crashlab watch                 # in a second terminal while you run your agent
crashlab watch --failed        # only finished runs that failed
crashlab watch --existing --json --timeout 60   # JSON lines, stop after 60 s
```

Prints a line when a run starts (status `running` in the store) and when it finishes,
with status, outcome, steps and cost; a failed run gets the same first-failure pointer as
`crashlab show`. Runs already in the store are skipped unless `--existing`. Ctrl-C stops
it (exit 0).

**Limits.** It polls the store index every `--interval` seconds (default 1); there is no
file-system watching. Each poll looks at the newest 200 runs, so if more than 200 runs
start within one interval some are not printed. A run that starts and finishes between
two polls is printed once, as finished. `--timeout` is approximate (polls x interval).

## Timing and cost in the HTML report

`crashlab report RUN --html out.html` now has a *Step timing and cost* section: one row
per step with its start offset, duration, tokens and estimated cost, and a waterfall bar
(overlapping bars are concurrent calls), plus the total LLM time, tool time, slowest step
and cost. It only uses what was recorded: durations are wall-clock times measured around
each call (network included); a chat step with no known price shows `?` and is counted
separately instead of being guessed; a run with no recorded durations gets a sentence
saying so instead of a chart. Steps served from a cassette in a replay or fork show the
time of that run, not of the original call, and the report says so.
