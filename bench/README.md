# Diagnosis eval

`diagnose_eval.py` scores `crashlab.diagnose` on `diagnose_cases.json`: 31 hand-labelled
runs of `examples/refund_bot` recorded live on Groq gpt-oss-120b, Gemini, and local
qwen2.5:3b / llama3.2:3b (Ollama), with and without injected faults. The runs are in
`runs/` and `../examples/refund_bot/demo/` (secrets and local paths scrubbed).

```bash
python bench/diagnose_eval.py --heuristic                        # rule-based baseline
python bench/diagnose_eval.py --model openai/gpt-oss-120b \
  --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY --sleep 3
python bench/diagnose_eval.py --model gemini-3.7-flash --sleep 7 \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --api-key-env GEMINI_API_KEY
```

Each run is imported into a scratch store, re-checked with the current runtime checks and
oracles, and diagnosed with its fault switches, model and provider hidden from the
diagnoser. A case lists the accepted labels (e.g. a 502 retry loop may be
`STEP_REPETITION` or `TOOL_ERROR_MISHANDLED`), the expected culprit step and a window of
steps a reasonable answer may point at. Culprit step: the tool step that executed the
wrong action; if no tool ran, the chat step whose reply was wrong; for an external failure
the step where it shows.

The 31 cases: 21 failures (6 `WRONG_TOOL_SELECTION`: replied or claimed a refund without
calling `issue_refund`; 1 `BAD_TOOL_ARGUMENTS`; 1 `TOOL_ERROR_MISHANDLED`; 4
`STEP_REPETITION`; 3 `TASK_SPEC_VIOLATION`; 6 `EXTERNAL_FAILURE`), 8 clean runs and 2
schema-drift runs that still refunded correctly.

## Results

Rows are prompt diagnose-v2 (2026-10-19) unless marked v3. Prompt v3 (2026-10-21) adds the two rules below, also written after reading the v2 misses on this set, so its gain is partly fitted too. The 2026-10-21 heuristic rules
(HTML / error tool results count as external failures, a failed check that names no step
no longer outranks a step_repetition finding) were written after looking at the Day 6
misses on this same set, so its jump from 51.6% to 80.6% is partly fitted to these cases.


| diagnoser | cases | label acc. | label acc. (failures) | step exact | step in window | false alarms (8 clean) | evidence quotes valid | cost |
|---|---|---|---|---|---|---|---|---|
| heuristic (no LLM), 2026-10-21 rules | 31 | 80.6% | 71.4% | 66.7% | 76.2% | 0 | see results | $0 |
| heuristic (no LLM), Day 6 rules | 31 | 51.6% | 28.6% | 28.6% | 38.1% | 0 | 9/9 | $0 |
| Groq openai/gpt-oss-120b, prompt v3 (2026-10-21) | 31 | 90.3% | 85.7% | 71.4% | 90.5% | 0 | 44/46 | $0.017 total, 10.7 s/case |
| Groq openai/gpt-oss-120b, prompt v2 | 31 | 74.2% | 61.9% | 52.4% | 66.7% | 0 | 46/47 | $0.016 total, 6.3 s/case |
| gemini-flash-lite-latest, prompt v3 (2026-10-22) | 31 | 87.1% | 81.0% | 76.2% | 90.5% | 0 | 25/27 | free tier ($0.025 at list price), 1.0 s/case |
| gemini-3.7-flash (partial) | 8 | 87.5% | 87.5% | 87.5% | 87.5% | n/a | 9/10 | free tier |
| gemini-flash-latest (partial) | 7 | 85.7% | 85.7% | 85.7% | 85.7% | n/a | 7/8 | free tier |

The Gemini rows are the first 8 / 7 cases only: the free tier allows 20 requests per model
per day and ran out. On those same first 8 cases gpt-oss-120b scored 7/8 labels and 6/8 exact
steps, the heuristic 1/8 and 1/8. Small sample: read the Gemini rows as "works", not as a
ranking.

What the models miss: runs where get_order returned a 502 page and the agent apologised
are labelled `EXTERNAL_FAILURE` in the set; gpt-oss-120b mostly calls them `NO_FAILURE`
(4 of 6). A lookup loop that also blew the step budget gets `TASK_SPEC_VIOLATION` at the last
step (the failed budget check) instead of `STEP_REPETITION` at the first repeat. The run
whose whole answer was `{}` comes back `UNKNOWN` from every model.

The heuristic only fires on failed checks and error-level findings, so on passing runs
with warn-level findings (502 pages, bad arguments) it says `NO_FAILURE`.

History: with prompt v1 the same model scored 51.6% because quotes such as
`reply: I have issued a refund` carried the trace's display labels and failed the verbatim
check (25/64 quotes valid, 9 answers downgraded to `UNKNOWN`). Matching now ignores those
labels; re-validating the v1 answers gives 77.4%, so v1 vs v2 is within run-to-run noise.

`results/*.json` has every case: label, step, root cause, evidence and validation notes.

## Seed fault-injection benchmark (`crashlab bench`, Day 10)

Pre-registered before the first live run (this section was committed first; fault labels
and culprit rules live in `src/crashlab/bench/catalog.py` and each agent's
`ground_truth()`).

**Tasks and split.** `refund-bot` is the **dev** agent: the heuristics and the diagnosis
prompt were developed on its runs, so its numbers are optimistic. `sql-analyst`
(`examples/sql_analyst`, SQLite text-to-SQL, written for Day 10) is the held-out **test**
agent: nothing was tuned on it. Report the test split as the headline.

**Faults** (same four switches on both agents):

| fault | injection | accepted labels | culprit step | window |
|---|---|---|---|---|
| `spec_drop` | a rule is removed from the system prompt (30-day window / revenue = paid orders) | TASK_SPEC_VIOLATION | refund: the `issue_refund` call; sql: the first query without a `paid` filter (else the last query) | first step → culprit (refund) / → final answer (sql) |
| `tool_5xx` | the data tool returns an HTML 502 page on every call | EXTERNAL_FAILURE, TOOL_ERROR_MISHANDLED, STEP_REPETITION | the first 502 result | that step and the next |
| `schema_drift` | `total` renamed `amount_total` (refund: in the result; sql: in the DB, while the prompt still says `orders.total`) | TOOL_ERROR_MISHANDLED, BAD_TOOL_ARGUMENTS | refund: `issue_refund`; sql: the first failing query | from the fault to the culprit / final answer |
| `misleading_desc` | a tool description is wrong (refund: "refund total × 0.9"; sql: "money is in cents") | BAD_TOOL_ARGUMENTS, TASK_SPEC_VIOLATION | refund: `issue_refund`; sql: the first query dividing by 100 (else the last query) | first lookup/query → culprit / final answer |

**Episode loop.** For each model, task × condition (clean + 4 faults) × n episodes (n=3):
run the agent live (same model as agent and diagnoser), then oracles → heuristic
diagnosis and LLM diagnosis (fault switches, model and provider hidden) → if the run
failed and the LLM called it a failure, fix → `verify` (k=3 patched forks, 3 unpatched
controls, regression on up to 2 clean passing runs of the same task), once per task ×
fault cell to fit free-tier quotas.

**Metrics** (Wilson 95% intervals; small n, so read the intervals, not the points):
- *fault triggered*: faulted runs whose oracles failed (a fault the agent shrugged off is
  not scored for diagnosis);
- *detected*: triggered runs the diagnoser called a failure; *flagged by findings*: an
  error/critical runtime finding, no oracle needed;
- *label ok / step exact / step in window*: on triggered runs;
- *fix VERIFIED* (and VERIFIED-or-LIKELY): of the verified cells; regressions counted;
- *clean false alarms*: failure labels on clean runs that passed;
- cost (agent + diagnosis + fix + verify) and seconds per episode.

**Reproduce offline.** Every episode run is exported to `bench/runs/seed-<model>/`
(cassette + run header with its diagnosis and fix attempts). `crashlab bench --offline
--runs-dir bench/runs/seed-<model> --label <model>` rescores them with no keys and no
network (the heuristic baseline is recomputed; LLM diagnoses are read from the runs).

### Seed results

2026-10-22, n=3 episodes per cell, k=3.

Full tables: `bench/results/seed_*.md`. Each cell below is k/n, then the Wilson 95% CI.
**Headline = test split (`sql-analyst`, held out; nothing was tuned on it).**

| model (agent = diagnoser) | split | fault triggered | LLM label ok | LLM step in window | heuristic label ok | fix VERIFIED | clean false alarms | cost* |
|---|---|---|---|---|---|---|---|---|
| Gemini `gemini-flash-lite-latest` | **test** | 6/12 (25–75%) | **6/6 (61–100%)** | 6/6 (61–100%) | 2/6 (10–70%) | 0/3 (1/3 LIKELY) | 0/3 | $0.072 |
| Groq `openai/gpt-oss-120b` † | **test** | 11/11 (74–100%) | 3/11 (10–57%) | 5/11 (21–72%) | 0/11 (0–26%) | 0/2 | – (0 clean passes) | $0.025 |
| Gemini flash-lite | dev | 6/12 | 4/6 (30–90%) | 4/6 | 6/6 | 2/2 | 0/3 | $0.028 |
| Groq gpt-oss-120b † | dev | 8/12 | 8/8 (68–100%) | 7/8 | 8/8 | 1/2 (2/2 incl. LIKELY) | 0/3 | $0.020 |

\* List-price estimate. Both runs used free tiers, so nothing was actually billed.

**Caveats. Read these before you quote any number from the table.**
- **The sample is small.** Each cell has 3 episodes and each verify has k=3. Most of the
  intervals are 40 to 60 points wide, so the two models are not distinguishable on most metrics.
- † **The Groq test-split numbers are not valid.** On the first run, gpt-oss-120b on Groq
  ended **every** clean `sql-analyst` episode with an empty reply after a correct query, so
  the clean baseline failed 3/3. That means "triggered 11/11" mostly measures this agent bug,
  not the injected faults. The agent now asks once for an answer when it gets an empty reply
  (commit 87bed39). The rerun after that fix (`seed_groq-gpt-oss-120b-sqlv2`) got through only
  1 clean episode, which passed, before Groq's tokens-per-day quota ran out. A second
  rerun on 2026-10-22 (`seed_groq-gpt-oss-120b-sqlv3`) recorded 0 episodes: the quota had
  still not reset. **There are still no valid Groq test-split numbers.** The Groq run is
  also partial: 29/30 episodes, with the last `misleading_desc` episode missing.
- The heuristic baseline gets near-perfect labels on dev, where it was tuned, and 0–33% on test.
  That gap is the overfitting the split was designed to expose.
- Fixes verified on the dev agent (spec_drop, misleading_desc). On the test agent, the
  `tool_5xx` and `schema_drift`/`spec_drop` fixes were **REJECTED**: a prompt patch cannot
  repair a broken tool or database. That rejection is the correct verdict.
- Several faults did not trigger: `schema_drift` on refund-bot (0/3 on both models), and on
  Gemini, `tool_5xx` on refund-bot and `schema_drift` on sql-analyst (0/3). Those cells say
  nothing about diagnosis.
