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

The LLM rows below are prompt diagnose-v2 (2026-10-19). The 2026-10-21 heuristic rules
(HTML / error tool results count as external failures, a failed check that names no step
no longer outranks a step_repetition finding) were written after looking at the Day 6
misses on this same set, so its jump from 51.6% to 80.6% is partly fitted to these cases.


| diagnoser | cases | label acc. | label acc. (failures) | step exact | step in window | false alarms (8 clean) | evidence quotes valid | cost |
|---|---|---|---|---|---|---|---|---|
| heuristic (no LLM), 2026-10-21 rules | 31 | 80.6% | 71.4% | 66.7% | 76.2% | 0 | see results | $0 |
| heuristic (no LLM), Day 6 rules | 31 | 51.6% | 28.6% | 28.6% | 38.1% | 0 | 9/9 | $0 |
| Groq openai/gpt-oss-120b | 31 | 74.2% | 61.9% | 52.4% | 66.7% | 0 | 46/47 | $0.016 total, 6.3 s/case |
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
