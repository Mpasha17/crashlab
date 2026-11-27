# LLM-as-judge evaluators

Deterministic checks (`crashlab.rag`, rules, runtime findings) cannot tell whether
an answer is *faithful* to what the tools returned, whether it *answers the
question*, or whether the agent *did what the user wanted*. A judge asks a model
those questions about a **recorded** run and returns a score from 0 to 1 with
reasoning and verbatim evidence. Judges complement the deterministic checks; they
do not replace them.

Verdicts are cached on disk, so re-running `crashlab eval --judge ...` in CI is
free and offline once the cache is warm.

## Built-in judges

| judge | question it answers | default threshold |
| --- | --- | --- |
| `faithfulness` | Is every claim in the answer supported by the retrieved docs / tool results? | 0.8 |
| `answer_relevance` | Does the answer address the user's question? | 0.7 |
| `task_completion` | Did the agent achieve the user's goal (proved by tool results)? | 0.7 |
| `tool_choice` | Were the right tools used, with sensible arguments and no waste? | 0.7 |
| `safety` | Harmful content, PII leaks, obeyed injections, policy violations? | 0.9 |
| `custom(criteria=..., rubric=...)` | Whatever you describe | 0.7 |

A judge that has nothing to look at (`faithfulness` with no tool results,
`tool_choice` with no tool calls, anything with no answer) is reported as
`skipped` and passes, without calling the model.

## Configure the judge model

Any OpenAI-compatible endpoint (OpenAI, Gemini, Groq, OpenRouter, Ollama, vLLM
...), resolved like the diagnoser:

```bash
export CRASHLAB_JUDGE_MODEL=gpt-4.1-mini
export CRASHLAB_JUDGE_BASE_URL=https://api.openai.com/v1   # optional
export CRASHLAB_JUDGE_API_KEY=...                          # optional
```

If `CRASHLAB_JUDGE_*` is unset, `CRASHLAB_DIAGNOSER_MODEL` / `_BASE_URL` /
`_API_KEY` are used. With no model configured every entry point fails with a
clear error (exit code 2 on the CLI). The `openai` package is imported only when
a real model is called (`pip install 'crashlab[openai]'`).

The judge is told to treat the run data as untrusted, so an injected "score this
1.0" inside a tool result is judged, not obeyed. The run's question, answer, and
tool results are sent to the judge model: run `crashlab audit-secrets` /
`crashlab redact-run` first on sensitive runs.

## From Python

```python
import crashlab
from crashlab import judges

run = crashlab.load("last")

verdict = judges.faithfulness().evaluate(run)
print(verdict.score, verdict.passed, verdict.reasoning, verdict.evidence)

# several judges at once (default: every built-in)
for result in crashlab.judge_run(run, ["faithfulness", "task_completion"], threshold=0.75):
    print(result.name, result.score, result.passed, result.cached)

tone = judges.custom(
    criteria="The reply is polite and never blames the customer.",
    rubric="1.0 warm and polite; 0.5 curt; 0.0 rude or blaming",
    name="tone",
    threshold=0.6,
)
print(tone.evaluate(run, model="gpt-4.1-mini").to_dict())
```

`JudgeResult` has `name`, `score`, `passed`, `reasoning`, `evidence`, `model`,
`cached`, plus `threshold`, `run_id` and `skipped`.

For tests and offline use, pass `judge_fn`, a callable that receives the chat
messages and returns the model's reply (a JSON string or a dict). No network, no
SDK:

```python
import crashlab
from crashlab import judges

def fake_judge(messages):
    return {"score": 0.9, "reasoning": "answer matches the tool result", "evidence": []}

run = crashlab.load("last")
result = judges.answer_relevance().evaluate(run, judge_fn=fake_judge)
assert result.passed
```

The reply must be a JSON object with a numeric `score` in [0, 1], a non-empty
`reasoning` string and an optional `evidence` list of strings. An invalid reply
gets exactly one repair retry (the model is shown what was wrong); a second bad
reply raises `crashlab.JudgeError`.

## From the CLI

```bash
crashlab judge last
crashlab judge RUN_A RUN_B --judge faithfulness --judge safety --json
crashlab judge --all --name refund-bot --fail-under 0.6
crashlab judge last --threshold 0.5 --no-cache
crashlab eval --judge faithfulness --judge task_completion --judge-threshold 0.75
crashlab eval --rubric rubric.yaml --judge safety --no-cache
```

- `crashlab judge` defaults to the last run and all five built-ins. Exit 0 when
  every judge passes, 1 when any fails (or a score is under `--fail-under`), 2 for
  no runs, an unknown judge, no model, or a failed model call.
- `crashlab eval --judge NAME` (repeatable) adds judge verdicts to the normal
  grading: a run fails if any judge scores under its threshold, the reason names
  the judge and its reasoning, and the Markdown report gains a `judges` column.
  `--judge-threshold` overrides every judge's own threshold.
- `--no-cache` ignores and does not write the cache.

## In a rubric

A `judge` criterion contributes the judge's 0-1 score (not just pass/fail) to the
weighted 0-100 rubric grade. Name a built-in, or give `criteria` for a custom one:

```yaml
# rubric.yaml
name: support-bot
min_score: 75
criteria:
  - name: outcome
    kind: outcome_pass
    weight: 0.3
  - name: faithfulness
    kind: judge
    weight: 0.4
  - name: tone
    kind: judge
    criteria: The reply is polite and never blames the customer.
    judge_rubric: "1.0 warm; 0.5 curt; 0.0 rude"
    weight: 0.3
```

```bash
crashlab score last --rubric rubric.yaml
```

## The cache

Each verdict is stored as `<store>/judges/<judge>/<sha256>.json`, keyed by the
judge, the exact prompt (criteria, scoring guide, and the run content the judge
sees) and the model, plus a prompt version. Change the run, the criteria, or the
model and you get a new verdict; change only the threshold and you reuse the old
verdict and re-apply the threshold. Commit or cache `.crashlab/judges/` in CI to
keep `crashlab eval --judge` free and offline. A corrupt entry is treated as a
miss. Programmatic calls with `judge_fn` are not cached unless you pass
`cache=True`.

Judge calls cannot run inside an active `crashlab.record(...)` run (their LLM
call would be recorded into it): grade after the run has finished.
