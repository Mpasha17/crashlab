# Prompt versions and A/B comparison

"Which prompt change broke (or fixed) my agent?" Crashlab fingerprints the system
prompt of every recorded chat step, so each run can be tied to a prompt version, and
compares outcomes between versions with a significance test. Nothing here calls a model
unless you ask for a judge metric.

## Prompt ids

Every chat step gets a short id such as `p-3f9a1c`, stored on the step as
`crashlab.prompt.id` when it is recorded (one hash of the system prompt, so it is
cheap). The id is a hash of the *normalized* prompt, so only a real wording change makes a
new version. These do not change it:

- whitespace and line breaks;
- template placeholders: `{name}`, `{{ name }}`, `${name}`, `%(name)s`;
- ISO dates and timestamps (`2026-11-27`, `2026-11-27T10:00:00Z`) and UUIDs.

Runs recorded before this feature have no stored id. Crashlab computes it from the stored
request the first time you look, with the same result, so old runs are included in every
command below.

```python
import crashlab

crashlab.prompt_id("You are a   refund bot for {customer}.")  # 'p-...'
```

## Name a version

Ids tell versions apart; labels name them. Use whichever is easier to type, both work
everywhere a version is expected.

```python
import crashlab

with crashlab.prompt_version("support-v3"):
    ...  # every chat step in the block carries the label

with crashlab.record("support-bot", prompt_version="support-v3"):
    ...  # or label the whole run
```

`crashlab.prompt_version(label, templates={"planner": PLANNER_TEMPLATE})` also
fingerprints named prompt templates and stores `{name: id}` on each step as
`crashlab.prompt.templates`. Labels never change recording, replay or request hashes.

## Commands

```bash
crashlab prompts list --since 7d
crashlab prompts list --agent support-bot --json
crashlab prompts show p-3f9a1c
crashlab prompts diff support-v2 support-v3
crashlab prompts ab support-v2 support-v3
crashlab prompts ab p-3f9a1c p-81be07 --metric cost
crashlab prompts ab support-v2 support-v3 --metric judge:task_completion --fail-on-regression
```

- `list`: each version with first and last seen, runs, pass rate, error and finding rates,
  average cost, latency and tokens. `--since` takes `2h`, `7d` or an ISO date; `--agent`
  keeps one agent (a run name or a step's agent).
- `show ID`: the full prompt text (as first recorded, secrets already redacted) and the
  summary. `ID` is a prompt id, any unique prefix of one, or a label.
- `diff A B`: a unified text diff of the two prompts and the metric deltas.
- `ab A B`: compares outcomes of runs grouped by prompt version.

## A/B comparison

`ab` puts runs that used version A and runs that used version B side by side, with sample
sizes, and tests whether the difference is real. A run that used both versions is left
out of both groups.

| `--metric` | Better is | Test |
| --- | --- | --- |
| `pass_rate` (default) | higher | Fisher's exact test, or a two-proportion z test when every expected count is at least 5 |
| `cost` | lower | Mann-Whitney U |
| `latency` | lower | Mann-Whitney U |
| `judge:<name>` | higher | Mann-Whitney U on the judge's 0-1 scores |

A failed or errored run counts as not passing; runs with an unknown outcome are skipped.
`judge:<name>` uses [LLM-as-judge](judges.md) verdicts, which are cached, so a re-run costs
nothing. Mann-Whitney is exact for samples of up to 12 runs with no ties, and a normal
approximation with tie and continuity corrections otherwise. The statistics are pure
Python (no scipy).

The verdict is about B against A at significance level `--alpha` (default 0.05):

- `better` or `worse`: B differs significantly from A in the good or bad direction;
- `inconclusive`: the difference is not significant. With fewer than 5 runs on a side the
  result says so; record more runs before deciding.

`--json` prints sample sizes, both values, the delta, the p-value, the test and the
verdict. With `--fail-on-regression` the exit code is 1 when B is significantly worse
than A, so it works as a CI gate. Exit code 2 means an unknown version or metric, no data,
or a judge failure.

## From Python

```python
import crashlab
from crashlab.prompts import compare_prompts, list_prompts
from crashlab.store import Store

store = Store()
versions, runs_without_prompt = list_prompts(store, since="7d")
for v in versions:
    print(v.id, v.labels, v.summary()["pass_rate"])

result = compare_prompts(store, "support-v2", "support-v3", metric="pass_rate")
print(result.verdict, result.p_value)
```

`crashlab.prompts` also exports the tests themselves (`fisher_exact`, `two_proportion_z`,
`mann_whitney`) and `diff_prompts`.
