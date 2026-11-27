# Automatic repair: `crashlab repair`

`diagnose`, `fix` and `verify` each answer one question about a failed run. `crashlab
repair` runs them as one loop and only stops when a patch is **verified** or it runs
out of ideas:

1. **Diagnose** the run: the label, the failing step and verbatim evidence
   (`crashlab diagnose`; heuristic, no LLM call, when no diagnoser model is set).
2. **Propose** candidate patches in crashlab's patch format: restores of what past
   passing runs had (always, offline), plus LLM-proposed system prompt line edits, tool
   description / schema edits and model param changes when a model is configured.
   Tool result overrides are counterfactuals: they prove a cause, so they are reported
   as evidence and never chosen as the fix.
3. **Verify** each candidate by forking the failed run `--trials` times with the patch
   and `--trials` times without it (the control), plus your past passing runs as a
   regression check. Errored trials (429, 5xx) never count as fails, and the verdict
   uses Wilson intervals, so a flaky agent cannot fake a fix.
4. **Iterate**: rejected attempts are fed back to the fix model ("this did not work,
   try a different target or wording") for the next round, up to `--max-iterations`.

```bash
crashlab repair last
crashlab repair last --model gpt-4o-mini --trials 5 --candidates 3 --max-iterations 3
crashlab repair last --write-patch fixes/refund.json --export-test tests/crashlab
crashlab repair last --judge safety --judge-model gpt-4o-mini --min-pass-rate 1.0
crashlab repair last --dry-run
crashlab repair last --json
```

Each attempt prints as it finishes, then a table of every round:

```text
✗ round 1 fix-1 [REJECTED] 0/3 vs control 0/3 rate 0.00: system prompt: append "4. Only refund orders with a receipt."
✓ round 2 fix-2 [VERIFIED] 3/3 vs control 0/3 rate 1.00: system prompt: append "4. Only refund delivered orders, ..."
FIXED after 2 round(s)
```

| Result | Meaning | Exit |
| --- | --- | --- |
| `fixed` | a patch is VERIFIED and its judged pass rate reaches `--min-pass-rate` | 0 |
| `partial` | the best patch beat the control but is only LIKELY (raise `--trials`) or below the pass rate | 1 |
| `not_fixed` | no candidate beat the control, or nothing could be proposed | 1 |
| usage error | unknown run, a run that passed, bad options | 2 |

`--write-patch` writes the chosen patch (`.json`, or `.yaml` with PyYAML) for
`crashlab fork --patch` or your code review. `--export-test` writes the same offline
pytest regression test as `crashlab export-test --fix`: the failing cassette, and the
patched passing fork replayed with the fix re-applied.

## Judges and oracles

The agent's own `@crashlab.check` oracles grade every fork. `--judge NAME` (repeatable)
adds LLM-as-judge criteria ([judges.md](judges.md)): a patched trial only counts as a
pass when the checks pass **and** every judge passes, so a "fix" that makes the agent
rude or unsafe is not accepted.

## Python

```python
import crashlab

result = crashlab.repair(
    "last",
    max_iterations=3,
    candidates=3,
    trials=5,
    diagnoser="gpt-4o-mini",      # or fn(messages) -> reply; heuristic when unset
    judges=["safety"],
    judge_model="gpt-4o-mini",
    oracle=lambda run: "refund" not in (run.output or "").lower(),
    max_cost_usd=1.00,
)
print(result.status, result.verified_pass_rate, result.cost)
for attempt in result.attempts:
    print(attempt.iteration, attempt.fix_id, attempt.verdict, attempt.pass_rate)
if result.fixed:
    result.write_patch("fixes/refund.json")
```

`propose_fn=fn(messages) -> reply` replaces the fix model, and `client=` is passed to
every fork, so with a fake fix model and a fake agent client the whole loop is
deterministic and offline (that is how `tests/test_repair.py` runs it).

## Audit trail

Nothing is hidden: every attempt is a stored verification (listed by
`crashlab report RUN` under "Fix attempts", children tagged `verify:<id>`), and the
whole repair (diagnosis, attempts, chosen patch, evidence, cost) is saved as
`<store>/runs/<id>/repairs/<repair_id>.json` (plain JSON; `repairs(run_id)` from
`crashlab.repair` lists them).

A VERIFIED repair is observed outcomes under the patch vs the unpatched control, not a
proof that the patch addresses the root cause in general; review the patch before you
ship it, and keep the exported regression test.
