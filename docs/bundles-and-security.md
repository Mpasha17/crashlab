# Debug bundles and secret audits

A recorded run is the most useful thing you can hand a teammate — and the
riskiest, because cassettes can hold API keys, bearer tokens, or personal
data that leaked into a prompt or tool result. `crashlab bundle` packs a
run into a portable `.crashlab.zip` archive with those secrets redacted by
default. `crashlab audit-secrets` and `crashlab redact-run` scan or clean a
run that is already in the store.

## Pack a shareable bundle

```bash
crashlab bundle last
crashlab bundle last -o debug.crashlab.zip
crashlab bundle last --no-html
crashlab bundle last --no-redact --no-html
```

The archive contains:

- `manifest.json` — crashlab version, root run id, included run ids, whether
  the runs were redacted, and a secret-audit summary (field paths and
  redacted previews only; never the raw values)
- `runs/<run_id>.jsonl` — the root run and, by default, every linked child
  or subagent run, in the same portable JSONL format as `crashlab export`
- `diagnosis.json` — the stored diagnosis, or a heuristic one if the run
  has not been diagnosed yet (no model call)
- `context.json` — the `analyze_context` prompt breakdown plus the
  `audit_tool_schemas` summary
- `report.html` — the HTML trace report (omit with `--no-html`)

`--no-redact` leaves secrets in the packed runs. Prefer the default.

## Import a bundle

```bash
crashlab unbundle debug.crashlab.zip
crashlab import debug.crashlab.zip
```

`crashlab import` of a `.zip` path delegates to the same importer as
`crashlab unbundle`. Existing JSONL exports are unchanged: `crashlab import
run.jsonl` still adds a single run.

## Audit and redact a run in the store

```bash
crashlab audit-secrets last
crashlab audit-secrets last --include-pii --json
crashlab audit-secrets --all --fail-on-match
crashlab redact-run last
crashlab redact-run last --include-pii --in-place
```

`audit-secrets` walks every step, the run input and output, metadata, and
findings. It flags OpenAI `sk-` keys, Anthropic `sk-ant-` keys, GitHub
`ghp_` / `github_pat_` tokens, AWS `AKIA` ids, Slack `xox[baprs]-` tokens,
Google `AIza` keys, `Bearer` tokens, JWTs starting `eyJ`, and PEM private
keys. `--include-pii` also flags email addresses and phone-like numbers.
`--fail-on-match` exits `1` when a hit is found, so a CI job can refuse a
leaky recording.

`redact-run` replaces each hit with a `[REDACTED:<kind>]` placeholder.
Without `--in-place` it only prints how many replacements would be made.

## In Python

```python
from crashlab.bundle import audit_secrets, export_bundle, import_bundle, redact_run
from crashlab.store import Store, load

run = load("last")
report = audit_secrets(run)
report = audit_secrets(run, include_pii=True)
for finding in report.findings:
    print(finding.step_idx, finding.field_path, finding.kind, finding.redacted_preview)
redacted, n = redact_run(run, include_pii=True)
store = Store()
manifest = export_bundle(
    store, run.run_id, "debug.crashlab.zip", redact=True, include_html=True
)
ids = import_bundle(store, "debug.crashlab.zip")
```

`export_bundle(..., include_children=False)` packs only the root run.
`redact_run(..., store=store, in_place=True)` writes the cleaned cassette
back to the store.
