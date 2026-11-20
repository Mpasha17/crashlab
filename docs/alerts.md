# Live production alerts

Watch the local store and fire an alert when a finished run fails, a runtime
finding is serious enough, or cost or latency goes over a cap. Alerts can go to
a webhook (generic JSON, Slack, or Discord), a local JSONL file, or a Python
callback. Delivery is stdlib-only: `urllib.request` and `pathlib`.

This is for a production or staging loop that already records into a crashlab
store. It polls the same index `crashlab watch` uses; it does not call a model.

## `crashlab watch`

Existing `watch` flags are unchanged. Alerting starts when you pass a sink:

```bash
crashlab watch --webhook https://example.com/hooks/crashlab
crashlab watch --webhook https://example.com/hooks/crashlab --webhook-format json
crashlab watch --slack-webhook https://hooks.slack.com/services/T000/B000/XXX
crashlab watch --webhook https://discord.example/api/webhooks/000/xxx --webhook-format discord
crashlab watch --alert-file alerts.jsonl --existing --timeout 60
crashlab watch --webhook https://example.com/hooks/crashlab --min-severity error --max-cost-usd 0.5 --max-latency-ms 8000 --cooldown 300
```

`--webhook-format` is `json` (the default), `slack` or `discord`. `--slack-webhook`
is the same as `--webhook` with `--webhook-format slack`; you can use both and
`--alert-file` together. `--min-severity` is the finding floor (`info`, `warn`,
`error`, `critical`; default `warn`). Cost and latency caps are off until you set
them. `--cooldown` (default 300 seconds) is per `(agent name, alert kind,
signature)`, so a tight loop of the same failure does not retry the webhook on
every poll.

A run raises an alert when any of these hold:

- the outcome is `fail`, the status is `error`, or the run recorded an exception;
- a runtime finding is at or above `--min-severity`;
- `usage.cost_usd` is above `--max-cost-usd`;
- wall time (`usage.wall_time_ms`, or `ended_at - started_at` if that is missing)
  is above `--max-latency-ms`.

Only finished runs that `watch` itself reports are dispatched (so `--existing`
includes what is already in the store, and `--failed` / `--name` still filter
which runs are seen). `--json` output stays one watch event per line; alert
delivery is silent on stdout.

## In Python

```python
from crashlab.alerts import (
    AlertDispatcher,
    AlertRule,
    CallbackSink,
    JsonlSink,
    WebhookSink,
    evaluate_run_alerts,
)
from crashlab.store import load

run = load("last")
rule = AlertRule(
    min_severity="warn",
    max_cost_usd=0.50,
    max_latency_ms=8000,
    cooldown_seconds=300,
)
events = evaluate_run_alerts(run, rule)

dispatcher = AlertDispatcher(
    rule,
    sinks=[
        WebhookSink("https://example.com/hooks/crashlab", format="json", timeout=5.0),
        JsonlSink("alerts.jsonl"),
        CallbackSink(print),
    ],
)
dispatcher.dispatch(run)
```

`evaluate_run_alerts` only decides what would fire; it does not send or apply
the cooldown. `AlertDispatcher.dispatch` does both. `WebhookSink` formats are
`json`, `slack` (`{"text": ...}`) and `discord` (`{"content": ...}`). A failed
POST is logged with the URL redacted (`_safe_url` drops query parameters and
webhook path tokens) and is not raised, so a down hook cannot stop `watch`.

**Limits.** This is polling, not a push from the agent process: a run that starts
and finishes between two polls is seen once, as finished. Cooldown is in-memory
for that `watch` process (or that dispatcher); restarting it can send the same
alert again. There is no retry queue, no authentication beyond whatever you put
in the URL or `WebhookSink(headers=...)`, and no delivery receipt. Missing cost
or latency is skipped, not treated as zero. Webhook secrets belong in the URL or
headers, not in the JSONL file, which stores the alert event as structured JSON.
