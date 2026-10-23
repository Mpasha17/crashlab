# Demo: smolagents empty-`:memory:` bug

Terminal capture of the Day 11 flagship finding. **No API keys.**

## Quick look (no install)

- Animated GIF: [`smolagents-text-to-sql.gif`](smolagents-text-to-sql.gif)
- Self-contained play-through: open [`terminal.html`](terminal.html) in a browser
  (no network assets)
- Transcript: [`transcript.txt`](transcript.txt)

## Replay the no-LLM A/B

```bash
# needs smolagents + sqlalchemy (e.g. a Day-11 real-agents venv)
REAL_AGENTS=/path/to/real-agents bash docs/demo/demoscript.sh
```

What you should see:

1. A recorded crashlab check failing on the live run (oracle expects Woodrow Wilson).
2. `python repro.py`: upstream path → `no such table: receipts`; with `StaticPool` →
   `('Woodrow Wilson',)`.

## Offline README path (refund_bot cassettes)

Separate from this smolagents demo — the main README "Try it with no API key" section
imports scrubbed `examples/refund_bot/demo/*.jsonl` runs. Helper:
[`quickstart-offline.sh`](quickstart-offline.sh).

Full write-up: [../case-studies/smolagents-text-to-sql.md](../case-studies/smolagents-text-to-sql.md).
