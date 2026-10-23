# Demo: smolagents empty-`:memory:` bug

Terminal capture of the Day 11 flagship finding. **No API keys.**

## Replay

```bash
# needs smolagents + sqlalchemy (e.g. the Day 11 real-agents venv)
REAL_AGENTS=/path/to/real-agents bash docs/demo/demoscript.sh
# or just read the recorded transcript:
cat docs/demo/transcript.txt
```

Animated GIF: [`smolagents-text-to-sql.gif`](smolagents-text-to-sql.gif).

Open [`terminal.html`](terminal.html) in a browser for a self-contained play-through of
the same transcript (no network assets).

## What you should see

1. A recorded crashlab check failing on the live run (oracle expects Woodrow Wilson).
2. `python repro.py`: upstream path → `no such table: receipts`; with `StaticPool` →
   `('Woodrow Wilson',)`.

Full write-up: [../case-studies/smolagents-text-to-sql.md](../case-studies/smolagents-text-to-sql.md).
