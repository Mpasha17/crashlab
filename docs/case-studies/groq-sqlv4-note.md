# Note: Groq sql-analyst sqlv4 (2026-10-23)

Not a third-party finding — documents the held-out test-split rerun after Day-10
invalid Groq numbers (empty-reply bug).

- **15/15** episodes completed on `openai/gpt-oss-120b`
- Clean: **1/3 pass** (ep0). ep1/ep2 hit Groq `tool_use_failed` before the
  `f0b6e3e` nudge was in the running process
- LLM label ok **3/11 (10–57%)**; heuristic label 0/11, detect 11/11
- Fix VERIFIED **0/3** (REJECTED on tool_5xx / spec_drop / misleading_desc — expected
  for prompt patches against tool/DB faults)
- Cost ~$0.03 list / $0 free tier
- Offline rescore identical

Full table: `bench/results/seed_groq-gpt-oss-120b-sqlv4.md`. Replaces sqlv1–v3 as the
Groq test-split reference.

## After `failed_generation` recovery (`b31ca7e`)

A separate **clean smoke** on the same agent (no injected faults) passed **3/3** on
2026-10-23 ~10:57 IST Asia/Calcutta. That does not rewrite the sqlv4 test-split table
above (still 1/3 clean during the 15/15 run, before the recovery was in-process); it
means the denominator agent is healthy again for further work. See
`docs/launch/STATUS.md`.
