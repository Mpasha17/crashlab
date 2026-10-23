# Show HN: crashlab – crash-test your AI agents (record → diagnose → verify a fix)

**Hug:** Show HN: crashlab – crash-test your AI agents

**Text:**

I built crashlab after watching too many agent demos fail the same way twice.

It is a local-first Python library. You keep your agent loop. You wrap the LLM client
and decorate tools. crashlab then:

- records every LLM and tool call (OpenAI, Anthropic, LiteLLM; sync + async + streaming)
- replays a failed run exactly, offline (no keys, no network)
- diagnoses the failing step with verbatim evidence
- proposes a typed patch and verifies it (k live forks + controls + regression)
- exports a pytest regression test and a self-contained HTML report

Honest early evidence: we ran it on huggingface/smolagents' official text-to-sql
example and found a real bug — CodeAgent was querying an empty `:memory:` SQLite
because the default pool is per-thread and the executor hops threads. A StaticPool
fix works; no-LLM A/B confirms it. Write-up in the repo under docs/case-studies/.

Pre-alpha (0.0.1), not on PyPI yet. Looking for people building agents who will
break it.

https://github.com/Mpasha17/crashlab
