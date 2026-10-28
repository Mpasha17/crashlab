# Show HN: crashlab – crash-test your AI agents (record → diagnose → verify a fix)

**Title:** Show HN: crashlab – crash-test your AI agents

**Text:**

I built crashlab after watching too many agent demos fail the same way twice.

It is a local-first Python library. You keep your agent loop. You wrap the LLM client
and decorate tools. crashlab then:

- records every LLM and tool call (OpenAI, Anthropic, LiteLLM; sync + async + streaming)
- replays a failed run exactly, offline (no keys, no network)
- diagnoses the failing step with verbatim evidence
- proposes a typed patch and verifies it (k live forks + controls + regression)
- exports a pytest regression test and a self-contained HTML report

Honest early evidence: we ran huggingface/smolagents' official text-to-sql example
unmodified, and the oracle failed — CodeAgent was querying an empty `:memory:` SQLite.
crashlab pinned the cause: LocalPythonExecutor runs each code block in a fresh
ThreadPoolExecutor, and SQLAlchemy's default SingletonThreadPool opens a new, empty
in-memory DB in that thread. We weren't first: it's a known bug with an open,
unmerged fix PR (https://github.com/huggingface/smolagents/pull/2475). crashlab reproduced and diagnosed it independently and adds
a no-LLM repro confirming the StaticPool fix. Write-up in the repo under docs/case-studies/.

We also crashed the OpenAI Agents SDK customer_service example (triage answers
follow-ups itself on a smaller open model). Two prompt patches were correctly
REJECTED by crashlab verify for regressing past good FAQ runs — the interesting
outcome is the rejection, not a fake green check.

Pre-alpha (0.0.1): `pip install crashlab`. Looking for people building agents who will
break it.

https://github.com/Mpasha17/crashlab
