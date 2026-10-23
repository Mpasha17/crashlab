# Draft launch posts — not published

## X / Twitter (≤280, paste-ready)

Crash-test your AI agents.

crashlab: record → offline replay → diagnose → verify a typed patch.

Found a real bug in huggingface/smolagents text-to-sql (empty :memory: SQLite across threads). StaticPool fix.

github.com/Mpasha17/crashlab (pre-alpha, not on PyPI)

## X alt (≤280, mentions REJECTED)

Crash-test your AI agents.

Record LLM+tool calls, replay failures offline, diagnose with evidence, verify patches (incl. REJECTED).

Real bug in smolagents text-to-sql: empty :memory: SQLite across threads.

github.com/Mpasha17/crashlab · pre-alpha, not on PyPI

## X / Twitter (longer / thread starter)

Crash-test your AI agents.

crashlab records every LLM + tool call from your existing loop (OpenAI / Anthropic /
LiteLLM), replays failures offline, diagnoses the failing step with verbatim evidence,
proposes a typed patch, and verifies it against the failure and your past good runs.

We pointed it at huggingface/smolagents' text-to-sql example and found a real bug:
CodeAgent was querying an empty in-memory SQLite because `:memory:` is per-thread.
One-line StaticPool fix. No upstream issue filed yet.

github.com/Mpasha17/crashlab (pre-alpha, not on PyPI)

## LinkedIn (slightly longer)

Shipping crashlab — an open-source library to crash-test AI agents.

Wrap your client. Decorate your tools. When a run fails, crashlab:
1. Replays it exactly, offline
2. Diagnoses the failing step with quotes from the trace
3. Proposes a minimal typed patch (system prompt, tool description, …)
4. Verifies the fix with repeated live forks + regression on past good runs
5. Exports a pytest regression test

Day-11 evidence:
- smolagents text-to-sql: empty `:memory:` SQLite across threads. StaticPool fix;
  no-LLM A/B. Flagship.
- OpenAI Agents customer_service: triage answers FAQ follow-ups on a small open
  model. Two prompt patches correctly REJECTED for regression — that rejection is
  the point of verify.
- CrewAI: no logic bug in the crew; found a crashlab recording gap
  (`with_raw_response`) and fixed it.

Pre-alpha. Feedback welcome. github.com/Mpasha17/crashlab
