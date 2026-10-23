# Draft launch posts — not published

## X / Twitter (short)

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

Day-11 evidence: on the unmodified smolagents text-to-sql example, the agent hit
`no such table: receipts` because SQLAlchemy's `:memory:` pool is per-thread and
CodeAgent runs tools in a fresh ThreadPoolExecutor. A StaticPool one-liner fixes it;
we reproduced it with no LLM.

Pre-alpha. Feedback welcome. github.com/Mpasha17/crashlab
