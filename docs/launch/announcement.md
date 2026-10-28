# Draft launch posts — not published

## X / Twitter (≤280, paste-ready)

Crash-test your AI agents.

crashlab: record → offline replay → diagnose → verify a typed patch.

Reproduced + diagnosed a known smolagents text-to-sql bug (empty :memory: SQLite across threads; open fix PR #2475).

github.com/Mpasha17/crashlab (pre-alpha · pip install crashlab)

## X alt (≤280, mentions REJECTED)

Crash-test your AI agents.

Record LLM+tool calls, replay failures offline, diagnose with evidence, verify patches (incl. REJECTED).

Reproduced a known smolagents bug (empty :memory: SQLite across threads, PR #2475).

github.com/Mpasha17/crashlab · pip install crashlab

## X / Twitter (longer / thread starter)

Crash-test your AI agents.

crashlab records every LLM + tool call from your existing loop (OpenAI / Anthropic /
LiteLLM), replays failures offline, diagnoses the failing step with verbatim evidence,
proposes a typed patch, and verifies it against the failure and your past good runs.

We pointed it at huggingface/smolagents' text-to-sql example: CodeAgent was querying
an empty in-memory SQLite because `:memory:` is per-thread and the executor runs code in a
fresh thread. Known bug with an open fix PR (https://github.com/huggingface/smolagents/pull/2475); crashlab reproduced and
diagnosed it independently and adds a no-LLM repro.

github.com/Mpasha17/crashlab (pre-alpha · pip install crashlab)

## LinkedIn (slightly longer)

Shipping crashlab — an open-source library to crash-test AI agents.

Wrap your client. Decorate your tools. When a run fails, crashlab:
1. Replays it exactly, offline
2. Diagnoses the failing step with quotes from the trace
3. Proposes a minimal typed patch (system prompt, tool description, …)
4. Verifies the fix with repeated live forks + regression on past good runs
5. Exports a pytest regression test

Evidence (honest labels):
- **Flagship:** smolagents text-to-sql empty `:memory:` across threads — a known bug
  (open fix PR #2475) that crashlab reproduced and diagnosed independently; no-LLM A/B
  confirms the StaticPool fix.
- OpenAI Agents customer_service: triage answers FAQ follow-ups on a small open model.
  Two prompt patches correctly **REJECTED** for regression — that rejection is the point.
- refund-bot demos: policy_drop **VERIFIED** (Groq 120b/20b + Gemini); misleading_desc
  **VERIFIED** (20b + Gemini); html_502 **LIKELY** only. Small n.
- CrewAI: crashlab recording gap (`with_raw_response`) found and fixed; 3/3 pass.

Pre-alpha. `pip install crashlab`. Feedback welcome. github.com/Mpasha17/crashlab
