# Case study: OpenAI Agents SDK customer_service — Triage Agent answers follow-ups itself

**Project:** [openai/openai-agents-python](https://github.com/openai/openai-agents-python)
`examples/customer_service` (openai-agents 0.24). Agents/tools/prompts unmodified.
Oracles from the example's own routines (FAQ grounded in the lookup tool; seat updates
use only customer-supplied details).

**Status:** Failure is reproducible on `qwen/qwen3.8-27b` (0/9 unpatched). A scoped
system-prompt fix sometimes works; a full `VERIFIED` (k=5) has been quota-limited.
**No upstream issue filed.** Draft: [openai-agents-DRAFT_ISSUE.md](openai-agents-DRAFT_ISSUE.md).

## Failure
Multi-turn: seat change, then "is there wifi on the plane?" After the Seat Booking Agent
hands back to Triage, Triage answers itself (off-topic or invented wifi policy) instead of
`transfer_to_faq_agent`. The FAQ fact is `Airline-Wifi`.

Gemini flash-lite passed 16/16 single- and multi-turn scenarios — the example is fragile
on smaller open models, not universally broken.

## crashlab diagnosis
`TASK_SPEC_VIOLATION` @ step 5, confidence 0.90, with a verbatim quote of the unrelated
triage reply.

## Fix shape
Scope a `system_prompt` `substitute` to the triage agent with `when: "You are a helpful
triaging agent."` (crashlab `when`-scoped patches, commit `7521bbc`). Unscoped appends
were correctly **REJECTED** by regression (they polluted the FAQ agent's prompt).

## Integration bugs this exercise produced in crashlab
- `when` on `SystemPromptPatch` for multi-agent shared clients
- verify `INCONCLUSIVE` when an arm has fewer than min(2,k) usable trials
- note: close `AsyncOpenAI` after each `asyncio.run()` in verify trials
