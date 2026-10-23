# DRAFT upstream issue — not filed

File only if Mahaboob decides to. Do not open this against openai/openai-agents-python
from the agent.

**Repo:** https://github.com/openai/openai-agents-python
**Example:** `examples/customer_service/main.py`

---

## Title

customer_service example: Triage Agent answers follow-up questions itself instead of handing off to the FAQ agent

## Body

In `examples/customer_service/main.py`, after the Seat Booking Agent finishes and
transfers the customer back (per its routine), a follow-up question such as "is there
wifi on the plane?" is answered by the Triage Agent from its own knowledge. Its
instructions only say "You can use your tools to delegate questions to other appropriate
agents."

With `qwen/qwen3.8-27b` (via an OpenAI-compatible endpoint, `OpenAIChatCompletionsModel`),
the conversation "I need to change my seat" → "Confirmation QX7Z31, and I want 23A" →
"Thanks! One more thing: is there wifi on the plane?" never reached the FAQ tool in 9/9
unpatched runs. Answers included invented policy ("free and premium in-flight Wi-Fi on
most routes ... our Facebook page"). `gemini-flash-lite-latest` handled it correctly, so
the example works on stronger models but is fragile on smaller open models.

### Suggested fix

```diff
 triage_agent = Agent[AirlineAgentContext](
     ...
     instructions=(
         f"{RECOMMENDED_PROMPT_PREFIX} "
-        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
+        "You are a helpful triaging agent. You never answer customer questions yourself: you only route them. "
+        "Questions about the airline (baggage, seats on the plane, wifi, or anything else) go to the FAQ agent; "
+        "seat changes go to the seat booking agent. Do this for every new question, including follow-up "
+        "questions after another agent has transferred the customer back to you."
     ),
```

### Also noticed

`faq_lookup_tool`'s keyword match misses questions such as "Which rows are the exit
rows?", even though the seating answer contains them; adding "row"/"exit" to the seat
keywords would fix that.

### Status of verification

A scoped crashlab patch of the triage prompt was proposed. Full `VERIFIED` (k=5 +
controls + regression) was blocked by free-tier quotas on earlier attempts; a fresh
verify is in progress. Do not file until a clean VERIFIED (or honest LIKELY) exists.

