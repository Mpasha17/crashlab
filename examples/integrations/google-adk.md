# crashlab with Google ADK

Tested with `google-adk` 2.13 using `LiteLlm` (`litellm` 1.106) against Gemini
`gemini-flash-lite-latest` via its OpenAI-compatible endpoint. Record, replay, diagnose,
fix, verify, export-test and the HTML report work. Both fixer proposals for a dropped
quantity rule were correctly REJECTED (the agent asked "would you like 5?" instead of
ordering); re-inserting the rule in place was VERIFIED 3/3 vs control 0/3, 2/2 regression.

```python
import asyncio, crashlab, crashlab.integrations.litellm
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

crashlab.integrations.litellm.patch()   # before the first model call

@crashlab.agent(name="adk-shop")
def run_agent(msg: str) -> str:
    async def main():
        agent = LlmAgent(name="shop", instruction=PROMPT, tools=[get_price, place_order],
                         model=LiteLlm(model="openai/" + MODEL, api_base=BASE_URL, api_key=KEY))
        svc = InMemorySessionService()
        runner = Runner(agent=agent, app_name="shop", session_service=svc)
        s = await svc.create_session(app_name="shop", user_id="u")
        out = ""
        async for ev in runner.run_async(user_id="u", session_id=s.id,
                new_message=types.Content(role="user", parts=[types.Part(text=msg)])):
            if ev.is_final_response() and ev.content and ev.content.parts:
                out = "".join(p.text or "" for p in ev.content.parts)
        return out
    return asyncio.run(main())
```

Gotchas

- **Native Gemini models (`model="gemini-..."`) go through `google-genai`, which crashlab
  does not record yet.** Use `LiteLlm` (or an OpenAI-compatible endpoint) for now.
- `pip install mcp` too: LiteLLM 1.106's `acompletion` with tools imports `mcp` and fails
  with `APIConnectionError: No module named 'mcp'` otherwise (not a crashlab issue).
- ADK + LiteLLM + Groq gpt-oss fails live on the 2nd model call:
  `'messages.2': property 'reasoning_content' is unsupported` (ADK sends the reasoning back).
  Reproduced without crashlab; crashlab records it as an errored chat step.
