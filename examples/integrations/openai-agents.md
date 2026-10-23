# crashlab with the OpenAI Agents SDK (`openai-agents`)

Tested with `openai-agents` 0.24 against Groq and Gemini through their OpenAI-compatible
endpoints. We drove the SDK's own `examples/customer_service` unmodified with scripted
customers.

```python
import asyncio, os
import crashlab
from openai import AsyncOpenAI
from agents import (OpenAIChatCompletionsModel, RunConfig, Runner,
                    set_default_openai_api, set_default_openai_client, set_tracing_disabled)

@crashlab.agent(name="my-support-bot")
def converse(turns: list[str], model: str = "gemini-flash-lite-latest") -> str:
    client = crashlab.wrap(AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["KEY"]))
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")      # the Responses API is OpenAI-only
    set_tracing_disabled(True)                      # no OpenAI trace export without a key
    cfg = RunConfig(model=OpenAIChatCompletionsModel(model=model, openai_client=client))

    async def go():
        agent, items, out = triage_agent, [], []
        for t in turns:
            items.append({"role": "user", "content": t})
            res = await Runner.run(agent, items, run_config=cfg)
            out.append(f"[{res.last_agent.name}] {res.final_output}")
            items, agent = res.to_input_list(), res.last_agent
        return "\n".join(out)

    return asyncio.run(go())
```

Notes:

- **Create the client inside the `@crashlab.agent` function.** `crashlab fork` and
  `crashlab verify` rerun that function, and the patched requests go through the wrapped
  client.
- **Pass a model object, not a string, for names that contain a slash.** For a string
  model, the SDK's `MultiProvider` treats `openai/gpt-oss-20b` as a provider prefix plus a
  model, which gives a 404. `qwen/...` fails with `UserError: Unknown prefix`.
  `OpenAIChatCompletionsModel(model=..., openai_client=client)` avoids both.
- **Multi-agent apps share one client.** Every agent's instructions arrive as the system
  prompt of its own requests. Scope a fix to one agent with
  `{"kind": "system_prompt", ..., "when": "<phrase only in that agent's instructions>"}`.
  Without `when`, an `append` lands in every agent's prompt, and a `substitute` fails on
  agents whose prompt lacks the text. `crashlab fix` (prompt `fix-v2`) proposes `when`
  itself.
- **Tool calls and tool results are recorded** as the tool-call parts of the LLM requests
  and responses. We did not test whether stacking `@crashlab.tool()` under the SDK's
  `@function_tool` keeps the schema that the SDK derives from the signature, so separate
  `execute_tool` steps are untested for this SDK.
- - **Close the async client at the end of each `@crashlab.agent` call.**
  `crashlab verify` / `fork` call the entry function many times; each call uses
  `asyncio.run()`, which closes the loop. An `AsyncOpenAI` left open across calls
  raises `RuntimeError: Event loop is closed` on the next trial. Pattern:
  `try: ... finally: await client.close()` inside the coroutine.

- **Groq quirk.** `openai/gpt-oss-20b` on Groq rejects the SDK's zero-argument handoff tools
  (`'required' present but 'properties' is missing`, 400), even though the schema does have
  `"properties": {}`. Use another model, or another provider for that model.
