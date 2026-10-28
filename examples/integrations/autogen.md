# crashlab with AutoGen (autogen-agentchat)

Tested with `autogen-agentchat` / `autogen-ext[openai]` 0.7.5, `openai` 3.40, against Groq
`openai/gpt-oss-20b` (OpenAI-compatible endpoint). Record, replay, diagnose, fix, verify
(VERIFIED 3/3 vs control 0/3, 2/2 regression), export-test and the HTML report all work.

```python
import asyncio, crashlab
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

@crashlab.tool()
def get_price(item: str) -> str: ...

@crashlab.tool(side_effects=True)
def place_order(item: str, quantity: int) -> str: ...

async def _main(msg: str) -> str:
    mc = OpenAIChatCompletionClient(model=MODEL, base_url=BASE_URL, api_key=KEY,
        model_info=ModelInfo(vision=False, function_calling=True, json_output=False,
                             family="unknown", structured_output=False))
    mc._client = crashlab.wrap(mc._client)  # no public way to pass a client
    agent = AssistantAgent("shop", model_client=mc, tools=[get_price, place_order],
                           system_message=PROMPT, reflect_on_tool_use=True)
    res = await agent.run(task=msg)
    await mc.close()  # else "Event loop is closed" noise across verify trials
    return res.messages[-1].content

@crashlab.agent(name="ag-shop")
def run_agent(msg: str) -> str:   # sync entry point: replay/fork/verify call it
    return asyncio.run(_main(msg))
```

Notes

- The client lives on the private `_client` attribute; wrap it right after construction.
- `@crashlab.tool` functions work as AutoGen tools unchanged (signature and docstring are kept).
- Structured output (`json_output` / `output_content_type`) goes through
  `beta.chat.completions.parse` / `.stream`, which crashlab does not intercept yet.
- Close the model client inside the coroutine: every verify trial runs its own `asyncio.run`.
- `model_client_stream=True` works: the stream is recorded and replays offline (tested
  on Groq gpt-oss-120b).
