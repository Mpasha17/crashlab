# crashlab with PydanticAI

Tested with `pydantic-ai-slim[openai]` 2.63.0 (`Agent.run` and `Agent.run_stream`, async),
Groq `openai/gpt-oss-20b`. Every stage works.

```python
import crashlab
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

@crashlab.tool(side_effects=True)
def place_order(item: str, quantity: int) -> str:
    """Place an order for `quantity` units of `item`."""

@crashlab.agent(name="pai-shop")
async def run_agent(msg: str) -> str:
    client = crashlab.wrap(AsyncOpenAI(base_url=BASE_URL, api_key=KEY))
    model = OpenAIChatModel(MODEL, provider=OpenAIProvider(openai_client=client))
    agent = Agent(model, system_prompt=PROMPT, tools=[get_price, place_order])
    return (await agent.run(msg)).output
```

Notes

- Pass `@crashlab.tool` functions in `tools=[...]`; PydanticAI reads the wrapped
  function's signature and docstring for the schema.
- PydanticAI checks `isinstance(response, ChatCompletion)`. Groq adds
  `service_tier="on_demand"`, which the OpenAI model rejects on validation; crashlab
  0.0.1 then replayed a generic object and PydanticAI raised `UnexpectedModelBehavior`.
  Fixed in 51c603b.
- `PYDANTIC_AI_NO_BANNER=1` hides the startup banner in CI logs.
