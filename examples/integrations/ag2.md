# crashlab with AG2 (`ag2` 1.x)

Tested with `ag2` 1.2.2 (the new `ag2` package, `from ag2 import Agent`), `openai` 3.40,
Groq `openai/gpt-oss-20b`. Record, replay (needs crashlab after 51c603b: the replayed
completion must be a real `ChatCompletion`), diagnose, export-test and the HTML report work.

```python
import asyncio, crashlab
from dataclasses import dataclass
from ag2 import Agent
from ag2.config import OpenAICompletionsConfig

@dataclass(slots=True)
class RecordedConfig(OpenAICompletionsConfig):  # no public way to pass a client
    def create(self):
        client = OpenAICompletionsConfig.create(self)
        client._client = crashlab.wrap(client._client)
        return client

@crashlab.agent(name="ag2-shop")
def run_agent(msg: str) -> str:
    async def main():
        cfg = RecordedConfig(model=MODEL, base_url=BASE_URL, api_key=KEY, temperature=0)
        agent = Agent("shop", prompt=PROMPT, config=cfg, tools=[get_price, place_order])
        return (await agent.ask(msg)).body
    return asyncio.run(main())
```

Notes

- `OpenAICompletionsConfig.create()` builds an `AsyncOpenAI` on the private `_client` of
  the returned `OpenAIClient`; override `create()` to wrap it. (`OpenAIConfig` is a
  deprecated alias.) The Responses config (`OpenAIResponsesConfig`) is analogous.
- With crashlab 0.0.1 from PyPI, replay failed with
  `UnsupportedToolError: Unsupported tool type 'function' for provider 'openai-completions'`
  (the replayed response was not a real SDK model). Fixed on main.
