# crashlab with Haystack agents

Tested with `haystack-ai` 3.4.1 (`haystack.components.agents.Agent` +
`OpenAIChatGenerator`), `openai` 3.40, Groq `openai/gpt-oss-20b`. Record, replay,
diagnose, fix, verify (VERIFIED 3/3 vs 0/3, 2/2 regression), export-test and the HTML
report all work.

```python
import crashlab
from openai import OpenAI
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import create_tool_from_function
from haystack.utils import Secret

@crashlab.agent(name="hs-shop")
def run_agent(msg: str) -> str:
    gen = OpenAIChatGenerator(model=MODEL, api_base_url=BASE_URL, api_key=Secret.from_token(KEY))
    gen.client = crashlab.wrap(OpenAI(base_url=BASE_URL, api_key=KEY))  # warm_up keeps it
    agent = Agent(chat_generator=gen, system_prompt=PROMPT,
                  tools=[create_tool_from_function(get_price), create_tool_from_function(place_order)])
    return agent.run(messages=[ChatMessage.from_user(msg)])["last_message"].text
```

Notes

- `warm_up()` only creates `gen.client` when it is `None`, so assigning a wrapped client
  first is enough. For `run_async`, set `gen.async_client = crashlab.wrap(AsyncOpenAI(...))`.
- `create_tool_from_function` reads the `@crashlab.tool` function's signature and docstring fine.
- With crashlab 0.0.1 from PyPI, replay failed with
  `AssertionError: Unexpected response type for non-streaming request` (Haystack checks
  `isinstance(..., ChatCompletion)`; Groq's extra `service_tier` made the rebuild fall
  back to a generic object). Fixed on main (51c603b).
- Streaming (`streaming_callback=...`) records and replays offline (tested on Groq gpt-oss-120b).
- Gemini's OpenAI-compatible endpoint returns `400` on the second model call (after a tool
  call) with this framework; reproduced without crashlab. Groq works.
