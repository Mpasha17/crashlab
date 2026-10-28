# crashlab with Agno

Tested with `agno` 3.1.11 (`OpenAILike` model), `openai` 3.40, Groq `openai/gpt-oss-20b`.
Record, replay, diagnose, fix, verify (VERIFIED 3/3 vs 0/3, 2/2 regression), export-test
and the HTML report all work. The smoothest integration of the set: a public `client=`.

```python
import crashlab
from openai import OpenAI
from agno.agent import Agent
from agno.models.openai.like import OpenAILike

@crashlab.agent(name="agno-shop")
def run_agent(msg: str) -> str:
    llm = OpenAILike(id=MODEL, base_url=BASE_URL, api_key=KEY,
                     client=crashlab.wrap(OpenAI(base_url=BASE_URL, api_key=KEY)))
    agent = Agent(model=llm, tools=[get_price, place_order], instructions=PROMPT)
    return agent.run(msg).content
```

Notes

- For `agent.arun(...)` also pass `async_client=crashlab.wrap(AsyncOpenAI(...))`.
- `@crashlab.tool` functions work as Agno tools unchanged.
- `agent.run(msg, stream=True)` is recorded (5/5 steps) and replays offline.
