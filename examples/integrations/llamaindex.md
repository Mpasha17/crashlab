# crashlab with LlamaIndex

Tested with `llama-index-core` 0.14.25 (`FunctionAgent` workflow), `llama-index-llms-openai-like`
0.8, Groq `openai/gpt-oss-20b`, in its own venv. Every stage works.

```python
import crashlab
from openai import AsyncOpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai_like import OpenAILike

@crashlab.agent(name="li-shop")
async def run_agent(msg: str) -> str:
    llm = OpenAILike(model=MODEL, api_base=BASE_URL, api_key=KEY, is_chat_model=True,
                     is_function_calling_model=True, context_window=32000,
                     async_openai_client=crashlab.wrap(AsyncOpenAI(base_url=BASE_URL, api_key=KEY)))
    agent = FunctionAgent(tools=[get_price, place_order], llm=llm, system_prompt=PROMPT)
    return str(await agent.run(user_msg=msg))
```

Notes

- `FunctionAgent` always streams; the chat steps are recorded as streams and replay fine.
- Plain `@crashlab.tool` functions work as `tools=[...]`.
- The exported tests print many `PydanticDeprecatedSince20` warnings; they come from
  llama-index, not crashlab.
