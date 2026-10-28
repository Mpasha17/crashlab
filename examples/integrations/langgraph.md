# crashlab with LangGraph

Tested with `langgraph` 1.2.18 (`create_react_agent`), `langchain-openai`, `openai` 3.40,
Groq `openai/gpt-oss-20b` (sync: every stage works) and Gemini flash-lite (async +
`astream`, streaming: recorded and replayed, but every Gemini run hit the upstream 400
below, so fix/verify were not exercised on that path).

```python
import crashlab
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
@crashlab.tool(side_effects=True)   # crashlab innermost, LangChain's @tool outermost
def place_order(item: str, quantity: int) -> str:
    """Place an order for `quantity` units of `item`."""

@crashlab.agent(name="lg-shop")
def run_agent(msg: str) -> str:
    oc = crashlab.wrap(OpenAI(base_url=BASE_URL, api_key=KEY))
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=KEY, temperature=0,
                     client=oc.chat.completions, root_client=oc)
    agent = create_react_agent(llm, [get_price, place_order], prompt=PROMPT)
    return agent.invoke({"messages": [("user", msg)]})["messages"][-1].content
```

Notes

- `crashlab.wrap(ChatOpenAI(...))` raises `TypeError`: wrap the OpenAI client and hand
  it to `ChatOpenAI(client=..., root_client=...)`. For async, wrap an `AsyncOpenAI` and
  pass `async_client=oc.chat.completions, root_async_client=oc`; `streaming=True` and
  `astream(stream_mode="messages")` are recorded and replayed.
- One recorded step per LLM call and per tool call (3 AIMessage + 2 ToolMessage = 5 steps).
- Replaying `with_raw_response` (what `ChatOpenAI` uses) needs crashlab after 51c603b;
  replaying a provider error (e.g. a 400) needs 914f23b.
- Read keys with `os.environ.get(...)`: replay and exported tests run without them.
- A fix that changes a `side_effects=True` call (order 5 instead of 10) can only be
  graded with `crashlab verify --allow-live-side-effects` against sandboxed tools;
  without it the trials are not counted (INCONCLUSIVE, with a hint).
- Gemini through `ChatOpenAI` streaming: the second turn of a tool loop can get a 400
  (Gemini wants its `thought_signature` back, which LangChain drops). Upstream issue.
