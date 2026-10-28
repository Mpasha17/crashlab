# crashlab with LangChain AgentExecutor / create_tool_calling_agent

Tested with `langchain` 1.4.7 + `langchain-classic` 1.0.8 (AgentExecutor moved there in
LangChain 1.x), `langchain-openai` 1.7.3, `openai` 3.40, Groq `openai/gpt-oss-20b`.
Record, replay, diagnose, fix, verify, export-test and the HTML report work. The fixer's
fix-1 (an appended "inform the customer" rule) was correctly REJECTED (0/3: the agent asked
instead of ordering); re-inserting the dropped rule in place was VERIFIED 3/3 vs control 0/3,
2/2 regression runs.

```python
import crashlab
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

@tool
@crashlab.tool()          # @crashlab.tool innermost, LangChain's @tool outermost
def get_price(item: str) -> str:
    """Look up the unit price of an item."""

@crashlab.agent(name="lc-shop")
def run_agent(msg: str) -> str:
    oc = crashlab.wrap(OpenAI(base_url=BASE_URL, api_key=KEY))
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=KEY,
                     client=oc.chat.completions, root_client=oc)   # both are needed
    prompt = ChatPromptTemplate.from_messages([("system", PROMPT), ("human", "{input}"),
                                               ("placeholder", "{agent_scratchpad}")])
    tools = [get_price, place_order]
    ex = AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools)
    return ex.invoke({"input": msg})["output"]
```

Notes

- Pass the wrapped client as `client=oc.chat.completions, root_client=oc`; `ChatOpenAI`
  otherwise builds its own. It calls `with_raw_response.create`, which crashlab records
  and (after 51c603b) replays as a raw response.
- `from langchain.agents import AgentExecutor` is gone in LangChain 1.x; install
  `langchain-classic`.
