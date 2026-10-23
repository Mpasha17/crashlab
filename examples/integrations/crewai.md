# crashlab with CrewAI

Tested with `crewai` 1.15 against Groq (`openai/gpt-oss-20b`) via the OpenAI-compatible
endpoint.

```python
from crewai import Agent, Crew, LLM, Process, Task
import crashlab

llm = LLM(model="openai/gpt-oss-20b", api_key=KEY, base_url=GROQ_BASE)
llm.model = "openai/gpt-oss-20b"  # CrewAI strips the openai/ prefix; Groq needs it back
llm._client = crashlab.wrap(llm._client)  # 1.15 keeps the client here

@crashlab.agent(name="my-crew")
def run(prompt: str) -> str:
    agent = Agent(..., llm=llm, tools=[...])
    return str(Crew(agents=[agent], tasks=[Task(..., agent=agent)], process=Process.sequential).kickoff())
```

## Friction fixed in crashlab

CrewAI calls ``client.chat.completions.with_raw_response.create``, not plain ``create``.
Before commit (Day 11), that path bypassed recording (0 steps, empty responses). crashlab
now records ``with_raw_response.create`` and parses the ``LegacyAPIResponse`` for the
cassette while still returning the raw object CrewAI expects.

## Other notes

- Build the LLM with ``model="openai/<name>"`` then assign ``llm.model`` to the full
  provider id when the API needs a slash (Groq gpt-oss).
- CrewAI tools are not ``@crashlab.tool``; tool use still shows up as ``tool_calls`` on
  the chat step. Oracles should look there (and at ``execute_tool`` if you also wrap the
  tool function).
- Prefer a **separate venv** from `openai-agents` (openai 3.x vs some stacks on 2.x).
