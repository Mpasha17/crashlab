# Framework integration notes

How to wrap popular agent frameworks with crashlab. Each note was written while running
that framework live on a free-tier OpenAI-compatible endpoint.

| Framework | Doc | Status |
|---|---|---|
| OpenAI Agents SDK | [openai-agents.md](openai-agents.md) | Works; close AsyncOpenAI across verify trials; use `when` for multi-agent prompts |
| smolagents | [smolagents.md](smolagents.md) | Works; thread fallback for CodeAgent; `:memory:` SQLite caveat |
| CrewAI | [crewai.md](crewai.md) | Works after `with_raw_response` recording fix |
| AutoGen (autogen-agentchat 0.7) | [autogen.md](autogen.md) | Works; wrap the private `_client`; close it per trial |
| AG2 (`ag2` 1.x) | [ag2.md](ag2.md) | Works; subclass the config's `create()` to wrap the client |
| Haystack Agent | [haystack.md](haystack.md) | Works; assign a wrapped `gen.client` |
| DSPy ReAct | [dspy.md](dspy.md) | Works with `engine="litellm"`, `cache=False`; the default engine is not recorded |
| Google ADK | [google-adk.md](google-adk.md) | Works via `LiteLlm`; native Gemini (google-genai) not recorded yet |
| LangChain AgentExecutor | [langchain.md](langchain.md) | Works; `ChatOpenAI(client=..., root_client=...)`, `langchain-classic` |
| Agno | [agno.md](agno.md) | Works; public `client=` |
| LangGraph | [langgraph.md](langgraph.md) | Works (sync); async streaming records and replays; `ChatOpenAI(client=..., root_client=...)` |
| PydanticAI | [pydantic-ai.md](pydantic-ai.md) | Works; `OpenAIProvider(openai_client=wrapped)` |
| LlamaIndex | [llamaindex.md](llamaindex.md) | Works; `OpenAILike(async_openai_client=wrapped)` |

LlamaIndex pins openai 2.x and the OpenAI Agents SDK needs 3.x: use separate venvs.
