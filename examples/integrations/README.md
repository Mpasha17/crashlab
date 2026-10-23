# Framework integration notes

How to wrap popular agent frameworks with crashlab. Each note was written while running
that framework live on a free-tier OpenAI-compatible endpoint.

| Framework | Doc | Status |
|---|---|---|
| OpenAI Agents SDK | [openai-agents.md](openai-agents.md) | Works; close AsyncOpenAI across verify trials; use `when` for multi-agent prompts |
| smolagents | [smolagents.md](smolagents.md) | Works; thread fallback for CodeAgent; `:memory:` SQLite caveat |
| CrewAI | [crewai.md](crewai.md) | Works after `with_raw_response` recording fix |

Deferred: LlamaIndex (openai 2.x pin conflict with openai-agents 3.x in a shared venv —
use an isolated venv next time).
