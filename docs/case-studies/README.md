# Case studies

Real (not injected) findings from pointing crashlab at popular OSS agent examples.
Upstream issues/PRs are **not** filed from these notes unless Mahaboob decides to.

| finding | verdict | draft issue |
|---|---|---|
| [smolagents text_to_sql empty `:memory:` DB](smolagents-text-to-sql.md) | fix verified (no-LLM A/B) | [DRAFT](smolagents-DRAFT_ISSUE.md) |
| [OpenAI Agents SDK triage follow-up](openai-agents-customer-service.md) | failure real; both prompt fixes **REJECTED** (no routing gain + regression) | [DRAFT](openai-agents-DRAFT_ISSUE.md) |

Demo for the flagship smolagents finding: [../demo/](../demo/).

Tried and deferred: LlamaIndex (openai 2.x vs openai-agents 3.x pin conflict in the
shared Day-11 venv). Prefer a separate venv next time.
| [Groq sqlv4 test-split note](groq-sqlv4-note.md) | bench rerun, not an upstream bug | — |
