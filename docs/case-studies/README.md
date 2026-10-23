# Case studies

Real (not injected) findings from pointing crashlab at popular OSS agent examples.
Upstream issues/PRs are **not** filed from these notes unless Mahaboob decides to.

| finding | verdict | draft issue |
|---|---|---|
| [smolagents text_to_sql empty `:memory:` DB](smolagents-text-to-sql.md) | fix verified (no-LLM A/B) | [DRAFT](smolagents-DRAFT_ISSUE.md) |
| [OpenAI Agents SDK triage follow-up](openai-agents-customer-service.md) | failure real; both prompt fixes **REJECTED** (no routing gain + regression) | [DRAFT](openai-agents-DRAFT_ISSUE.md) |
| [CrewAI `with_raw_response` recording](crewai-integration.md) | crashlab gap fixed; 3/3 live scenarios pass | — |
| [Groq sqlv4 test-split note](groq-sqlv4-note.md) | bench rerun, not an upstream bug | — |
| [Groq 20b `output_parse_failed`](groq-output-parse-failed-note.md) | **fixed**; live 20b sql clean 3/3 | — |
| [sql-analyst 20b fault smokes](sql-20b-fault-smokes-note.md) | observational after recovery fix | — |
| [refund-bot policy_drop fork](refund-policy-fork-note.md) | **VERIFIED** patched 3/3 control 0/3 (Groq); k=2 was anecdote | — |
| [refund-bot html_502 verify](refund-html502-verify-note.md) | **LIKELY** patched 3/3 control 1/3 (Groq) | — |
| [refund-bot misleading_desc](refund-misleading-desc-note.md) | **VERIFIED** on 20b + Gemini; 120b verify INCONCLUSIVE (TPD) | — |

| [Worker C live scoreboard](worker-c-live-scoreboard.md) | session 2026-10-23 IST | — |

Demo for the flagship smolagents finding: [../demo/](../demo/).

Tried and deferred: LlamaIndex (openai 2.x vs openai-agents 3.x pin conflict in the
shared Day-11 venv). Prefer a separate venv next time.
