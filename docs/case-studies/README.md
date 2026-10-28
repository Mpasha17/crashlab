# Case studies

Real (not injected) findings from pointing crashlab at popular OSS agent examples.
Upstream issues/PRs are **not** filed from these notes unless Mahaboob decides to.

| finding | verdict | draft issue |
|---|---|---|
| [smolagents text_to_sql empty `:memory:` DB](smolagents-text-to-sql.md) | known bug (open fix PR [#2475](https://github.com/huggingface/smolagents/pull/2475)); reproduced + fix verified (no-LLM A/B) | [not filed: dup of #2475](smolagents-DRAFT_ISSUE.md) · [commented with repro](https://github.com/huggingface/smolagents/pull/2475#issuecomment-6472807342) |
| [OpenAI Agents SDK triage follow-up](openai-agents-customer-service.md) | failure real; both prompt fixes **REJECTED** | [DRAFT](openai-agents-DRAFT_ISSUE.md) |
| [CrewAI `with_raw_response` recording](crewai-integration.md) | crashlab gap fixed; 3/3 live scenarios pass | — |
| [Groq sqlv4 test-split note](groq-sqlv4-note.md) | bench rerun, not an upstream bug | — |
| [Groq 20b `output_parse_failed`](groq-output-parse-failed-note.md) | fixed; live 20b sql clean 3/3 | — |
| [sql-analyst 20b fault smokes](sql-20b-fault-smokes-note.md) | observational after recovery fix | — |
| [refund-bot policy_drop fork](refund-policy-fork-note.md) | **VERIFIED** on 120b, 20b, Gemini flash-lite (3/3 vs 0/3) | — |
| [refund-bot html_502 verify](refund-html502-verify-note.md) | **LIKELY** only (3/3 vs control 1/3) | — |
| [refund-bot misleading_desc](refund-misleading-desc-note.md) | **VERIFIED** on 20b + Gemini; 120b INCONCLUSIVE (TPD) | — |
| [Worker C live scoreboard](worker-c-live-scoreboard.md) | session 2026-10-23 IST | — |

Demo for the flagship smolagents reproduction: [../demo/](../demo/).

Tried and deferred: LlamaIndex (openai 2.x vs openai-agents 3.x pin conflict in the
shared Day-11 venv). Prefer a separate venv next time.
