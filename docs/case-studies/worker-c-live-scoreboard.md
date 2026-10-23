# Worker C live scoreboard — 2026-10-23 (Asia/Calcutta)

Session ~11:39–14:00 IST. Keys via sourced env only. Live store not committed.

## Verifies (honest labels)

| fault / experiment | model | verdict | evidence |
|---|---|---|---|
| policy_drop → restore-policy `SystemPromptPatch` | gpt-oss-120b | **VERIFIED** | patched 3/3, control 0/3 (parent `01M5KPB26Y…`) |
| html_502 → no-retry system prompt | gpt-oss-120b | **LIKELY** | patched 3/3, control 1/3 + 1 clean regression |
| misleading_desc → `tool_description` fix | gpt-oss-20b (fix from 120b diagnoser) | **VERIFIED** | patched 3/3, control 0/3, regression 1/1 |
| misleading_desc `crashlab fix --verify` | gpt-oss-120b | **INCONCLUSIVE** | TPD 429s (Limit 200000) |
| openai-agents triage (prior) | — | **REJECTED** | unchanged; do not claim VERIFIED |

## Clean / integration smokes

| smoke | result |
|---|---|
| Groq refund_bot clean A17 (120b) | **pass** 4/4 |
| Groq sql_analyst clean (120b) | **3/3 pass** (earlier); mid-day reconfirm hit TPD mid-run |
| Groq sql_analyst clean (20b) | **0/3** `output_parse_failed` — [handoff](groq-output-parse-failed-note.md) |
| Groq schema_drift A17 (120b) | **pass** |
| Groq html_502 (20b) ×3 | **pass** (stops); 120b earlier **fail** loop |
| CrewAI lookup ×3 (20b) | **3/3 pass** (+ recheck 3/3) |
| Gemini flash-lite refund clean / C03 refuse | **pass** / **pass** |
| Gemini flash-lite sql clean | **3/3** after one Q2 429 retry |

## Quota notes
- Groq **gpt-oss-120b** TPD ~200k/day — repeatedly near cap; prefer 20b for forks after ~12:00.
- Gemini free-tier: intermittent 429; recovered same session.

## Commits (Worker C docs only)
See `git log --author` / STATUS stamps; secret-scan green on each push.
