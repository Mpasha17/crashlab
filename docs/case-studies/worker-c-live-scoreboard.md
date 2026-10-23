# Worker C live scoreboard — 2026-10-23 (Asia/Calcutta)

Session ~11:39–14:00 IST. Keys via sourced env only. Live store not committed.

## Verifies (honest labels)

| fault / experiment | model | verdict | evidence |
|---|---|---|---|
| policy_drop → restore-policy `SystemPromptPatch` | gpt-oss-120b | **VERIFIED** | patched 3/3, control 0/3 |
| policy_drop → restore-policy (replicate) | gpt-oss-20b | **VERIFIED** | patched 3/3, control 0/3 |
| html_502 → no-retry system prompt | gpt-oss-120b | **LIKELY** | patched 3/3, control 1/3 + 1 regression |
| misleading_desc → `tool_description` fix | gpt-oss-20b (fix from 120b diagnoser) | **VERIFIED** | patched 3/3, control 0/3, regression 1/1 |
| misleading_desc `crashlab fix --verify` | gpt-oss-120b | **INCONCLUSIVE** | TPD 429s (Limit 200000) |

## Clean / integration smokes

| smoke | result |
|---|---|
| Groq refund_bot clean A17 (120b) | **pass** 4/4 |
| Groq sql_analyst clean (120b) | **3/3 pass** (earlier; mid-day TPD flake on reconfirm) |
| Groq sql_analyst clean (20b) | was **0/3** `output_parse_failed` → **fixed** → **3/3** |
| Groq schema_drift A17 (120b) | **pass** |
| Groq html_502 (20b) ×3 | **pass** (stops); 120b earlier **fail** loop |
| CrewAI lookup ×3 (20b) | **3/3 pass** (+ recheck 3/3) |
| Gemini flash-lite refund clean / C03 refuse | **pass** / **pass** |
| Gemini flash-lite sql clean | **3/3** (with occasional 429 retry) |

## Code fix this session
- `examples/sql_analyst`: recover Groq `output_parse_failed` via `failed_generation` prose/JSON
  (`069876b`). Offline tests green; live 20b sql **3/3**.

## Quota notes
- Groq **gpt-oss-120b** TPD ~200k/day — near cap mid-session; prefer **20b** / Gemini after.
- Gemini free-tier: intermittent 429; usually recovers quickly.

## Commits (Worker C)
Docs + sql_analyst fix only; secret-scan green; no PyPI / force-push / upstream issues.
