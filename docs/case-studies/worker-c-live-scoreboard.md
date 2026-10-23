# Worker C live scoreboard — 2026-10-23 (Asia/Calcutta)

**Session closed ~13:41 IST** (soft stop before 14:00). Keys via sourced env only. Live store not committed.

## Verifies (honest labels)

| fault / experiment | model | verdict | evidence |
|---|---|---|---|
| policy_drop → restore-policy | gpt-oss-120b | **VERIFIED** | patched 3/3, control 0/3 |
| policy_drop → restore-policy | gpt-oss-20b | **VERIFIED** | 3/3 vs 0/3 (×2 incl. ~13:26 re-verify) |
| policy_drop → restore-policy | gemini-flash-lite-latest | **VERIFIED** | patched 3/3, control 0/3 |
| html_502 → no-retry prompt | gpt-oss-120b | **LIKELY** | k=3: 3/3 vs 1/3; k=5: **INCONCLUSIVE** (TPD) |
| misleading_desc → tool_description | gpt-oss-20b | **VERIFIED** | 3/3 vs 0/3 (+ regression; re-verify ~13:00) |
| misleading_desc → tool_description | gemini-flash-lite-latest | **VERIFIED** | 3/3 vs 0/3 + regression |
| misleading_desc CLI verify | gpt-oss-120b | **INCONCLUSIVE** | TPD 429s |

## Clean / integration smokes

| smoke | result |
|---|---|
| Groq refund clean A17 (120b / 20b) | **pass** / **pass** (reconfirmed through ~13:30) |
| Groq sql clean (120b) | **3/3** earlier; TPD flake mid-day |
| Groq sql clean (20b) | **fixed** `output_parse_failed` → **3/3** (reconfirmed) |
| Groq schema_drift (120b / 20b) | **pass** / **pass** |
| Groq html_502 | 120b fail-loop once; **20b 5+/5 pass** (stops) |
| Gemini html_502 | **pass** (stops) |
| CrewAI ×3 (20b) | **3/3** (multiple rechecks) |
| Gemini refund clean / C03 refuse | **pass** / **pass** |
| Gemini sql clean | **3/3** |

## sql fault smokes (1q)

| provider | schema_drift | spec_drop | tool_5xx | misleading |
|---|---|---|---|---|
| gpt-oss-20b | pass | fail | fail | fail |
| gemini-flash-lite | 429 | fail | fail | — |

## Code fix this session
- `examples/sql_analyst`: recover Groq `output_parse_failed` via `failed_generation` prose/JSON
  (`069876b` + follow-ups). Offline tests green; live 20b sql **3/3**.

## Offline pins (`tests/crashlab/`)
- `test_refund_bot_01m5kq4eee` — misleading_desc fail (20b)
- `test_refund_bot_01m5ksfg22` — policy_drop fail (20b)
- `test_refund_bot_01m5ksj8sf` — policy_drop fail (Gemini)
- `test_refund_bot_01m5ksmj8q` — misleading_desc fail (Gemini)
- `test_refund_bot_01m5kpm44v` — html_502 fail + `_fixed` (120b)

## Quota at close (~13:41 IST)
- gpt-oss-20b: **667/1000** req remaining
- gpt-oss-120b: **727/1000** req remaining (TPD still the fragile limit ~200k/day)
- Prefer **20b** / Gemini for further live work

## Worker C commits (selected)
`069876b` fix sql recovery · docs verifies/scoreboard · `tests/crashlab/` exports · tip near `2faf260`/`f777371`/this close
