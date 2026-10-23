### groq-gpt-oss-120b-sqlv4

| split | diagnoser | fault triggered | detected | label ok | step exact | step in window | fix VERIFIED | VERIFIED or LIKELY | clean false alarms | cost |
|---|---|---|---|---|---|---|---|---|---|---|
| test | groq-gpt-oss-120b-sqlv4 | 11/12 (92%, 65–98) | 7/11 (64%, 35–85) | 3/11 (27%, 10–57) | 3/11 (27%, 10–57) | 7/11 (64%, 35–85) | 0/3 (0%, 0–56) | 0/3 (0%, 0–56) | 0/1 (0%, 0–79) | $0.0301 |
| test | heuristic | 11/12 (92%, 65–98) | 11/11 (100%, 74–100) | 0/11 (0%, 0–26) | 0/11 (0%, 0–26) | 8/11 (73%, 43–90) | – | – | 0/1 (0%, 0–79) | $0 |
| all | groq-gpt-oss-120b-sqlv4 | 11/12 (92%, 65–98) | 7/11 (64%, 35–85) | 3/11 (27%, 10–57) | 3/11 (27%, 10–57) | 7/11 (64%, 35–85) | 0/3 (0%, 0–56) | 0/3 (0%, 0–56) | 0/1 (0%, 0–79) | $0.0301 |
| all | heuristic | 11/12 (92%, 65–98) | 11/11 (100%, 74–100) | 0/11 (0%, 0–26) | 0/11 (0%, 0–26) | 8/11 (73%, 43–90) | – | – | 0/1 (0%, 0–79) | $0 |

Per cell (task × fault): triggered / LLM label ok / heuristic label ok / fix verdicts

| task | condition | runs | triggered | LLM label | heuristic label | fix verdicts |
|---|---|---|---|---|---|---|
| sql-analyst | clean | 3 | – | 2/3 no-failure | 1/3 no-failure | – |
| sql-analyst | spec_drop | 3 | 3/3 | 0/3 | 0/3 | REJECTED |
| sql-analyst | tool_5xx | 3 | 3/3 | 3/3 | 0/3 | REJECTED |
| sql-analyst | schema_drift | 3 | 2/3 | 0/2 | 0/2 | – |
| sql-analyst | misleading_desc | 3 | 3/3 | 0/3 | 0/3 | REJECTED |
