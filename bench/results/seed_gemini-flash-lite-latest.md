### gemini-flash-lite-latest

| split | diagnoser | fault triggered | detected | label ok | step exact | step in window | fix VERIFIED | VERIFIED or LIKELY | clean false alarms | cost |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | gemini-flash-lite-latest | 6/12 (50%, 25–75) | 4/6 (67%, 30–90) | 4/6 (67%, 30–90) | 4/6 (67%, 30–90) | 4/6 (67%, 30–90) | 2/2 (100%, 34–100) | 2/2 (100%, 34–100) | 0/3 (0%, 0–56) | $0.0284 |
| dev | heuristic | 6/12 (50%, 25–75) | 6/6 (100%, 61–100) | 6/6 (100%, 61–100) | 6/6 (100%, 61–100) | 6/6 (100%, 61–100) | – | – | 0/3 (0%, 0–56) | $0 |
| test | gemini-flash-lite-latest | 6/12 (50%, 25–75) | 6/6 (100%, 61–100) | 6/6 (100%, 61–100) | 3/6 (50%, 19–81) | 6/6 (100%, 61–100) | 0/3 (0%, 0–56) | 1/3 (33%, 6–79) | 0/3 (0%, 0–56) | $0.0721 |
| test | heuristic | 6/12 (50%, 25–75) | 6/6 (100%, 61–100) | 2/6 (33%, 10–70) | 0/6 (0%, 0–39) | 3/6 (50%, 19–81) | – | – | 0/3 (0%, 0–56) | $0 |
| all | gemini-flash-lite-latest | 12/24 (50%, 31–69) | 10/12 (83%, 55–95) | 10/12 (83%, 55–95) | 7/12 (58%, 32–81) | 10/12 (83%, 55–95) | 2/5 (40%, 12–77) | 3/5 (60%, 23–88) | 0/6 (0%, 0–39) | $0.1005 |
| all | heuristic | 12/24 (50%, 31–69) | 12/12 (100%, 76–100) | 8/12 (67%, 39–86) | 6/12 (50%, 25–75) | 9/12 (75%, 47–91) | – | – | 0/6 (0%, 0–39) | $0 |

Per cell (task × fault): triggered / LLM label ok / heuristic label ok / fix verdicts

| task | condition | runs | triggered | LLM label | heuristic label | fix verdicts |
|---|---|---|---|---|---|---|
| refund-bot | clean | 3 | – | 3/3 no-failure | 3/3 no-failure | – |
| refund-bot | spec_drop | 3 | 3/3 | 3/3 | 3/3 | VERIFIED |
| refund-bot | tool_5xx | 3 | 0/3 | 0/0 | 0/0 | – |
| refund-bot | schema_drift | 3 | 0/3 | 0/0 | 0/0 | – |
| refund-bot | misleading_desc | 3 | 3/3 | 1/3 | 3/3 | VERIFIED |
| sql-analyst | clean | 3 | – | 3/3 no-failure | 3/3 no-failure | – |
| sql-analyst | spec_drop | 3 | 2/3 | 2/2 | 2/2 | REJECTED |
| sql-analyst | tool_5xx | 3 | 3/3 | 3/3 | 0/3 | REJECTED |
| sql-analyst | schema_drift | 3 | 0/3 | 0/0 | 0/0 | – |
| sql-analyst | misleading_desc | 3 | 1/3 | 1/1 | 0/1 | LIKELY |
