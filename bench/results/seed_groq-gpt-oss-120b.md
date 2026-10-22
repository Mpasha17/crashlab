### groq-gpt-oss-120b

> Partial: daily quota exhausted: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m57bxy

| split | diagnoser | fault triggered | detected | label ok | step exact | step in window | fix VERIFIED | VERIFIED or LIKELY | clean false alarms | cost |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | groq-gpt-oss-120b | 8/12 (67%, 39–86) | 8/8 (100%, 68–100) | 8/8 (100%, 68–100) | 7/8 (88%, 53–98) | 7/8 (88%, 53–98) | 1/2 (50%, 10–90) | 2/2 (100%, 34–100) | 0/3 (0%, 0–56) | $0.0195 |
| dev | heuristic | 8/12 (67%, 39–86) | 8/8 (100%, 68–100) | 8/8 (100%, 68–100) | 6/8 (75%, 41–93) | 6/8 (75%, 41–93) | – | – | 0/3 (0%, 0–56) | $0 |
| test | groq-gpt-oss-120b | 11/11 (100%, 74–100) | 5/11 (46%, 21–72) | 3/11 (27%, 10–57) | 3/11 (27%, 10–57) | 5/11 (46%, 21–72) | 0/2 (0%, 0–66) | 0/2 (0%, 0–66) | – | $0.0253 |
| test | heuristic | 11/11 (100%, 74–100) | 11/11 (100%, 74–100) | 0/11 (0%, 0–26) | 0/11 (0%, 0–26) | 8/11 (73%, 43–90) | – | – | – | $0 |
| all | groq-gpt-oss-120b | 19/23 (83%, 63–93) | 13/19 (68%, 46–85) | 11/19 (58%, 36–77) | 10/19 (53%, 32–73) | 12/19 (63%, 41–81) | 1/4 (25%, 5–70) | 2/4 (50%, 15–85) | 0/3 (0%, 0–56) | $0.0447 |
| all | heuristic | 19/23 (83%, 63–93) | 19/19 (100%, 83–100) | 8/19 (42%, 23–64) | 6/19 (32%, 15–54) | 14/19 (74%, 51–88) | – | – | 0/3 (0%, 0–56) | $0 |

Per cell (task × fault): triggered / LLM label ok / heuristic label ok / fix verdicts

| task | condition | runs | triggered | LLM label | heuristic label | fix verdicts |
|---|---|---|---|---|---|---|
| refund-bot | clean | 3 | – | 3/3 no-failure | 3/3 no-failure | – |
| refund-bot | spec_drop | 3 | 3/3 | 3/3 | 3/3 | VERIFIED |
| refund-bot | tool_5xx | 3 | 2/3 | 2/2 | 2/2 | LIKELY |
| refund-bot | schema_drift | 3 | 0/3 | 0/0 | 0/0 | – |
| refund-bot | misleading_desc | 3 | 3/3 | 3/3 | 3/3 | – |
| sql-analyst | clean | 3 | – | 2/3 no-failure | 0/3 no-failure | – |
| sql-analyst | spec_drop | 3 | 3/3 | 0/3 | 0/3 | – |
| sql-analyst | tool_5xx | 3 | 3/3 | 3/3 | 0/3 | REJECTED |
| sql-analyst | schema_drift | 3 | 3/3 | 0/3 | 0/3 | REJECTED |
| sql-analyst | misleading_desc | 2 | 2/2 | 0/2 | 0/2 | – |
