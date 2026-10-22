### groq-gpt-oss-120b-sqlv2

> Partial: daily quota exhausted: RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m57bxy

| split | diagnoser | fault triggered | detected | label ok | step exact | step in window | fix VERIFIED | VERIFIED or LIKELY | clean false alarms | cost |
|---|---|---|---|---|---|---|---|---|---|---|
| test | groq-gpt-oss-120b-sqlv2 | – | – | – | – | – | – | – | 0/1 (0%, 0–79) | $0.0010 |
| test | heuristic | – | – | – | – | – | – | – | 0/1 (0%, 0–79) | $0 |
| all | groq-gpt-oss-120b-sqlv2 | – | – | – | – | – | – | – | 0/1 (0%, 0–79) | $0.0010 |
| all | heuristic | – | – | – | – | – | – | – | 0/1 (0%, 0–79) | $0 |

Per cell (task × fault): triggered / LLM label ok / heuristic label ok / fix verdicts

| task | condition | runs | triggered | LLM label | heuristic label | fix verdicts |
|---|---|---|---|---|---|---|
| sql-analyst | clean | 1 | – | 1/1 no-failure | 1/1 no-failure | – |
