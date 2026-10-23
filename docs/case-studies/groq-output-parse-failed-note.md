# Note: Groq gpt-oss-20b `output_parse_failed` on sql-analyst (2026-10-23)

## Symptom (before fix)
`examples.sql_analyst` clean smoke on `openai/gpt-oss-20b`: **0/3**. Each run ended in
`openai.BadRequestError` with `code: output_parse_failed` and a prose `failed_generation`
that already contained the answer (e.g. `198.99`, `"Germany"`).

## Cause
Recovery in `examples/sql_analyst/agent.py` only triggered on `tool_use_failed` and only
parsed JSON `{"arguments":{"answer":...}}`.

## Fix (Worker C)
Also treat `output_parse_failed` as recoverable; prefer JSON `arguments.answer`, else
return the `failed_generation` prose (oracles accept `run.output`). Tests in
`tests/test_sql_analyst.py`.

## Live after fix
Groq `openai/gpt-oss-20b` clean smoke **3/3 pass** (2026-10-23 ~12:38 IST Asia/Calcutta).
