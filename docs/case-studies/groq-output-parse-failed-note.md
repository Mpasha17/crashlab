# Note: Groq gpt-oss-20b `output_parse_failed` on sql-analyst (2026-10-23)

Worker C live. **Not fixed here** — handoff for Worker A if worth extending recovery.

## Symptom
`examples.sql_analyst` clean smoke on `openai/gpt-oss-20b`: **0/3**. Each run ends in
`openai.BadRequestError` with `code: output_parse_failed` and a `failed_generation`
string (often free-form reasoning that already contains the answer, e.g. `"Germany"`).

## Contrast
- Same agent on **gpt-oss-120b**: clean **3/3** (after `b31ca7e` `tool_use_failed` recovery).
- Recovery in `examples/sql_analyst/agent.py` only triggers when the exception text
  contains `tool_use_failed`, then parses JSON `{"arguments":{"answer":...}}` from
  `failed_generation`. `output_parse_failed` is a different Groq code and often not JSON.

## Suggested follow-up (Worker A)
Consider also catching `output_parse_failed` / parsing prose-or-JSON `failed_generation`
when an answer is recoverable — without weakening oracles. Until then, treat **20b sql
clean as red** on this account.
