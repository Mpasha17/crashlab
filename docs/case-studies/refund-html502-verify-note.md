# Note: refund-bot html_502 → no-retry SystemPromptPatch (2026-10-23)

Live Groq (`openai/gpt-oss-120b`) `crashlab fix --verify -k 3`, Worker C.

## Parent
- `FAULT_HTML_502=1` on A17 → run `01M5KPM44VAHXC9J56ASYMJ8DV`
- **fail** `within_step_budget` (11 steps / budget 8)
- Findings: `step_repetition`, `tool_error_streak`, `suspicious_tool_result` (502 HTML ×5)

## Diagnosis
- Label: `STEP_REPETITION` (+ `TOOL_ERROR_MISHANDLED`), confidence 0.93
- Root cause: repeated identical `get_order` after 502 HTML

## fix-1
Append to system prompt: if a tool returns an error page / non-JSON, do **not** retry
identical args; inform the user / abort.

## Verify (`-k 3`, 1 clean regression)
| arm | result |
|---|---|
| patched | **3/3 pass** |
| control | **1/3 pass** (2/3 still hit step budget) |
| regression (clean A17) | **1/1 pass** |

**Verdict: LIKELY** (better than control; Wilson CIs overlap — raise `-k` for VERIFIED).
Cost ~$0.0024 list / free tier. Log: `/workspace/crashlab-live/worker_c_html502_verify.log`.

## Honesty
Not VERIFIED. Control sometimes stops without the patch, so the fault is partly
stochastic. Matches Day-10 refund-bot `tool_5xx` **LIKELY** band.
