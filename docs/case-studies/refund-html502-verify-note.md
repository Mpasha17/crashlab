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


## Follow-up: model variance (same day)
- **gpt-oss-20b** html_502 A17 ×3: all **pass** (stopped/apologized within budget; 3–7 steps). No verify needed.
- **gpt-oss-120b** earlier parent looped to 11 steps (**fail**); later 120b calls hit **TPD** (~199k/200k used).
- Explains control **1/3 pass** in the 120b LIKELY verify — stopping without a patch is common on smaller/faster models.


## More model variance
- **gemini-flash-lite-latest** html_502 A17 ×2: both **pass** (apology within budget).
- Together with gpt-oss-20b passes, the 120b step-budget loop looks **model-specific / stochastic**,
  which is why control scored 1/3 in the LIKELY verify — not a broken oracle.


## k=5 re-attempt (12:50 IST)
`crashlab fix --verify -k 5` on same parent: **INCONCLUSIVE** — mostly Groq 120b TPD
429s (2/2 patched usable pass, 1/1 control usable pass, 7 errored). Prior **LIKELY**
(k=3) still stands; do not upgrade.

## Offline pin
`tests/crashlab/test_refund_bot_01m5kpm44v.py` (fail + `_fixed` with fix-1).
