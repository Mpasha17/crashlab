# Note: refund-bot misleading_desc → tool_description fix (2026-10-23)

Worker C live evidence. Fault: `get_order` description claims a 10% restocking fee
and tells the model to refund `total * 0.9`.

## Failure (reproduced)
| model | run | outcome |
|---|---|---|
| gpt-oss-120b | `01M5KPR7A04MHE52MCCJ8CP2NH` | **fail** refunded $44.10 vs total $49 |
| gpt-oss-20b | `01M5KQ4EEE5E95NW6SJMGJB84M` | **fail** same under-refund |

## Diagnosis / fix
- **gpt-oss-120b diagnoser:** `TASK_SPEC_VIOLATION` (0.90). fix-1 =
  `tool_description` on `get_order` clarifying `total` is already the refundable
  amount (yaml under the 120b run’s `fixes/fix-1.yaml`).
- **gpt-oss-20b diagnoser:** wrongly `UNKNOWN` / blamed the oracle — **no fix**.
  Do not trust 20b alone for this fault.

## Verify

### gpt-oss-120b `crashlab fix --verify -k 3`
**INCONCLUSIVE** ×2 — all trials `RateLimitError` (Groq TPD on 120b). No verdict.

### gpt-oss-20b manual forks (same fix-1.yaml, k=3 + control + 1 regression)
Parent `01M5KQ4EEE…`. Agent model 20b (forks follow parent model).

| arm | result |
|---|---|
| patched | **3/3 pass** (refunded $49) |
| control | **0/3 pass** (still $44.10) |
| regression (clean A17 `01M5KQ4GXX…` + patch) | **1/1 pass** |

**Verdict: VERIFIED** under Day-10 convention (`patched 3/3`, `control 0/3`), with
caveats: fix authored by **120b** diagnoser; confirmations run on **20b**; not the
CLI `crashlab verify` path (blocked on 120b TPD). Logs:
`worker_c_misleading*.log` in the live store (not committed).


## gemini-flash-lite-latest (same day)
Parent `01M5KSMJ8QRWSQ3Y4YF29NYKCP` fail ($44.10). Same fix-1.yaml from 120b diagnoser:
manual k=3+control+regression → patched **3/3**, control **0/3**, regression **pass**.
**VERIFIED** on Gemini flash-lite (cross-provider confirmation of the tool_description fix).


## Offline pin
Failure cassette exported: `tests/crashlab/test_refund_bot_01m5kq4eee.py` (20b parent
`01M5KQ4EEE…`). Replay asserts the under-refund oracle fail without keys.

## gpt-oss-20b re-verify (~13:00 IST)
Fresh parent `01M5KTNVV8XT2FE5N02GD2VNJE` fail ($44.10). Same fix-1.yaml:
patched **3/3**, control **0/3** — **VERIFIED** again (stability check).
