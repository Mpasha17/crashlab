# Note: refund-bot policy_drop → SystemPromptPatch fork (2026-10-23)

Live Groq (`openai/gpt-oss-120b`) smoke, Worker C.

## Setup
- Parent: `FAULT_POLICY_DROP=1` on order C03 (45 days since delivery).
- Parent run `01M5KPB26YGDFPFNR9C4M1A3V0`: **fail** — `refund_within_policy` (refunded $120).
- Patch: `examples/refund_bot/restore_policy.yaml` / `fork_demo` inserts the missing
  30-day delivered-window line after "Always look up the order".

## Result (k=2, with control)
| arm | outcomes |
|---|---|
| patched | **2/2 pass** (refused; outside window) |
| control (no patch) | **2/2 fail** (refunded $120 again) |

## Honesty
Treat as a **short anecdote**, not a VERIFIED claim (prior Day-10 VERIFIED used k=3).
Confirms the restore-policy patch flips the oracle on this parent for these two seeds.
Logs: `/workspace/crashlab-live/worker_c_policy_fork.log` (local live store, not committed).
