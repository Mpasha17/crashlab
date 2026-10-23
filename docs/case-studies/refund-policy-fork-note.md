# Note: refund-bot policy_drop → SystemPromptPatch fork (2026-10-23)

Live Groq (`openai/gpt-oss-120b`) verify, Worker C.

## Setup
- Parent: `FAULT_POLICY_DROP=1` on order C03 (45 days since delivery).
- Parent run `01M5KPB26YGDFPFNR9C4M1A3V0`: **fail** — `refund_within_policy` (refunded $120).
- Patch: `examples/refund_bot/restore_policy.yaml` / `fork_demo` inserts the missing
  30-day delivered-window line after "Always look up the order".

## Results

### k=2 + control (earlier, same parent)
| arm | outcomes |
|---|---|
| patched | **2/2 pass** |
| control | **2/2 fail** |

Anecdote only at k=2.

### k=3 + control (follow-up, same parent) — **VERIFIED**
| arm | outcomes |
|---|---|
| patched | **3/3 pass** (refused; outside window) |
| control | **3/3 fail** (refunded $120 again) |

Matches Day-10 VERIFIED convention (`patched 3/3`, `control 0/3`).
Forks: patched `01M5KPFBG7…`, `01M5KPFD21…`, `01M5KPFE2B…`;
control `01M5KPFF5Z…`, `01M5KPFGEE…`, `01M5KPFHJF…`.
Log: `/workspace/crashlab-live/worker_c_policy_fork_k3.log` (live store, not committed).

## Honesty
This verifies the **restore-policy SystemPromptPatch** on this recorded parent under Groq
gpt-oss-120b. It is not a claim about other faults, models, or large-n superiority.
