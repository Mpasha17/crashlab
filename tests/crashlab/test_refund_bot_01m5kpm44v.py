"""Exported by `crashlab export-test` from run 01M5KPM44VAHXC9J56ASYMJ8DV (refund-bot).
Diagnosis: step 3 STEP_REPETITION: The agent repeatedly called get_order with identical
    arguments despite receiving the same 502 error, causing unnecessary repetition and
    exceeding the step budget.
Fix: system prompt: append "If a tool call fails (e.g., returns an error page or
    non‑JSON respons…"
Verification verdict: INCONCLUSIVE
Offline: replays recorded cassettes; no API keys and no network needed.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.crashlab
CASSETTES = Path(__file__).parent / "cassettes"
EXPECTED_CHECKS = {
    "refund_amount_equals_order_total": "pass",
    "refund_within_policy": "pass",
    "no_unbacked_refund_claim": "pass",
    "within_step_budget": "fail",
}
FIX = (
    'system prompt: append "If a tool call fails (e.g., returns an error page or non‑JSON respons…"'  # noqa: E501
)


def test_refund_bot_01m5kpm44v_reproduces(crashlab_replay):
    run = crashlab_replay(CASSETTES / "refund_bot_01m5kpm44v.jsonl")
    assert run.outcome == "fail"
    assert {c.name: c.status for c in run.checks} == EXPECTED_CHECKS


def test_refund_bot_01m5kpm44v_fix_keeps_passing(crashlab_replay):
    run = crashlab_replay(CASSETTES / "refund_bot_01m5kpm44v_fixed.jsonl")
    # the fix is pinned: the recorded patch is re-applied to every replayed request
    assert run.patch_description == FIX
    assert run.outcome == "pass"
    assert all(c.status == "pass" for c in run.checks), run.checks
