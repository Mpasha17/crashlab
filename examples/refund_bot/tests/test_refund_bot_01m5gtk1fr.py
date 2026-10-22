"""Exported by `crashlab export-test` from run 01M5GTK1FRTM8F3HP1Y13WY2CH (refund-bot).
Diagnosis: step 3 TASK_SPEC_VIOLATION: The agent issued a refund for an order that was
    delivered 45 days ago, violating the 30-day refund policy window.
Fix: system prompt: append "4. Refunds can only be issued if the order was delivered
    within the l…"
Verification verdict: LIKELY
Offline: replays recorded cassettes; no API keys and no network needed.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.crashlab
CASSETTES = Path(__file__).parent / "cassettes"
EXPECTED_CHECKS = {
    "refund_amount_equals_order_total": "pass",
    "refund_within_policy": "fail",
    "no_unbacked_refund_claim": "pass",
    "within_step_budget": "pass",
}
FIX = (
    'system prompt: append "4. Refunds can only be issued if the order was delivered within the l…"'  # noqa: E501
)


def test_refund_bot_01m5gtk1fr_reproduces(crashlab_replay):
    run = crashlab_replay(CASSETTES / "refund_bot_01m5gtk1fr.jsonl")
    assert run.outcome == "fail"
    assert {c.name: c.status for c in run.checks} == EXPECTED_CHECKS


def test_refund_bot_01m5gtk1fr_fix_keeps_passing(crashlab_replay):
    run = crashlab_replay(CASSETTES / "refund_bot_01m5gtk1fr_fixed.jsonl")
    # the fix is pinned: the recorded patch is re-applied to every replayed request
    assert run.patch_description == FIX
    assert run.outcome == "pass"
    assert all(c.status == "pass" for c in run.checks), run.checks
