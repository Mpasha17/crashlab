"""Exported by `crashlab export-test` from run 01M5KSJ8SFV917KPRM0SS9MMCK (refund-bot).
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


def test_refund_bot_01m5ksj8sf_reproduces(crashlab_replay):
    run = crashlab_replay(CASSETTES / "refund_bot_01m5ksj8sf.jsonl")
    assert run.outcome == "fail"
    assert {c.name: c.status for c in run.checks} == EXPECTED_CHECKS
