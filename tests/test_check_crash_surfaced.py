"""A built-in check that crashes must say so on the run (a ``check_error`` finding and a
warning), never vanish into a debug log: a RecursionError in bad_tool_args silently
turned that check off for the run."""

from __future__ import annotations

import warnings

import pytest

import crashlab
from crashlab.checks import detectors


@crashlab.tool()
def lookup(key: str) -> str:
    return key


@pytest.fixture
def broken(monkeypatch):
    def boom(self, step, h):
        raise ZeroDivisionError("detector bug")

    monkeypatch.setattr(detectors.BadToolArgs, "on_step", boom)

    def boom_end(self, run, h):
        raise KeyError("end bug")

    monkeypatch.setattr(detectors.NoLlmCalls, "on_end", boom_end)


def test_crashing_check_is_a_finding_and_a_warning_not_a_silent_skip(broken):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with crashlab.record("crashy"):
            lookup("a")
            lookup("b")  # crashes again: still one finding per check
    run = crashlab.load("last")
    errs = [f for f in run.findings if f.kind == "check_error"]
    assert {f.key for f in errs} == {"bad_tool_args", "no_llm_calls"}
    bad = next(f for f in errs if f.key == "bad_tool_args")
    assert bad.severity == "warn" and bad.steps == [0, 1]
    assert (
        "bad_tool_args crashed" in bad.message and "ZeroDivisionError: detector bug" in bad.message
    )
    assert "incomplete" in bad.message
    msgs = [str(w.message) for w in caught if "crashed" in str(w.message)]
    assert len(msgs) == 2  # once per check, not per step


def test_agent_still_runs_normally(broken):
    @crashlab.agent(name="crashy-agent")
    def agent() -> str:
        return lookup("x")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert agent() == "x"
    assert crashlab.load("last").status == "ok"
