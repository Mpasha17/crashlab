"""Tool results holding numbers strict JSON cannot store: NaN / inf (stored as strings)
came back as the strings "nan" / "inf" on replay, so ``math.isnan(x)`` raised TypeError
in a replay of a run that worked; an int over Python's 4300-digit str() limit crashed
the agent while its step was written. Both now round-trip."""

from __future__ import annotations

import math
import warnings

import pytest

import crashlab


@crashlab.tool
def stats() -> dict:
    return {"mean": float("nan"), "bounds": [1.5, float("-inf"), float("inf")], "n": 0}


@crashlab.tool
def lone() -> float:
    return float("nan")


@crashlab.tool
def factorial_like() -> list:
    return [10**5000 + 7, -(10**6000), 3]


@crashlab.agent(name="numbers")
def numbers() -> list:
    s = stats()
    big = factorial_like()
    return [math.isnan(s["mean"]), s["bounds"][1] == -math.inf, s["bounds"][2] == math.inf,
            math.isnan(lone()), big[0] % 10, big[1] < 0, big[1].bit_length(), big[2],
            s["n"]]  # fmt: skip


EXPECTED = [True, True, True, True, 7, True, 19932, 3, 0]


def test_non_finite_floats_and_huge_ints_replay_as_numbers(crashlab_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # nothing stops recording
        assert numbers() == EXPECTED
    run = crashlab.load("last")
    assert run.status == "ok" and len(run.steps) == 3
    replayed = crashlab.replay(run)
    assert replayed.status == "ok", replayed.error
    assert replayed.output == EXPECTED
    # the source run's recorded step is not changed by serving it
    assert crashlab.load(run.run_id).steps[0].response["result"]["mean"] == "nan"


def test_a_value_that_cannot_be_stored_stops_recording_not_the_agent(crashlab_dir, monkeypatch):
    from crashlab.store import cassette

    real = cassette.json.dumps

    def dumps(obj, *a, **k):
        if isinstance(obj, dict) and "boom" in str(obj.get("response")):  # the step line
            raise ValueError("cannot serialize")
        return real(obj, *a, **k)

    monkeypatch.setattr(cassette.json, "dumps", dumps)

    @crashlab.tool
    def boom() -> str:
        return "boom"

    @crashlab.agent(name="unstorable")
    def agent() -> str:
        return boom() + "!"

    with pytest.warns(RuntimeWarning, match="cannot store a value"):
        assert agent() == "boom!"
