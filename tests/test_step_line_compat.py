"""Cassette lines written by main stay readable by released crashlab 0.0.2, whose models
reject unknown keys (it skipped such steps as "damaged"); and main itself keeps steps
written by a newer crashlab instead of dropping them."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import crashlab
from crashlab.store.cassette import line_to_step, read_cassette, step_to_line

FIX = Path(__file__).parent / "fixtures" / "v0_0_2" / "compat_shop.jsonl"


def _v002_keys():
    """Step / ErrorInfo keys a 0.0.2 cassette line has (0.0.2 accepts exactly these)."""
    step_keys, err_keys = set(), set()
    for raw in FIX.read_text().splitlines()[1:]:
        line = json.loads(raw)
        step_keys |= set(line)
        if line.get("error"):
            err_keys |= set(line["error"])
    return step_keys, err_keys


@crashlab.tool()
def lookup(key: str) -> int:
    return {"a": 1}[key]


@crashlab.agent(name="state-agent")
def agent() -> str:
    try:
        lookup("b")
    except KeyError as e:
        return f"missing {e.args[0]}"
    return "?"


def test_lines_with_error_state_use_only_keys_0_0_2_knows(crashlab_dir):
    agent()
    step_keys, err_keys = _v002_keys()
    raw = next(crashlab_dir.rglob("cassette.jsonl")).read_text().splitlines()
    line = json.loads(raw[0])
    assert set(line) <= step_keys, set(line) - step_keys
    assert set(line["error"]) <= err_keys, set(line["error"]) - err_keys
    # main still sees the state and replays with it
    run = crashlab.load("last")
    assert run.steps[0].error.state == {"args": ["b"]}
    assert "crashlab.error.state" not in run.steps[0].attributes
    assert crashlab.replay(run).output == "missing b"


def test_reads_state_written_inline_by_earlier_main():
    line = step_to_line(json.loads(FIX.read_text().splitlines()[3]))
    line["error"]["state"] = {"args": ["x"]}  # how main wrote it on Nov 5
    assert line_to_step(line).error.state == {"args": ["x"]}


def test_unknown_future_keys_keep_the_step_with_a_warning(tmp_path):
    lines = [json.loads(r) for r in FIX.read_text().splitlines()[1:]]
    lines[1]["from_the_future"] = {"x": 1}
    lines[2]["error"]["also_new"] = True
    p = tmp_path / "cassette.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        steps = read_cassette(p)
    assert len(steps) == 5  # nothing dropped
    msg = " ".join(str(w.message) for w in caught)
    assert "newer crashlab" in msg and "from_the_future" in msg and "also_new" in msg
