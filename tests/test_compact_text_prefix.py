"""The compact writer decides the shared prefix by comparing JSON text, one whole-list
encoding per step. These tests pin the behaviours that earlier writers got wrong:
stale identity caches, NaN/Inf, values whose ``str()`` changes, edits deep in a long
history, and reading back what 0.0.3 wrote."""

from __future__ import annotations

import json
import random
import shutil
import warnings
from copy import deepcopy
from pathlib import Path

import pytest

import crashlab
from crashlab.models import Step
from crashlab.store import Store, compact
from crashlab.store.cassette import CassetteWriter, read_cassette
from tests.fakes import FakeOpenAI, text_response

FIX = Path(__file__).parent / "fixtures" / "compact_v1_0_0_3"


def _line(seq: int, msgs: list) -> dict:
    return {
        "seq": seq,
        "op": "chat",
        "name": "m",
        "hash": f"h{seq}",
        "run_id": "r",
        "span_id": f"s{seq}",
        "request": {"messages": msgs},
        "response": {"t": seq},
    }


def _read(path: Path):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        steps = read_cassette(path)
    return [s.model_dump(mode="json", exclude={"started_at"}) for s in steps], [
        str(w.message) for w in rec
    ]


def _pair(tmp: Path, monkeypatch, script):
    """Write the same message lists with the full writer (deep copies) and the compact
    writer (the caller's live objects), then return both readings."""
    monkeypatch.delenv(compact.ENV, raising=False)
    full = CassetteWriter(tmp / "full.jsonl")
    monkeypatch.setenv(compact.ENV, "1")
    comp = CassetteWriter(tmp / "comp.jsonl")
    for seq, msgs in enumerate(script()):
        full.append(_line(seq, deepcopy(msgs)))
        comp.append(_line(seq, msgs))
    return _read(tmp / "full.jsonl"), _read(tmp / "comp.jsonl"), tmp / "comp.jsonl"


def test_nan_inf_and_negative_zero_in_the_prefix(monkeypatch, tmp_path):
    def script():
        msgs = [{"role": "tool", "content": "x", "scores": [float("nan"), float("inf"), -0.0]}]
        for i in range(6):
            msgs = msgs + [{"role": "user", "content": f"q{i}", "v": float("-inf")}]
            yield msgs

    (full, _), (comp, warns), path = _pair(tmp_path, monkeypatch, script)
    assert not warns
    assert json.dumps(comp) == json.dumps(full)  # NaN != NaN, so compare the text
    assert '"keep": 6' in path.read_text().splitlines()[-1]


class Flaky:
    """``str()`` changes on every call, like an object that prints a counter or clock."""

    n = 0

    def __str__(self) -> str:
        Flaky.n += 1
        return f"flaky-{Flaky.n}"


def test_value_whose_str_changes_is_never_a_wrong_base(monkeypatch, tmp_path):
    def script():
        msgs = [{"role": "user", "content": "hi", "obj": Flaky()}]
        for i in range(5):
            msgs = msgs + [{"role": "assistant", "content": f"a{i}"}]
            yield msgs

    _, (comp, warns), path = _pair(tmp_path, monkeypatch, script)
    # never a digest failure: the writer refuses to compact on text it cannot reproduce
    assert not any("could not be expanded" in w for w in warns), warns
    assert [len(s["request"]["messages"]) for s in comp] == [2, 3, 4, 5, 6]
    assert all(str(s["request"]["messages"][0]["obj"]).startswith("flaky-") for s in comp)


def test_edits_deep_in_a_1500_message_history_and_reversed_file(monkeypatch, tmp_path):
    """Every 97 steps an old message is edited in place (so the kept prefix shrinks to
    just before it) and sometimes one is deleted; the file is then read reversed."""
    rng = random.Random(7)

    def script():
        msgs: list[dict] = [{"role": "system", "content": "sys"}]
        for i in range(750):
            if i and i % 97 == 0:
                msgs[rng.randrange(len(msgs))]["content"] += f" edited@{i}"
            if i and i % 211 == 0:
                msgs.pop(rng.randrange(1, len(msgs)))
            msgs.append({"role": "user", "content": f"q{i}"})
            yield msgs
            msgs.append({"role": "assistant", "content": f"a{i}", "n": i})

    (full, fw), (comp, cw), path = _pair(tmp_path, monkeypatch, script)
    assert not fw and not cw
    assert comp == full
    text = path.read_text().splitlines()
    keeps = [json.loads(x)["compact"]["request.messages"]["keep"] for x in text[1:]]
    assert min(keeps) < max(keeps) - 100  # some steps really kept a shorter prefix
    assert len(text[-1]) < 2000  # still compact at the end
    rev = tmp_path / "rev.jsonl"
    rev.write_text("\n".join(text[::-1]) + "\n")
    rcomp, rw = _read(rev)
    assert not rw and rcomp == full


@crashlab.agent(name="compact-mutating")
def mutating_agent(turns: int = 8) -> str:
    c = crashlab.wrap(FakeOpenAI([text_response(i, f"answer {i}") for i in range(turns)]))
    msgs = [{"role": "system", "content": "be brief"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"question {i}"})
        if i == 4:
            msgs[1]["content"] = "question 0 (rewritten in place)"
        out = c.chat.completions.create(model="fake-model", messages=msgs)
        msgs.append({"role": "assistant", "content": out.choices[0].message.content})
    return "done"


def test_agent_that_mutates_its_history_in_place_records_what_was_sent(monkeypatch, tmp_path):
    views = {}
    for mode in ("full", "comp"):
        monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / mode))
        if mode == "comp":
            monkeypatch.setenv(compact.ENV, "1")
        else:
            monkeypatch.delenv(compact.ENV, raising=False)
        mutating_agent()
        run = crashlab.load("last")
        views[mode] = [(s.request, s.raw_request) for s in run.steps]
        monkeypatch.delenv(compact.ENV, raising=False)
    assert views["comp"] == views["full"]
    firsts = [req["messages"][0] for req, _ in views["comp"]]
    assert "rewritten" not in json.dumps(firsts[:4]) and "rewritten" in json.dumps(firsts[4])


def test_store_path_with_raw_request_matches_full(monkeypatch, tmp_path):
    for mode in ("full", "comp"):
        if mode == "comp":
            monkeypatch.setenv(compact.ENV, "1")
        else:
            monkeypatch.delenv(compact.ENV, raising=False)
        s = Store(tmp_path / mode)
        msgs: list = []
        for i in range(40):
            msgs.append({"role": "user", "content": f"q{i} token=sk-live-ABCDEFGH{i:04d}"})
            s.add_step(
                Step(
                    run_id="r",
                    idx=i,
                    span_id=f"s{i}",
                    operation="chat",
                    name="m",
                    request_hash=f"h{i}",
                    request={"messages": msgs},
                    raw_request={"model": "m", "messages": msgs},
                    response={"content": "ok"},
                    started_at="2026-11-13T00:00:00Z",
                )
            )
    monkeypatch.delenv(compact.ENV, raising=False)
    full, fw = _read(Store(tmp_path / "full").cassette_path("r"))
    comp, cw = _read(Store(tmp_path / "comp").cassette_path("r"))
    assert not fw and not cw and comp == full
    assert "sk-live-ABCDEFGH" not in Store(tmp_path / "comp").cassette_path("r").read_text()


@pytest.mark.skipif(not FIX.exists(), reason="fixture missing")
def test_reads_compact_cassette_written_by_0_0_3(tmp_path):
    """A compact cassette written by the 0.0.3 writer still reads, step for step equal
    to the full-format recording of the same run made by 0.0.3."""
    shutil.copy(FIX / "comp.jsonl", tmp_path / "comp.jsonl")
    comp, cw = _read(tmp_path / "comp.jsonl")
    full, fw = _read(FIX / "full.jsonl")
    assert not cw and not fw
    assert any('"compact"' in x for x in (FIX / "comp.jsonl").read_text().splitlines())
    assert comp == full


@pytest.mark.skipif(not FIX.exists(), reason="fixture missing")
def test_writer_output_is_byte_identical_to_0_0_3(monkeypatch, tmp_path):
    """Same input, same bytes as the 0.0.3 writer (append steps and one in-place edit
    of an old message), so files from either version read in both."""
    monkeypatch.setenv(compact.ENV, "1")
    comp = CassetteWriter(tmp_path / "comp.jsonl")
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(12):
        if i == 6:
            msgs[1]["content"] += " (edited)"
        msgs.append({"role": "user", "content": f"q{i}", "n": i, "f": 1.5})
        comp.append({"seq": i, "op": "chat", "name": "m", "hash": f"h{i}", "run_id": "r",
                     "span_id": f"s{i}", "started_at": "2026-11-13T00:00:00Z",
                     "request": {"messages": msgs}, "response": {"t": i}})  # fmt: skip
        msgs.append({"role": "assistant", "content": f"a{i}"})
    assert (tmp_path / "comp.jsonl").read_bytes() == (FIX / "comp.jsonl").read_bytes()
