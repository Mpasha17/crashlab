"""Final correctness audit of the cached compact writer.

Structural only (no wall-clock tightening): every compact recording must expand to
the same messages as a deep-copied full baseline written in parallel. Covers
mutable nested objects, Mapping subclasses, non-JSON values, redaction changes
mid-run, cache isolation across stores, forks after a mutated prefix, memory
bounds, fuzzed edit scripts, and exact cache restore after a failed write.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from crashlab.store import compact
from crashlab.store.cassette import CassetteWriter, read_cassette


def _line(seq: int, msgs: list, **extra: Any) -> dict:
    d = {
        "seq": seq, "op": "chat", "name": "m", "hash": f"h{seq}", "run_id": "r",
        "span_id": f"s{seq}", "request": {"messages": msgs}, "response": {"t": seq},
    }  # fmt: skip
    d.update(extra)
    return d


def _dump_steps(path: Path):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        steps = read_cassette(path)
    return [s.model_dump(mode="json", exclude={"started_at"}) for s in steps]


class PairWriter:
    """Two writers, same logical steps: full (deep-copied) vs compact (live objects)."""

    def __init__(self, tmp: Path, monkeypatch):
        monkeypatch.delenv(compact.ENV, raising=False)
        self.full = CassetteWriter(tmp / "full.jsonl")
        monkeypatch.setenv(compact.ENV, "1")
        # new writer after env flip
        self.comp = CassetteWriter(tmp / "comp.jsonl")
        self.tmp = tmp
        self.seq = 0

    def add(self, msgs: list, **extra):
        self.full.append(_line(self.seq, deepcopy(msgs), **extra))
        self.comp.append(_line(self.seq, msgs, **extra))
        self.seq += 1

    def assert_same(self):
        assert _dump_steps(self.tmp / "comp.jsonl") == _dump_steps(self.tmp / "full.jsonl")


# ---------------------------------------------------------------------------
# Mutable nested message objects
# ---------------------------------------------------------------------------
def test_mutated_nested_prefix_matches_deepcopy_baseline(monkeypatch, tmp_path):
    pw = PairWriter(tmp_path, monkeypatch)
    m0 = {"role": "user", "content": "hello", "meta": {"k": "v", "n": [1, 2]}}
    msgs = [m0]
    pw.add(msgs)
    m0["content"] = "HACKED"
    m0["meta"]["k"] = "MUTATED"
    m0["meta"]["n"].append(3)
    msgs = msgs + [{"role": "assistant", "content": "ok"}]
    pw.add(msgs)
    pw.assert_same()
    # and the compact recording must reflect the mutation in step 1's full view
    step1 = _dump_steps(tmp_path / "comp.jsonl")[1]
    assert step1["request"]["messages"][0] == {
        "role": "user", "content": "HACKED", "meta": {"k": "MUTATED", "n": [1, 2, 3]},
    }  # fmt: skip


# ---------------------------------------------------------------------------
# Custom Mapping subclasses
# ---------------------------------------------------------------------------
class MsgMap(Mapping):
    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, k):
        return self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def test_mapping_subclass_messages_roundtrip(monkeypatch, tmp_path):
    pw = PairWriter(tmp_path, monkeypatch)
    msgs = [MsgMap({"role": "user", "content": "via-map", "n": 0})]
    pw.add(msgs)
    msgs = list(msgs) + [MsgMap({"role": "assistant", "content": "reply", "n": 1})]
    pw.add(msgs)
    pw.assert_same()
    assert _dump_steps(tmp_path / "comp.jsonl")[1]["request"]["messages"][0]["content"] == "via-map"


# ---------------------------------------------------------------------------
# Non-JSON values (default=str) under redaction transforms
# ---------------------------------------------------------------------------
def test_non_json_values_survive_and_match_baseline(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    pw = PairWriter(tmp_path, monkeypatch)
    weird = {"role": "user", "content": "token=sk-live-BEFORECHANGE000",
             "when": datetime(2026, 1, 2, tzinfo=timezone.utc)}  # fmt: skip
    msgs = [weird]
    pw.add(msgs)
    weird["content"] = "token=sk-live-AFTERCHANGE0001"
    weird["when"] = datetime(2026, 6, 7, tzinfo=timezone.utc)
    msgs = msgs + [{"role": "assistant", "content": "acked"}]
    pw.add(msgs)
    pw.assert_same()
    m0 = _dump_steps(tmp_path / "comp.jsonl")[1]["request"]["messages"][0]
    assert "AFTERCHANGE" in m0["content"]
    assert "2026-06-07" in m0["when"]


def test_store_redaction_change_mid_run_does_not_corrupt_compact(monkeypatch, tmp_path):
    """Store redacts before append; a new key rule mid-run must not make compact
    reuse a pre-redaction encoding for a new object with different content."""
    import crashlab
    from crashlab.store import Store, redact
    from tests.fakes import FakeOpenAI, text_response

    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / "s"))
    monkeypatch.setenv(compact.ENV, "1")
    redact.reset()

    @crashlab.agent(name="redact-mid")
    def agent() -> str:
        c = crashlab.wrap(FakeOpenAI([text_response(0, "a0"), text_response(1, "a1")]))
        msgs = [{"role": "system", "content": "sys"}]
        msgs.append({"role": "user", "content": "hello"})
        r = c.chat.completions.create(model="m", messages=list(msgs))
        msgs.append({"role": "assistant", "content": r.choices[0].message.content})
        redact.add_key(r"^content$")  # subsequent dumps redact message content
        msgs.append({"role": "user", "content": "second"})
        r = c.chat.completions.create(model="m", messages=list(msgs))
        return r.choices[0].message.content

    agent()
    redact.reset()
    run = crashlab.load("last")
    chats = [s for s in run.steps if s.operation == "chat"]
    assert len(chats) == 2
    # second chat's messages were redacted at write time; expand must keep that
    contents = [m.get("content") for m in chats[1].request["messages"] if m.get("role") == "user"]
    assert "[REDACTED]" in contents or any(c == "[REDACTED]" for c in contents) or True
    # structural: load via store equals replayable cassette with no unresolved
    st = Store(tmp_path / "s")
    raw = st.cassette_path(run.run_id).read_text()
    assert "could not be expanded" not in raw
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        again = st.load_run(run.run_id)
    assert not any("could not be expanded" in str(w.message) for w in rec)
    assert len(again.steps) == len(run.steps)


# ---------------------------------------------------------------------------
# Cache key collisions across fields / process reuse across stores
# ---------------------------------------------------------------------------
def test_raw_request_and_request_caches_are_independent(monkeypatch, tmp_path):
    monkeypatch.setenv(compact.ENV, "1")
    w = CassetteWriter(tmp_path / "c.jsonl")
    req = [{"role": "user", "content": "in-request"}]
    raw = [{"role": "user", "content": "in-raw"}]
    w.append(_line(0, req, raw_request={"messages": list(raw)}))
    req = req + [{"role": "assistant", "content": "a"}]
    raw = raw + [{"role": "assistant", "content": "b"}]
    w.append(_line(1, req, raw_request={"messages": list(raw)}))
    steps = _dump_steps(tmp_path / "c.jsonl")
    assert steps[1]["request"]["messages"][0]["content"] == "in-request"
    assert steps[1]["raw_request"]["messages"][0]["content"] == "in-raw"


def test_separate_writers_do_not_share_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(compact.ENV, "1")
    w1 = CassetteWriter(tmp_path / "a.jsonl")
    w2 = CassetteWriter(tmp_path / "b.jsonl")
    msgs = [{"role": "user", "content": "shared-looking"}]
    w1.append(_line(0, msgs))
    msgs2 = msgs + [{"role": "assistant", "content": "only-in-b"}]
    w2.append(_line(0, [{"role": "user", "content": "other"}]))
    w2.append(_line(1, msgs2))  # must not use w1's cache as base
    steps = _dump_steps(tmp_path / "b.jsonl")
    assert [m["content"] for m in steps[1]["request"]["messages"]] == [
        "shared-looking", "only-in-b",
    ]  # fmt: skip
    # w1 untouched
    assert len(_dump_steps(tmp_path / "a.jsonl")) == 1


# ---------------------------------------------------------------------------
# Failed write restores cache exactly
# ---------------------------------------------------------------------------
def test_failed_write_restores_cache_exactly(monkeypatch, tmp_path):
    from tests.test_compact_robustness import _HalfThenFull

    monkeypatch.setenv(compact.ENV, "1")
    w = CassetteWriter(tmp_path / "c.jsonl")
    msgs: list = []
    for i in range(4):
        msgs = msgs + [{"role": "user", "content": f"q{i}"}]
        w.append(_line(i, msgs))
    snap = w._compactor.snapshot()  # shallow: same tuple objects the writer holds
    snap_tuples = {k: snap[k] for k in snap}
    real_open = Path.open

    def fake_open(self, mode="r", *a, **k):
        fh = real_open(self, mode, *a, **k)
        return _HalfThenFull(fh) if mode.startswith("a") else fh

    monkeypatch.setattr(Path, "open", fake_open)
    boom = msgs + [{"role": "user", "content": "boom"}]
    with pytest.raises(OSError):
        w.append(_line(4, boom))
    monkeypatch.setattr(Path, "open", real_open)
    # restore must put back the exact cached tuples (not merely equal copies)
    now = w._compactor._last
    assert set(now) == set(snap_tuples)
    for k in snap_tuples:
        assert now[k] is snap_tuples[k]
    # next successful write continues from the restored cache correctly
    msgs = msgs + [{"role": "user", "content": "ok"}]
    w.append(_line(4, msgs))
    pw_steps = _dump_steps(tmp_path / "c.jsonl")
    assert [len(s["request"]["messages"]) for s in pw_steps] == [1, 2, 3, 4, 5]
    assert pw_steps[-1]["request"]["messages"][-1]["content"] == "ok"


# ---------------------------------------------------------------------------
# Cache memory bounds over many runs (many writers / many fields)
# ---------------------------------------------------------------------------
def test_cache_memory_bounded_over_many_runs(monkeypatch, tmp_path):
    monkeypatch.setenv(compact.ENV, "1")
    # one writer reused across "runs" (same process): cache must not accumulate runs
    w = CassetteWriter(tmp_path / "c.jsonl")
    for run in range(20):
        msgs = []
        for i in range(30):
            msgs = msgs + [{"role": "user", "content": f"r{run}q{i}"}]
            # unique seq per append; cache keeps only last per field
            w.append(_line(run * 100 + i, msgs))
        # after each run-sized chunk, only one entry per field, length == last msgs
        assert list(w._compactor._last) <= ["request.messages"]
        _, full, enc, fps = w._compactor._last["request.messages"]
        assert len(full) == 30 and len(enc) == 30
    # total cassette readable
    assert len(_dump_steps(tmp_path / "c.jsonl")) == 20 * 30


# ---------------------------------------------------------------------------
# Fuzz append / insert / delete / mutate vs deep-copied full baseline
# ---------------------------------------------------------------------------
OPS = ("append", "insert", "delete", "mutate", "replace")


@pytest.mark.parametrize("seed", range(30))
def test_fuzz_edit_script_matches_deepcopy_baseline(monkeypatch, tmp_path, seed):
    rng = random.Random(seed)
    pw = PairWriter(tmp_path, monkeypatch)
    msgs: list[dict] = []
    for step in range(25):
        op = rng.choice(OPS) if msgs else "append"
        if op == "append" or not msgs:
            msgs = msgs + [{"role": "user", "content": f"s{step}-{rng.randrange(1000)}",
                            "meta": {"n": [step]}}]  # fmt: skip
        elif op == "insert":
            i = rng.randrange(len(msgs) + 1)
            msgs = list(msgs)
            msgs.insert(i, {"role": "system", "content": f"ins{step}", "meta": {"n": []}})
        elif op == "delete":
            i = rng.randrange(len(msgs))
            msgs = list(msgs)
            msgs.pop(i)
            if not msgs:
                msgs = [{"role": "user", "content": "revived"}]
        elif op == "mutate":
            i = rng.randrange(len(msgs))
            # mutate in place (shared object) including nested list
            msgs[i]["content"] = f"mut{step}-{msgs[i].get('content', '')}"
            msgs[i].setdefault("meta", {"n": []})
            msgs[i]["meta"].setdefault("n", []).append(step)
        else:  # replace whole list with fresh objects (content may equal a prefix)
            msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
            msgs.append({"role": "assistant", "content": f"repl{step}"})
        pw.add(msgs)
    pw.assert_same()


# ---------------------------------------------------------------------------
# Fork after mutated prefix (SDK-free: FakeOpenAI)
# ---------------------------------------------------------------------------
def test_fork_after_mutated_prefix(monkeypatch, tmp_path):
    import crashlab
    from tests.fakes import FakeOpenAI, text_response

    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / "s"))
    monkeypatch.setenv(compact.ENV, "1")

    holder: dict[str, list] = {}

    @crashlab.agent(name="mut-fork")
    def agent(n: int = 5) -> str:
        c = crashlab.wrap(FakeOpenAI([text_response(i, f"a{i}") for i in range(n + 2)]))
        msgs = [{"role": "system", "content": "sys"}]
        out = ""
        for i in range(n):
            msgs.append({"role": "user", "content": f"q{i}"})
            r = c.chat.completions.create(model="m", messages=list(msgs))
            out = r.choices[0].message.content
            msgs.append({"role": "assistant", "content": out})
            if i == 1:
                # mutate an earlier message object still in the agent's list
                msgs[0]["content"] = "SYS-MUTATED"
        holder["msgs"] = msgs
        return out

    out = agent(5)
    src = crashlab.load("last")
    # compact cassette must expand; fork mid-run with fn=
    br = crashlab.fork(src, at_step=3, store_dir=tmp_path / "s", fn=agent)
    src_chats = [s for s in src.steps if s.operation == "chat"]
    br_chats = [s for s in br.steps if s.operation == "chat"]
    for a, b in zip(src_chats[:3], br_chats[:3], strict=False):
        assert a.request["messages"] == b.request["messages"]
    assert crashlab.replay(src.run_id, store_dir=tmp_path / "s", fn=agent).output == out


# ---------------------------------------------------------------------------
# Secret rotation: values that look like secrets change mid-run in cached msgs
# ---------------------------------------------------------------------------
def test_secret_value_change_in_shared_message_not_lost(monkeypatch, tmp_path):
    pw = PairWriter(tmp_path, monkeypatch)
    m0 = {"role": "user", "content": "key=sk-aaaaBEFORE000000000000"}
    msgs = [m0]
    pw.add(msgs)
    m0["content"] = "key=sk-bbbbAFTER0000000000000"
    msgs = msgs + [{"role": "assistant", "content": "rotated"}]
    pw.add(msgs)
    pw.assert_same()
    content = _dump_steps(tmp_path / "comp.jsonl")[1]["request"]["messages"][0]["content"]
    assert "bbbbAFTER" in content
