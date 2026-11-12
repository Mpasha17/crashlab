"""Safe append-only compact writing: the prefix kept is decided by comparing JSON text
(never object identity, never a fingerprint). Mutations must still match a deep-copied
full baseline. No wall-clock assertions: structural only.
"""

from __future__ import annotations

import base64
import gc
import tracemalloc
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from crashlab.store import compact
from crashlab.store.cassette import CassetteWriter, read_cassette


def _line(seq: int, msgs: list, **extra) -> dict:
    d = {"seq": seq, "op": "chat", "name": "m", "hash": f"h{seq}", "run_id": "r",
         "span_id": f"s{seq}", "request": {"messages": msgs}, "response": {"t": seq}}  # fmt: skip
    d.update(extra)
    return d


def _dump(path: Path):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return [s.model_dump(mode="json", exclude={"started_at"}) for s in read_cassette(path)]


class Pair:
    def __init__(self, tmp: Path, monkeypatch):
        monkeypatch.delenv(compact.ENV, raising=False)
        self.full = CassetteWriter(tmp / "full.jsonl")
        monkeypatch.setenv(compact.ENV, "1")
        self.comp = CassetteWriter(tmp / "comp.jsonl")
        self.tmp = tmp
        self.seq = 0

    def add(self, msgs, **extra):
        self.full.append(_line(self.seq, deepcopy(msgs), **extra))
        self.comp.append(_line(self.seq, msgs, **extra))
        self.seq += 1

    def check(self):
        assert _dump(self.tmp / "comp.jsonl") == _dump(self.tmp / "full.jsonl")


SAMPLES = [
    None, True, False, 0, 1, 12, -7, 1.5, 15, 0.0, -0.0, float("nan"), float("inf"),
    float("-inf"), "", "hi", 'quote"here', "line\nbreak", "你好", "a, b]", [1, 2], (1, 2),
    {"role": "user", "content": "x"}, {"role": "user", "content": "x", "meta": {"n": [1, 2]}},
    {"content": "x", "role": "user"}, {"when": datetime(2026, 1, 2, tzinfo=timezone.utc)},
    {"b64": base64.b64encode(b"\x00\xff" * 100).decode()}, [1, 2, {"a": "b"}], {"b": b"hi"},
    {"b": "b'hi'"},
]  # fmt: skip


def _brute_keep(prev: list, new: list) -> int:
    k = 0
    for a, b in zip(prev, new, strict=False):
        if compact._dumps_msg(a) != compact._dumps_msg(b):
            break
        k += 1
    return k


def _keep_via_writer(prev: list, new: list) -> int:
    c = compact.Compactor()
    c.compact(_line(0, prev))
    out = c.compact(_line(1, new))
    ref = out.get("compact", {}).get("request.messages")
    return ref["keep"] if ref else 0


def test_kept_prefix_equals_message_by_message_comparison():
    """Every pair of values, as a changed middle message: the writer keeps exactly the
    leading messages whose encodings are equal (numbers "1" vs "12", NaN, -0.0, tuples,
    bytes vs their str() form, key order)."""
    for a in SAMPLES:
        for b in SAMPLES:
            for prev, new in (
                (["x", a, "y"], ["x", b, "y", "z"]),
                ([a], [b, "z"]),
                ([a, a], [a, b]),
                ([a], [b]),
                ([], [b]),
            ):
                assert _keep_via_writer(prev, new) == _brute_keep(prev, new), (prev, new)


@pytest.mark.parametrize("seed", range(20))
def test_random_list_edits_keep_matches_brute_force(seed):
    import random

    rng = random.Random(seed)
    prev = [rng.choice(SAMPLES) for _ in range(rng.randrange(0, 12))]
    for _ in range(30):
        new = list(prev)
        op = rng.randrange(4)
        if op == 0 or not new:
            new.append(rng.choice(SAMPLES))
        elif op == 1:
            new[rng.randrange(len(new))] = rng.choice(SAMPLES)
        elif op == 2:
            new.pop(rng.randrange(len(new)))
        else:
            new.insert(rng.randrange(len(new) + 1), rng.choice(SAMPLES))
        assert _keep_via_writer(prev, new) == _brute_keep(prev, new), (prev, new)
        prev = new


def test_large_prefix_is_not_reencoded_message_by_message(monkeypatch, tmp_path):
    """Append-only steps encode only the new tail message by message; the large prefix
    goes through the one whole-list encoding (which is what makes mutations visible)."""
    monkeypatch.setenv(compact.ENV, "1")
    blob = base64.b64encode(bytes(range(256)) * 200).decode()  # ~50KB+
    w = CassetteWriter(tmp_path / "c.jsonl")
    msgs = [{"role": "user", "content": blob, "meta": {"k": "v"}}]
    w.append(_line(0, msgs))
    real = compact._dumps_msg
    seen: list[str] = []

    def counting(m):
        seen.append(str(m.get("content", ""))[:8] if isinstance(m, dict) else repr(m))
        return real(m)

    with mock.patch.object(compact, "_dumps_msg", side_effect=counting):
        for i in range(1, 6):
            msgs = msgs + [{"role": "assistant", "content": f"ack{i}"}]
            w.append(_line(i, msgs))
    assert seen == [f"ack{i}" for i in range(1, 6)], seen
    steps = _dump(tmp_path / "c.jsonl")
    assert steps[-1]["request"]["messages"][0]["content"] == blob
    assert '"keep": 1' in (tmp_path / "c.jsonl").read_text().splitlines()[1]


def test_mutating_large_prefix_is_visible(monkeypatch, tmp_path):
    pw = Pair(tmp_path, monkeypatch)
    blob = base64.b64encode(b"\xff\x00" * 5000).decode()
    m0 = {"role": "user", "content": blob, "meta": {"n": [1]}}
    msgs = [m0]
    pw.add(msgs)
    m0["content"] = base64.b64encode(b"\x11\x22" * 5000).decode()
    m0["meta"]["n"].append(2)
    msgs = msgs + [{"role": "assistant", "content": "ok"}]
    pw.add(msgs)
    pw.check()
    assert _dump(tmp_path / "comp.jsonl")[1]["request"]["messages"][0]["meta"] == {"n": [1, 2]}


def test_datetime_and_custom_objects_roundtrip(monkeypatch, tmp_path):
    class Tag:
        def __init__(self, s):
            self.s = s

        def __str__(self):
            return f"Tag({self.s})"

    pw = Pair(tmp_path, monkeypatch)
    msgs = [{"role": "user", "content": "hi",
             "when": datetime(2026, 5, 6, 7, 8, tzinfo=timezone.utc),
             "tag": Tag("a")}]  # fmt: skip
    pw.add(msgs)
    msgs = msgs + [{"role": "assistant", "content": "yo", "tag": Tag("b")}]
    pw.add(msgs)
    pw.check()
    assert "2026-05-06" in _dump(tmp_path / "comp.jsonl")[0]["request"]["messages"][0]["when"]
    assert "Tag(a)" in _dump(tmp_path / "comp.jsonl")[0]["request"]["messages"][0]["tag"]


def test_cache_cleanup_across_1000_runs_no_leak(monkeypatch, tmp_path):
    """One writer, 1000 short runs: cache stays one entry/field; RSS growth bounded."""
    monkeypatch.setenv(compact.ENV, "1")
    w = CassetteWriter(tmp_path / "c.jsonl")
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    for run in range(1000):
        msgs = []
        for i in range(5):
            msgs = msgs + [{"role": "user", "content": f"r{run}q{i} " + ("y" * 200)}]
            w.append(_line(run * 10 + i, msgs))
        assert list(w._compactor._last) == ["request.messages"]
        _seq, text, ends = w._compactor._last["request.messages"]
        assert ends is not None and len(ends) == 5 and text.endswith("]")
    gc.collect()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # 1000 runs × 5 small msgs must not pin hundreds of MB in the cache
    growth = peak - base
    assert growth < 30 * 1024 * 1024, growth


def test_redaction_then_compact_reuse_safe(monkeypatch, tmp_path):
    import crashlab
    from crashlab.store import redact
    from tests.fakes import FakeOpenAI, text_response

    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path / "s"))
    monkeypatch.setenv(compact.ENV, "1")
    redact.reset()
    big = base64.b64encode(b"z" * 8000).decode()

    @crashlab.agent(name="big-redact")
    def agent() -> str:
        c = crashlab.wrap(FakeOpenAI([text_response(i, f"a{i}") for i in range(6)]))
        msgs = [{"role": "system", "content": big}]
        out = ""
        for i in range(4):
            msgs.append({"role": "user", "content": f"q{i}"})
            out = (
                c.chat.completions.create(model="m", messages=list(msgs)).choices[0].message.content
            )
            msgs.append({"role": "assistant", "content": out})
            if i == 1:
                redact.add_key(r"^content$")
        return out

    agent()
    redact.reset()
    run = crashlab.load("last")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        again = crashlab.load(run.run_id)
    assert len(again.steps) == len(run.steps)
    assert not any("could not be expanded" in str(w.message) for w in rec)
