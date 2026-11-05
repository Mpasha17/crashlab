"""Safe append-only compact optimization: content fingerprints (not identity).

Large unchanged prefix messages may reuse immutable cached encodings when their
content fingerprint still matches; small messages use dumps. Mutations must still
match a deep-copied full baseline. No wall-clock assertions — structural only.
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


def test_fingerprint_agrees_with_dumps_equality():
    """fp(a)==fp(b) iff dumps(a)==dumps(b) on a spread of JSON-ish values."""
    from datetime import date

    samples = [
        None,
        True,
        False,
        0,
        1,
        -7,
        1.5,
        0.0,
        "",
        "hi",
        'quote"here',
        "line\nbreak",
        "你好",
        {"role": "user", "content": "x"},
        {"role": "user", "content": "x", "meta": {"n": [1, 2, 3]}},
        {"when": datetime(2026, 1, 2, tzinfo=timezone.utc)},
        {"d": date(2026, 3, 4)},
        {"b64": base64.b64encode(b"\x00\xff" * 100).decode()},
        [1, 2, {"a": "b"}],
    ]
    for i, a in enumerate(samples):
        for b in samples[i:]:
            assert (compact._content_fp(a) == compact._content_fp(b)) == (
                compact._dumps_msg(a) == compact._dumps_msg(b)
            )


def test_large_base64_prefix_reuses_encoding_not_identity(monkeypatch, tmp_path):
    """Unchanged large prefix: dumps of prefix messages must not be repeated."""
    monkeypatch.setenv(compact.ENV, "1")
    blob = base64.b64encode(bytes(range(256)) * 200).decode()  # ~50KB+
    assert len(blob) > compact.FP_THRESHOLD
    w = CassetteWriter(tmp_path / "c.jsonl")
    msgs = [{"role": "user", "content": blob, "meta": {"k": "v"}}]
    w.append(_line(0, msgs))
    real = compact._dumps_msg
    n = {"dumps": 0}

    def counting(m):
        if isinstance(m, dict) and len(str(m.get("content", ""))) > compact.FP_THRESHOLD:
            n["dumps"] += 1
        return real(m)

    with mock.patch.object(compact, "_dumps_msg", side_effect=counting):
        for i in range(1, 6):
            msgs = msgs + [{"role": "assistant", "content": f"ack{i}"}]
            w.append(_line(i, msgs))
    # the large prefix must not be re-serialized on append-only steps
    assert n["dumps"] == 0, n
    steps = _dump(tmp_path / "c.jsonl")
    assert steps[-1]["request"]["messages"][0]["content"] == blob


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
        _seq, _full, enc, fps = w._compactor._last["request.messages"]
        assert len(enc) == 5 and len(fps) == 5
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
