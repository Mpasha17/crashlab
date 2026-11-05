"""Compact cassette format v1 (store/compact.py): always readable, written only with
CRASHLAB_EXPERIMENTAL_COMPACT=1. A compact step that cannot be expanded is kept and
flagged, never dropped and never expanded wrongly without notice."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest

import crashlab
from crashlab.store import Store, compact, export_run, import_run
from crashlab.store.cassette import read_cassette
from crashlab.verify.oracles import run_checks
from tests.fakes import FakeOpenAI, text_response

FIX = Path(__file__).parent / "fixtures" / "v0_0_2"
TURNS = 30


@crashlab.tool()
def note(i: int) -> str:
    return f"noted {i}"


@crashlab.agent(name="compact-chat")
def chat_agent(turns: int = TURNS) -> str:
    c = crashlab.wrap(FakeOpenAI([text_response(i, f"answer {i} " * 20) for i in range(turns)]))
    msgs = [{"role": "system", "content": "be brief " * 30}]
    out = ""
    for i in range(turns):
        msgs.append({"role": "user", "content": f"question {i} " * 20})
        out = c.chat.completions.create(model="fake-model", messages=list(msgs))
        out = out.choices[0].message.content
        msgs.append({"role": "assistant", "content": out})
        note(i)
    return out


def record(monkeypatch, on: bool, store_dir: Path, turns: int = TURNS):
    monkeypatch.setenv("CRASHLAB_DIR", str(store_dir))
    if on:
        monkeypatch.setenv(compact.ENV, "1")
    else:
        monkeypatch.delenv(compact.ENV, raising=False)
    chat_agent(turns)
    run = crashlab.load("last")
    monkeypatch.delenv(compact.ENV, raising=False)
    return run, Store(store_dir).cassette_path(run.run_id)


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def view(steps):
    return [(s.idx, s.operation, s.request_hash, s.request, s.raw_request, s.response)
            for s in steps]  # fmt: skip


@pytest.fixture
def both(monkeypatch, tmp_path):
    full, fpath = record(monkeypatch, False, tmp_path / "full")
    comp, cpath = record(monkeypatch, True, tmp_path / "comp")
    return full, fpath, comp, cpath


def test_default_writer_is_the_old_format(monkeypatch, tmp_path):
    monkeypatch.delenv(compact.ENV, raising=False)
    _, path = record(monkeypatch, False, tmp_path / "s")
    assert not any("compact" in ln for ln in lines(path))


def test_roundtrip_identical_and_smaller(both):
    full, fpath, comp, cpath = both
    assert sum("compact" in ln for ln in lines(cpath)) >= TURNS - 1
    assert view(comp.steps) == view(full.steps)
    # deterministic byte counts, no timing: compact grows ~linearly, full ~quadratically
    assert cpath.stat().st_size < 0.35 * fpath.stat().st_size


def test_replay_export_and_check_compatible(both, tmp_path):
    full, _, comp, cpath = both
    store_dir = cpath.parents[2]
    assert crashlab.replay(comp.run_id, store_dir=store_dir).output == full.output
    out = export_run(Store(store_dir), comp.run_id, tmp_path / "exp.jsonl")
    assert not any("compact" in ln for ln in lines(out)[1:])  # export is always full
    rid, _ = import_run(Store(tmp_path / "imp"), out)
    assert crashlab.replay(rid, store_dir=tmp_path / "imp").output == full.output
    norm = lambda rs: [(r.name, r.passed) for r in rs]  # noqa: E731
    assert norm(run_checks(comp)) == norm(run_checks(full))


def test_import_of_a_raw_compact_cassette(both, tmp_path):
    full, _, comp, cpath = both
    exp = tmp_path / "exp.jsonl"
    export_run(Store(cpath.parents[2]), comp.run_id, exp)
    head = exp.read_text().splitlines()[0]
    raw = tmp_path / "raw.jsonl"
    raw.write_text(head + "\n" + cpath.read_text())
    rid, _ = import_run(Store(tmp_path / "imp"), raw)
    assert view(Store(tmp_path / "imp").load_run(rid).steps) == view(full.steps)


def test_mixed_old_and_new_lines(both, tmp_path):
    full, fpath, comp, cpath = both
    old = lines(fpath)
    new = lines(cpath)
    mixed = [o if i % 3 == 0 else n for i, (o, n) in enumerate(zip(old, new, strict=True))]
    p = tmp_path / "mixed.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in mixed))
    assert view(read_cassette(p)) == view(full.steps)


def test_crash_truncated_last_line(both, tmp_path):
    full, _, _, cpath = both
    text = cpath.read_text()
    p = tmp_path / "c.jsonl"
    p.write_text(text[: len(text) - 40])  # killed mid-write
    with pytest.warns(RuntimeWarning, match="damaged"):
        steps = read_cassette(p)
    assert view(steps) == view(full.steps)[:-1]


def test_damaged_base_keeps_and_flags_dependents(both, tmp_path):
    full, _, _, cpath = both
    raw = cpath.read_text().splitlines()
    chats = [i for i, x in enumerate(raw) if '"op": "chat"' in x]
    raw[chats[3]] = raw[chats[3]][:50]  # a base in the middle is unreadable
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(raw) + "\n")
    with pytest.warns(RuntimeWarning) as rec:
        steps = read_cassette(p)
    msgs = " ".join(str(w.message) for w in rec)
    assert "damaged" in msgs and "could not be expanded" in msgs
    assert len(steps) == len(full.steps) - 1  # only the damaged line itself is lost
    flagged = [s for s in steps if compact.UNRESOLVED_ATTR in s.attributes]
    assert flagged and all(s.operation == "chat" for s in flagged)
    good = {v[0]: v for v in view(full.steps)}
    for s in steps:
        if compact.UNRESOLVED_ATTR not in s.attributes:
            assert view([s])[0] == good[s.idx]


MUTATIONS = [
    lambda r, m: m.update(base=r.randint(-3, 10_000)),
    lambda r, m: m.update(keep=r.choice([-1, 0, 10**6, "3", None, 2.5])),
    lambda r, m: m.update(keep=max(0, m["keep"] + r.choice([-1, 1]))),
    lambda r, m: m.update(check="0" * 16),
    lambda r, m: m.pop("check"),
    lambda r, m: m.pop("base"),
    lambda r, m: m.update(base=None),
]


@pytest.mark.parametrize("seed", range(40))
def test_fuzzed_compact_data_never_crashes_drops_or_lies(both, tmp_path, seed):
    full, _, _, cpath = both
    r = random.Random(seed)
    data = lines(cpath)
    comp_ix = [i for i, ln in enumerate(data) if "compact" in ln]
    for i in r.sample(comp_ix, k=r.randint(1, 4)):
        meta = data[i]["compact"]
        kind = r.random()
        if kind < 0.1:
            meta["v"] = r.choice([2, 0, "1", None])
        elif kind < 0.15:
            data[i]["compact"] = r.choice([None, [], "x", 7])
        elif kind < 0.2:
            meta["request.nope"] = {"base": 0, "keep": 1}
        elif kind < 0.25:  # cycle: point at itself or a later step
            meta["request.messages"]["base"] = data[i]["seq"] + r.choice([0, 2])
        elif kind < 0.3:
            data[i]["request"]["messages"] = r.choice([None, "x", {}])
        else:
            r.choice(MUTATIONS)(r, meta["request.messages"])
    p = tmp_path / "f.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in data))
    with pytest.warns(RuntimeWarning):
        steps = read_cassette(p)
    good = {v[0]: v for v in view(full.steps)}
    for s in steps:
        if compact.UNRESOLVED_ATTR in s.attributes:
            continue
        if isinstance(s.request.get("messages"), list):  # unflagged => exactly right
            assert view([s])[0] == good[s.idx]
    # a step is dropped only if pydantic itself cannot hold it (non-list messages are fine)
    assert len(steps) == len(full.steps)


def test_v0_0_2_fixtures_with_compact_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv(compact.ENV, "1")
    rid, _ = import_run(Store(tmp_path / "s"), FIX / "compat_shop.jsonl")
    assert crashlab.replay(rid, store_dir=tmp_path / "s").output == (
        "ordered 2 A-1; B-2 is out of stock"
    )
    d = tmp_path / "store"
    shutil.copytree(FIX / "store", d)
    run = Store(d).load_run(Store(d).index.list_runs()[0]["run_id"])
    assert len(run.steps) == 5


def test_size_growth_is_linear_with_compact(monkeypatch, tmp_path):
    sizes = []
    for n in (20, 40):
        _, p = record(monkeypatch, True, tmp_path / f"c{n}", turns=n)
        sizes.append(p.stat().st_size)
    assert sizes[1] < 2.6 * sizes[0]  # doubling turns ~doubles size (full: ~3.5x)
