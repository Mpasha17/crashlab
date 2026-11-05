"""Runs recorded by the released crashlab 0.0.2 (tests/fixtures/v0_0_2, recorded from
tests/compat_agent.py at c5f8982) must load and replay on current main: exported
cassettes, and a whole 0.0.2 store directory (index schema included).

Direction: newer crashlab reads older runs (see docs/compatibility.md for what 0.0.2
makes of newer recordings: main now writes ``error.state`` as an attribute so 0.0.2
keeps those steps; compact lines are opt-in and not readable by 0.0.2)."""

from __future__ import annotations

import shutil
from pathlib import Path

import crashlab
from crashlab.store import Store, import_run

FIX = Path(__file__).parent / "fixtures" / "v0_0_2"


def _check(run):
    assert [s.name for s in run.steps] == [
        "fake-model", "get_stock", "get_stock", "place_order", "fake-model"
    ]  # fmt: skip
    err = run.steps[2].error
    assert err.type == "tests.compat_agent.OutOfStock" and err.state is None
    replayed = crashlab.replay(run.run_id, store_dir=run_store(run))
    assert replayed.output == "ordered 2 A-1; B-2 is out of stock"


_stores: dict[str, Path] = {}


def run_store(run):
    return _stores[run.run_id]


def test_exported_0_0_2_cassette_imports_and_replays(tmp_path):
    store = Store(tmp_path / "s")
    run_id, added = import_run(store, FIX / "compat_shop.jsonl")
    assert added
    _stores[run_id] = store.root
    _check(store.load_run(run_id))


def test_0_0_2_store_directory_loads_and_replays(tmp_path):
    root = tmp_path / "old-store"
    shutil.copytree(FIX / "store", root)
    store = Store(root)
    runs = store.list_runs()
    assert len(runs) == 1
    run = store.load_run("last")
    _stores[run.run_id] = store.root
    _check(run)
    # current main can keep recording into the old store
    assert len(Store(root).list_runs()) >= 2  # the replay run was added


def test_new_error_without_state_has_no_new_key():
    """What keeps steps written by main readable by 0.0.2 when there is no state."""
    from crashlab.models import ErrorInfo

    dumped = ErrorInfo(type="ValueError", message="x").model_dump(mode="json")
    assert set(dumped) == {"type", "message", "is_tool_error", "status_code", "traceback", "body"}
