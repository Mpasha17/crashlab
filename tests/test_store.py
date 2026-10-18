from __future__ import annotations

import json
import re

import crashlab
from crashlab.capture.normalize import canonical_json, request_hash
from crashlab.store import Store
from crashlab.store.cassette import read_lines
from crashlab.store.redact import REDACTED, Redactor


def test_canonical_json_and_hash_are_key_order_independent():
    a = {"b": 1, "a": {"y": [1, 2], "x": "é"}}
    b = {"a": {"x": "é", "y": [1, 2]}, "b": 1}
    assert canonical_json(a) == canonical_json(b) == '{"a":{"x":"é","y":[1,2]},"b":1}'
    assert request_hash(a) == request_hash(b)
    assert re.fullmatch(r"[0-9a-f]{64}", request_hash(a))
    assert request_hash(a) != request_hash({"b": 2, "a": {"y": [1, 2], "x": "é"}})


def test_redactor_defaults_and_hooks():
    r = Redactor()
    data = {
        "headers": {"Authorization": "Bearer abcdefghijklmnop", "X-Api-Key": "k123"},
        "api_key": "sk-proj-abcdefghijklmnopqrstuvwxyz",
        "max_tokens": 100,
        "text": "my key is sk-abcdefghijklmnopqrstuv and token Bearer zzzzzzzzzzzz ok",
    }
    out = r.redact(data)
    assert out["headers"] == {"Authorization": REDACTED, "X-Api-Key": REDACTED}
    assert out["api_key"] == REDACTED
    assert out["max_tokens"] == 100
    assert "sk-abc" not in out["text"] and "zzzz" not in out["text"]
    assert out["text"].startswith("my key is [REDACTED]")

    r.add_pattern(r"\b\d{4}-\d{4}-\d{4}-\d{4}\b")
    r.add_hook(lambda s: s.replace("Alice", "<name>"))
    r.add_key("ssn")
    out = r.redact({"note": "Alice paid with 4111-1111-1111-1111", "ssn": "123"})
    assert out == {"note": "<name> paid with [REDACTED]", "ssn": REDACTED}


def test_record_context_manager_writes_store(crashlab_dir):
    @crashlab.tool
    def add(a: int, b: int = 2) -> int:
        return a + b

    with crashlab.record("calc", input={"q": "1+2"}) as run:
        assert add(1) == 3
        run.output = 3
        run.outcome = "pass"

    store = Store()
    assert store.root == crashlab_dir.resolve()
    assert (crashlab_dir / "index.sqlite").exists()
    rows = store.list_runs()
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run.run_id
    assert (row["name"], row["status"], row["outcome"], row["steps"]) == ("calc", "ok", "pass", 1)
    run_json = json.loads((crashlab_dir / "runs" / run.run_id / "run.json").read_text())
    assert run_json["output"] == 3

    lines = read_lines(store.cassette_path(run.run_id))
    assert len(lines) == 1
    assert list(lines[0])[:4] == ["seq", "op", "name", "hash"]
    assert lines[0]["seq"] == 0 and lines[0]["op"] == "execute_tool"
    assert lines[0]["request"] == {"args": {"a": 1, "b": 2}}
    assert lines[0]["response"] == {"result": 3}

    loaded = crashlab.load("last")
    assert loaded.run_id == run.run_id
    assert loaded.steps[0].args == {"a": 1, "b": 2}
    assert crashlab.load(run.run_id[:20].lower()).run_id == run.run_id


def test_run_with_exception_is_error(crashlab_dir):
    try:
        with crashlab.record("boom"):
            raise RuntimeError("kaput")
    except RuntimeError:
        pass
    run = crashlab.load("last")
    assert run.status == "error" and run.outcome == "fail"
    assert run.error.type == "RuntimeError" and run.error.message == "kaput"
    assert run.ended_at is not None


def test_tool_outside_run_is_passthrough(crashlab_dir):
    @crashlab.tool()
    def f(x):
        return x * 2

    assert f(2) == 4
    assert Store().list_runs() == []


def test_load_missing_raises(crashlab_dir):
    import pytest

    with pytest.raises(crashlab.RunNotFoundError):
        crashlab.load("last")


def test_nested_record_rejected():
    import pytest

    with crashlab.record("outer"), pytest.raises(RuntimeError), crashlab.record("inner"):
        pass


def test_fork_mode_not_implemented_and_replay_needs_replay_api():
    import pytest

    with pytest.raises(NotImplementedError):
        crashlab.record("x", mode="fork")
    with pytest.raises(RuntimeError, match="crashlab.replay"), crashlab.record("x", mode="replay"):
        pass
