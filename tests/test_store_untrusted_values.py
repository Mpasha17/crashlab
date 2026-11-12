"""Values the store takes from outside: a provider's usage numbers, a malformed field, a
shared export file. None of them may crash the agent, stop the recording, or write
outside the store directory."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

import crashlab
from crashlab.models import Step
from crashlab.store import Store, export_run, import_run
from tests.fakes import FakeOpenAI, text_response

pytestmark = pytest.mark.requires_providers("openai")


def _huge_usage():
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage

    return ChatCompletion(
        id="x", model="fake-model", object="chat.completion", created=0,
        choices=[Choice(index=0, finish_reason="stop",
                        message=ChatCompletionMessage(role="assistant", content="hi"))],
        usage=CompletionUsage(prompt_tokens=2**70, completion_tokens=1, total_tokens=2**70 + 1),
    )  # fmt: skip


@crashlab.agent(name="huge-usage")
def huge_usage_agent() -> str:
    c = crashlab.wrap(FakeOpenAI([_huge_usage(), text_response(1, "ok")]))
    c.chat.completions.create(model="fake-model", messages=[{"role": "user", "content": "q"}])
    out = c.chat.completions.create(model="fake-model", messages=[{"role": "user", "content": "r"}])
    return out.choices[0].message.content


def test_token_count_beyond_64_bits_does_not_crash_the_agent():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert huge_usage_agent() == "ok"
    assert not [w for w in rec if "stopped recording" in str(w.message)]
    run = crashlab.load("last")
    assert len(run.steps) == 2
    assert run.usage.input_tokens == 2**70 + 200  # run.json keeps the exact value
    [row] = Store().list_runs()
    assert row["input_tokens"] == 2**63 - 1  # the index clamps


def _step(i: int, **kw) -> Step:
    base = dict(run_id="r", idx=i, span_id=f"s{i}", operation="chat", name="m",
                request_hash=f"h{i}", started_at="2026-11-13T00:00:00Z")  # fmt: skip
    base.update(kw)
    return Step(**base)


@pytest.mark.parametrize(
    "kw",
    [
        {"attributes": {"gen_ai.response.finish_reasons": [{"reason": "stop"}]}},
        {"attributes": {"gen_ai.response.finish_reasons": "stop"}},
        {"attributes": {"gen_ai.tool.call.id": {"id": 1}}},
        {"attributes": {"gen_ai.usage.input_tokens": "12"}},
        {"attributes": {"gen_ai.usage.output_tokens": [3]}},
        {"attributes": {"gen_ai.usage.input_tokens": -(2**80)}},
        {"duration_ms": float("nan")},
        {"duration_ms": float("inf")},
    ],
    ids=lambda kw: json.dumps(kw, default=str)[:40],
)
def test_malformed_index_fields_are_stored_not_fatal(tmp_path, kw):
    s = Store(tmp_path)
    s.add_step(_step(0, **kw))
    s.add_step(_step(1))
    import sqlite3
    from contextlib import closing

    from crashlab.store.cassette import read_cassette

    assert [x.idx for x in read_cassette(s.cassette_path("r"))] == [0, 1]
    with closing(sqlite3.connect(s.root / "index.sqlite")) as conn:
        assert conn.execute("SELECT idx FROM steps ORDER BY idx").fetchall() == [(0,), (1,)]


def _export(tmp_path: Path) -> tuple[Store, Path, list[dict]]:
    from tests.test_compact_format import chat_agent

    chat_agent(2)
    store = Store()
    run = crashlab.load("last")
    out = tmp_path / "e.jsonl"
    export_run(store, run.run_id, out)
    return store, out, [json.loads(x) for x in out.read_text().splitlines()]


@pytest.mark.parametrize(
    "bad",
    [
        "../escaped",
        "../../escaped",
        "/tmp/abs-escape",
        "a/b",
        "a\\b",
        ".",
        "..",
        "",
        "x\x00y",
        "-rf",
        "C:evil",
        "x" * 200,
    ],  # fmt: skip
)
def test_import_refuses_a_run_id_that_is_not_a_plain_name(tmp_path, bad):
    store, out, lines = _export(tmp_path)
    if bad == "<abs>":
        bad = str(tmp_path / "abs-escape")
    lines[0]["run"]["run_id"] = bad
    for ln in lines[1:]:
        ln["run_id"] = bad
    evil = tmp_path / "evil.jsonl"
    evil.write_text("".join(json.dumps(x) + "\n" for x in lines))
    before = sorted(p for p in tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="not a valid run id"):
        import_run(store, evil)
    after = sorted(p for p in tmp_path.rglob("*"))
    assert after == before  # nothing created anywhere, including tmp_path/abs-escape


def test_import_stores_steps_under_the_runs_own_id(tmp_path):
    store, out, lines = _export(tmp_path)
    rid = lines[0]["run"]["run_id"]
    for ln in lines[1:]:
        ln["run_id"] = "../../elsewhere"
    lines[0]["run"]["run_id"] = rid + "b"
    f = tmp_path / "mixed.jsonl"
    f.write_text("".join(json.dumps(x) + "\n" for x in lines))
    new_id, added = import_run(store, f)
    assert added and new_id == rid + "b"
    assert not list(tmp_path.rglob("elsewhere"))
    run = crashlab.load(new_id)
    assert run.steps and all(s.run_id == new_id for s in run.steps)


@pytest.mark.parametrize("ok", ["01M6PGSAZ3328RE4HF201K6H5Q", "r", "a.b-c_d", "R1"])
def test_ordinary_run_ids_still_work(tmp_path, ok):
    assert Store(tmp_path).run_dir(ok) == Store(tmp_path).root / "runs" / ok
