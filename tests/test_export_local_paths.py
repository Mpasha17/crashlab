"""Exported cassettes replace local paths with ``$PWD`` / ``$CRASHLAB_DIR`` ... tokens.
Those tokens also landed in recorded *responses*: a replayed model told the agent to read
``$PWD/data.txt``, the tool call no longer matched the recording and the exported test
failed with ``differing fields: request_hash``. Import puts this machine's paths back."""

from __future__ import annotations

import json
import os
from pathlib import Path

import crashlab
from crashlab.export import export_test
from crashlab.store import Store, import_run
from tests.fakes import FakeOpenAI, text_response, tool_call_response


@crashlab.tool()
def read_file(path: str) -> str:
    return f"contents of {Path(path).name}"


@crashlab.agent(name="pathbot")
def pathbot() -> str:
    here = os.path.join(os.getcwd(), "data.txt")
    client = crashlab.wrap(
        FakeOpenAI([tool_call_response(1, "read_file", {"path": here}), text_response(2, "ok")])
    )
    msgs = [{"role": "user", "content": f"summarise {here}"}]
    r = client.chat.completions.create(model="fake-model", messages=msgs)
    args = json.loads(r.choices[0].message.tool_calls[0].function.arguments)
    out = read_file(args["path"])
    msgs += [{"role": "tool", "content": out, "tool_call_id": "call_1"}]
    return (
        client.chat.completions.create(model="fake-model", messages=msgs).choices[0].message.content
    )


def _export_and_import(tmp_path, monkeypatch, project: Path, replay_in: Path):
    project.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(project)
    pathbot()
    written = export_test("last", out_dir=project / "tests" / "crashlab")
    cassette = next(p for p in written if p.suffix == ".jsonl")
    text = cassette.read_text()
    assert str(project) not in text and "$PWD/data.txt" in text
    replay_in.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(replay_in)
    fresh = Store(tmp_path / "fresh-store")
    run_id, _ = import_run(fresh, cassette, replace=True)
    return fresh, run_id


def test_exported_run_with_project_paths_replays(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    fresh, run_id = _export_and_import(tmp_path, monkeypatch, project, project)
    run = fresh.load_run(run_id)
    assert run.steps[1].request["args"]["path"] == str(project / "data.txt")
    crashlab.replay(run_id, store_dir=fresh.root)


def test_exported_run_replays_from_another_checkout(tmp_path, monkeypatch):
    """CI checks the repo out elsewhere: the paths are this checkout's, hashes follow."""
    fresh, run_id = _export_and_import(
        tmp_path, monkeypatch, tmp_path / "dev" / "proj", tmp_path / "ci" / "checkout"
    )
    crashlab.replay(run_id, store_dir=fresh.root)


def test_token_text_already_in_the_recording_is_not_expanded(tmp_path, monkeypatch):
    """A shell agent that really sent ``$PWD`` keeps it: only tokens export introduced
    are turned back into paths."""
    monkeypatch.chdir(tmp_path)

    @crashlab.tool()
    def sh(cmd: str) -> str:
        return f"ran {cmd} in {os.getcwd()}"

    @crashlab.agent(name="shellbot")
    def shellbot() -> str:
        return sh("ls $PWD")

    shellbot()
    cassette = next(p for p in export_test("last", out_dir=tmp_path / "t") if p.suffix == ".jsonl")
    fresh = Store(tmp_path / "fresh-store")
    run_id, _ = import_run(fresh, cassette, replace=True)
    step = fresh.load_run(run_id).steps[0]
    assert step.request["args"]["cmd"] == "ls $PWD"


def test_export_test_last_after_a_replay_exports_the_original(tmp_path, monkeypatch):
    """`crashlab replay last` then `crashlab export-test last` failed: 'is a fork/replay'."""
    monkeypatch.chdir(tmp_path)
    pathbot()
    original = crashlab.load("last")
    crashlab.replay(original)
    assert crashlab.load("last").replay_of == original.run_id
    written = export_test("last", out_dir=tmp_path / "t")
    assert any(original.run_id[:10].lower() in p.name for p in written)


def test_export_test_of_a_fork_by_id_still_refused(tmp_path, monkeypatch):
    import pytest

    from crashlab.export import ExportError

    monkeypatch.chdir(tmp_path)
    pathbot()
    replayed = crashlab.replay(crashlab.load("last"))
    with pytest.raises(ExportError, match="fork/replay"):
        export_test(replayed.run_id, out_dir=tmp_path / "t")
