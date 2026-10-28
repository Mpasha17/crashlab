"""Store robustness: odd strings and damaged files must not crash the recorded agent."""

from __future__ import annotations

import pytest

import crashlab
from crashlab.capture.normalize import request_hash


@crashlab.tool()
def fetch_half_emoji(q: str) -> str:
    return "half an emoji: \ud83d"


@crashlab.agent(name="surrogates")
def surrogate_agent(text: str) -> str:
    return fetch_half_emoji(text) + text


def test_lone_surrogates_do_not_crash_the_agent():
    out = surrogate_agent("in\udc00put")
    assert out == "half an emoji: \ud83din\udc00put"
    run = crashlab.load("last")
    assert run.status == "ok"
    assert run.input == {"text": "in\\udc00put"}
    assert run.steps[0].result == "half an emoji: \\ud83d"
    assert crashlab.replay(run).status == "ok"


def test_request_hash_accepts_surrogates():
    assert len(request_hash({"messages": [{"content": "x\ud83d"}]})) == 64


@crashlab.agent(name="plain")
def plain_agent(x: int) -> int:
    return x + 1


def test_unusable_store_dir_does_not_crash_the_agent(tmp_path, monkeypatch):
    not_a_dir = tmp_path / "a-file"
    not_a_dir.write_text("x")
    monkeypatch.setenv("CRASHLAB_DIR", str(not_a_dir))
    with pytest.warns(RuntimeWarning, match="runs unrecorded"):
        assert plain_agent(1) == 2


def test_corrupt_index_is_a_clean_store_error(tmp_path, monkeypatch, capsys):
    import sys

    from crashlab import cli
    from crashlab.store import StoreError

    store = tmp_path / "s"
    monkeypatch.setenv("CRASHLAB_DIR", str(store))
    plain_agent(1)
    (store / "index.sqlite").write_bytes(b"not a database" * 50)
    for suffix in ("-wal", "-shm"):
        (store / f"index.sqlite{suffix}").unlink(missing_ok=True)
    with pytest.raises(StoreError, match="not a usable crashlab index"):
        crashlab.Store().list_runs()
    with pytest.warns(RuntimeWarning, match="runs unrecorded"):
        assert plain_agent(2) == 3  # the agent still runs
    monkeypatch.setattr(sys, "argv", ["crashlab", "ls"])
    with pytest.raises(SystemExit) as info:
        cli.run()
    assert info.value.code == 2
    assert "not a usable crashlab index" in " ".join(capsys.readouterr().out.split())


def test_damaged_cassette_lines_are_skipped(crashlab_dir):
    plain_agent(1)
    fetch_half_emoji("a")  # not recorded: outside a run
    surrogate_agent("ok")
    run = crashlab.load("last")
    path = crashlab.Store().cassette_path(run.run_id)
    good = path.read_text("utf-8")
    path.write_text("{garbage\n" + good + good.splitlines()[0][:30], "utf-8")  # + truncated
    with pytest.warns(RuntimeWarning, match="skipped 2 damaged line"):
        again = crashlab.load(run.run_id)
    assert len(again.steps) == 1


@crashlab.tool()
def step_tool(x: int) -> int:
    return x


@crashlab.agent(name="disk-fails")
def disk_fails_agent(n: int) -> int:
    step_tool(1)
    run_id = crashlab.Store().list_runs(limit=1)[0]["run_id"]
    crashlab.Store().cassette_path(run_id).chmod(0o444)  # the disk "fails" mid-run
    return step_tool(2) + step_tool(3)


def test_write_failure_mid_run_does_not_crash_the_agent(crashlab_dir):
    import os

    if os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    with pytest.warns(RuntimeWarning, match="stopped recording run"):
        assert disk_fails_agent(1) == 5
    run = crashlab.load("last")
    assert "recording-incomplete" in run.tags and run.status == "ok"
    assert len(run.steps) == 1
