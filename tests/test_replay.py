"""Strict offline replay: served from the cassette, divergence detection, guards."""

from __future__ import annotations

import os
import socket

import pytest
from examples.refund_bot import agent as refund

import crashlab
from crashlab.replay.engine import PLACEHOLDER_KEY
from crashlab.store import Store
from tests import replay_agents as ag
from tests.fakes import FakeAPIError, FakeOpenAI, refund_script, text_response, tool_call_response


@pytest.fixture(autouse=True)
def _reset():
    ag.reset()
    refund.reset_db()
    yield
    ag.reset()
    refund.reset_db()


def _record_mini(task="t"):
    assert ag.mini(task) == "done"
    return crashlab.load("last")


# --- happy path ----------------------------------------------------------------------


def test_refund_bot_replay_is_identical_and_never_refunds(crashlab_dir):
    refund.run_agent("Refund my order A17", client=crashlab.wrap(FakeOpenAI(refund_script())),
                     model="fake-model", faults=[])  # fmt: skip
    src = crashlab.load("last")
    assert refund.REFUNDS and src.outcome == "pass"
    refund.reset_db()

    rep = crashlab.replay(src, overrides={"client": crashlab.wrap(FakeOpenAI([]))})
    assert refund.REFUNDS == []  # issue_refund (side effects) never ran
    assert rep.run_id != src.run_id and rep.replay_of == src.run_id
    assert "replay" in rep.tags and rep.status == "ok" and rep.outcome == "pass"
    assert [s.source for s in rep.steps] == ["cassette"] * 5
    assert [(s.operation, s.name, s.request_hash) for s in rep.steps] == [
        (s.operation, s.name, s.request_hash) for s in src.steps
    ]
    assert [s.response for s in rep.steps] == [s.response for s in src.steps]
    assert rep.output == src.output
    assert rep.usage.total_tokens == src.usage.total_tokens
    assert rep.steps[3].attributes["gen_ai.tool.call.id"] == "call_2"
    row = Store().list_runs(limit=1)[0]
    assert row["replay_of"] == src.run_id


def test_tools_never_execute_during_replay():
    ag.KNOBS["charge"] = True
    src = _record_mini()
    assert ag.CALLS == {"lookup": 1, "charge": 1}
    ag.CALLS.update(lookup=0, charge=0)
    rep = crashlab.replay(src)
    assert ag.CALLS == {"lookup": 0, "charge": 0}
    charge = rep.tool_steps("charge")[0]
    assert charge.source == "cassette" and charge.result == {"charged": 5.0}
    assert charge.attributes["crashlab.tool.side_effects"] is True


def test_replay_by_id_prefix_and_last():
    src = _record_mini()
    assert crashlab.replay(src.run_id[:20]).replay_of == src.run_id
    assert crashlab.replay("last").replay_of is not None  # "last" is now the replay itself


def test_async_agent_replay():
    import asyncio

    asyncio.run(ag.mini_async("x"))
    src = crashlab.load("last")
    rep = crashlab.replay(src)
    assert rep.output == src.output == "done" and rep.status == "ok"


async def test_replay_called_from_a_running_event_loop():
    await ag.mini_async("y")
    rep = crashlab.replay("last")
    assert rep.status == "ok" and rep.output == "done"


def test_record_block_needs_fn():
    def main():
        client = crashlab.wrap(FakeOpenAI([text_response(1, "hi")]))
        with crashlab.record("scripted") as run:
            run.output = (
                client.chat.completions.create(model="m", messages=[]).choices[0].message.content
            )

    main()
    src = crashlab.load("last")
    with pytest.raises(crashlab.ReplayError, match="fn="):
        crashlab.replay(src)
    assert crashlab.replay(src, fn=main).output == "hi"
    with pytest.raises(crashlab.ReplayError, match="did not start"):
        crashlab.replay(src, fn=lambda: None)


# --- divergence ----------------------------------------------------------------------


def test_request_divergence_has_readable_diff():
    src = _record_mini()
    ag.KNOBS["system"] = "be verbose"
    with pytest.raises(crashlab.DivergenceError) as ei:
        crashlab.replay(src)
    err = ei.value
    assert err.kind == "request" and err.step_idx == 0
    assert err.fields == ["system"]
    assert "\n-be brief\n+be verbose" in err.diff  # raw lines, no quoting
    assert "--- recorded step 0 system" in err.diff
    text = str(err)
    assert "step 0 (chat 'fake-model')" in text and "differing fields: system" in text
    assert err.source_run_id == src.run_id
    # the diverged replay is stored, marked as an error at the diverging step
    rep = crashlab.load(err.replay_run_id)
    assert rep.status == "error" and rep.outcome == "fail" and rep.replay_of == src.run_id
    assert rep.outcome_reason.startswith("replay diverged")
    assert rep.steps[0].error.type.endswith("DivergenceError")


def test_divergence_swallowed_by_user_code_is_still_raised():
    src = _record_mini()
    ag.KNOBS["key"] = "xyz"  # tool args change; mini() catches every Exception around it
    with pytest.raises(crashlab.DivergenceError) as ei:
        crashlab.replay(src)
    assert ei.value.kind == "request" and ei.value.step_idx == 1
    assert ei.value.fields == ["args.key"]
    assert ag.CALLS["lookup"] == 1  # only the recording ran it


def test_step_divergence_other_tool():
    src = _record_mini()
    ag.KNOBS["other_tool"] = True
    with pytest.raises(crashlab.DivergenceError) as ei:
        crashlab.replay(src)
    assert ei.value.kind == "step" and "recording has execute_tool 'lookup'" in str(ei.value)
    assert ag.CALLS["charge"] == 0  # the side-effect tool did not run


def test_missing_and_extra_steps():
    src = _record_mini()
    ag.KNOBS["stop_early"] = True
    with pytest.raises(crashlab.DivergenceError) as ei:
        crashlab.replay(src)
    assert ei.value.kind == "missing_steps" and ei.value.step_idx == 2
    ag.KNOBS.clear()
    ag.KNOBS["extra_call"] = True
    with pytest.raises(crashlab.DivergenceError) as ei:
        crashlab.replay(src)
    assert ei.value.kind == "extra_step" and ei.value.step_idx == 3


def test_output_divergence():
    src = _record_mini()
    ag.KNOBS["suffix"] = "!"
    with pytest.raises(crashlab.DivergenceError) as ei:
        crashlab.replay(src)
    assert ei.value.kind == "output" and ei.value.expected == "done"
    assert ei.value.actual == "done!"


def test_network_is_blocked_during_replay():
    src = _record_mini()
    ag.KNOBS["network"] = True
    with pytest.raises(crashlab.DivergenceError) as ei:
        crashlab.replay(src)
    assert ei.value.kind == "network" and "192.0.2.1" in str(ei.value)
    # the guard is removed afterwards
    assert socket.socket.connect.__qualname__ == "socket.connect"


def test_offline_placeholder_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = {}

    @crashlab.agent(name="cred-probe")
    def probe() -> str:
        seen["key"] = os.environ.get("OPENAI_API_KEY")
        client = crashlab.wrap(FakeOpenAI([text_response(1, "ok")]))
        return client.chat.completions.create(model="m", messages=[]).choices[0].message.content

    probe()
    assert seen["key"] is None
    crashlab.replay("last", fn=probe)
    assert seen["key"] == PLACEHOLDER_KEY
    assert "OPENAI_API_KEY" not in os.environ


# --- recorded errors are reproduced ----------------------------------------------------


def test_recorded_tool_exception_is_reproduced():
    ag.KNOBS["key"] = "missing"
    assert ag.mini("t") == "done"
    src = crashlab.load("last")
    assert src.steps[1].error.type == "KeyError"
    rep = crashlab.replay(src)  # the swallowed KeyError text reaches the next request
    assert rep.steps[1].error.type == "KeyError"
    assert rep.steps[2].request_hash == src.steps[2].request_hash


def test_recorded_llm_error_is_reproduced():
    @crashlab.agent(name="flaky")
    def flaky() -> str:
        client = crashlab.wrap(FakeOpenAI([FakeAPIError("upstream 503", status_code=503)]))
        try:
            client.chat.completions.create(model="m", messages=[])
        except FakeAPIError as exc:
            return f"gave up: {exc} ({exc.status_code})"
        return "?"

    out = flaky()
    rep = crashlab.replay("last", fn=flaky)
    assert rep.output == out == "gave up: upstream 503 (503)"
    assert rep.steps[0].error.status_code == 503


def test_run_that_raised_replays_to_the_same_error():
    @crashlab.agent(name="crashes")
    def crashes() -> str:
        client = crashlab.wrap(FakeOpenAI([tool_call_response(1, "lookup", {"key": "a"})]))
        client.chat.completions.create(model="m", messages=[])
        raise ValueError("bad state")

    with pytest.raises(ValueError):
        crashes()
    rep = crashlab.replay("last", fn=crashes)  # same error: reproduced, not a divergence
    assert rep.status == "error" and rep.error.type == "ValueError"


# --- policy-drop fault: the prompt is part of the recorded input -------------------


def test_fault_recorded_in_input_replays_without_env(monkeypatch):
    monkeypatch.setenv("FAULT_POLICY_DROP", "1")
    client = crashlab.wrap(FakeOpenAI(refund_script()))
    refund.run_agent("Refund A17", client=client, model="fake-model", faults=["policy_drop"])
    src = crashlab.load("last")
    assert src.input["faults"] == ["policy_drop"]
    assert "30 days" not in src.steps[0].request["system"]
    monkeypatch.delenv("FAULT_POLICY_DROP")
    rep = crashlab.replay(src, overrides={"client": crashlab.wrap(FakeOpenAI([]))})
    assert rep.status == "ok"
    # replaying it as a clean run diverges at the system prompt
    with pytest.raises(crashlab.DivergenceError) as ei:
        crashlab.replay(src, overrides={"client": crashlab.wrap(FakeOpenAI([])), "faults": []})
    assert ei.value.fields == ["system"]
    assert '+2. Only refund orders whose status is "delivered"' in ei.value.diff  # line by line


def test_tool_side_effect_alias():
    @crashlab.tool(side_effect=True)
    def pay() -> None: ...

    assert pay.__crashlab_tool__.side_effects is True


# --- CLI + store ---------------------------------------------------------------------


def test_cli_replay_of_a_crashed_run_reports_the_same_error():
    import re

    from typer.testing import CliRunner

    from crashlab.cli import app

    ag.KNOBS["crash"] = True
    with pytest.raises(RuntimeError):
        ag.mini("x")
    src = crashlab.load("last")
    res = CliRunner().invoke(app, ["replay", src.run_id], env={"COLUMNS": "200", "NO_COLOR": "1"})
    plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
    assert res.exit_code == 0, res.output
    assert "reproduced: 1/1 steps served from the cassette, same error as recorded" in plain
    assert "RuntimeError" in plain and "status: error" in plain


def test_cli_replay_success_divergence_and_errors():
    import re

    from typer.testing import CliRunner

    from crashlab.cli import app

    class _Runner(CliRunner):  # rich adds bold codes even with NO_COLOR
        def invoke(self, *a, **kw):
            res = super().invoke(*a, **kw)
            res.plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
            return res

    runner = _Runner()
    env = {"COLUMNS": "200", "TERM": "xterm-256color", "NO_COLOR": "1"}
    src = _record_mini()
    res = runner.invoke(app, ["replay", src.run_id], env=env)
    assert res.exit_code == 0, res.output
    assert "reproduced: 3/3 steps served from the cassette" in res.plain
    assert f"(replay of {src.run_id})" in res.plain
    assert "identical requests and output" in res.plain
    ls = runner.invoke(app, ["ls"], env=env)
    assert "mini (replay)" in ls.plain
    show = runner.invoke(app, ["show", "last"], env=env)
    assert f"replay of   {src.run_id}" in show.plain and "chat ↺" in show.plain

    ag.KNOBS["system"] = "be verbose"
    res = runner.invoke(app, ["replay", src.run_id], env=env)
    assert res.exit_code == 1
    assert "replay diverged" in res.plain and "differing fields: system" in res.plain
    assert "+be verbose" in res.plain and "Diverged replay stored as" in res.plain

    with crashlab.record("no-entrypoint"):
        pass
    res = runner.invoke(app, ["replay", "last"], env=env)
    assert res.exit_code == 2 and "cannot replay" in res.plain


def test_index_schema_v1_is_migrated(tmp_path):
    import sqlite3

    from crashlab.store.sqlite import _SCHEMA, SQLiteIndex

    db = tmp_path / "index.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA.replace("    replay_of TEXT,\n", ""))
    conn.commit()
    assert "replay_of" not in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    conn.close()
    idx = SQLiteIndex(db)
    idx.list_runs()  # opens + migrates
    conn = sqlite3.connect(db)
    assert "replay_of" in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("2",)
