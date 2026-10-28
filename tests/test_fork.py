"""crashlab.fork: strict replay before step N, patched live calls from N on. Fake clients
only (no network, no keys)."""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from examples.refund_bot import agent as refund
from examples.refund_bot import fork_demo
from typer.testing import CliRunner

import crashlab
from crashlab import patch as P
from crashlab.cli import app
from crashlab.store import Store
from tests import replay_agents as ag
from tests.fakes import (
    AsyncFakeOpenAI,
    FakeAnthropic,
    FakeOpenAI,
    a_text,
    a_tool_use,
    chunks_for_text,
    responses_text,
    text_response,
    tool_call_response,
)


@pytest.fixture(autouse=True)
def _reset():
    ag.reset()
    refund.reset_db()
    yield
    ag.reset()
    refund.reset_db()


def _record(**knobs):
    ag.KNOBS.update(knobs)
    ag.mini("t")
    src = crashlab.load("last")
    ag.reset()
    return src


def _live(*responses, **knobs):
    """Script the fake client the fork's live calls will hit."""
    ag.KNOBS.update(knobs)
    ag.KNOBS["script"] = list(responses)


def _fake():
    return ag.CLIENTS[-1]


APPEND = P.SystemPromptPatch(op="append", text="Be kind.")


# --- boundaries ----------------------------------------------------------------------


def test_fork_at_0_applies_patch_to_every_llm_call_and_links_the_runs():
    src = _record()
    _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "kind done"))
    br = crashlab.fork(src, at_step=0, patch=APPEND)
    assert br.output == "kind done" and br.status == "ok"
    assert (br.forked_from, br.fork_step, br.replay_of) == (src.run_id, 0, None)
    assert br.patch.patches == [APPEND] and br.patch_description.startswith("system prompt")
    assert "fork" in br.tags
    assert [s.source for s in br.steps] == ["patched", "cassette", "patched"]
    sent = _fake().calls
    assert len(sent) == 2
    assert all(c["messages"][0]["content"] == "be brief\nBe kind." for c in sent)
    assert br.steps[0].request["system"] == "be brief\nBe kind."
    assert br.steps[0].attributes["crashlab.fork.patches"] == [APPEND.describe()]
    assert ag.CALLS["lookup"] == 0  # identical tool call: served from the parent's cassette
    again = crashlab.load(br.run_id)
    assert again.forked_from == src.run_id and again.patch == br.patch


def test_steps_before_the_fork_step_are_served_exactly():
    src = _record()
    _live(text_response(9, "later"))
    br = crashlab.fork(src, at_step=2, patch=P.ModelParamPatch(params={"temperature": 1}))
    assert [s.source for s in br.steps] == ["cassette", "cassette", "patched"]
    assert [s.request_hash for s in br.steps[:2]] == [s.request_hash for s in src.steps[:2]]
    assert len(_fake().calls) == 1 and _fake().calls[0]["temperature"] == 1
    assert br.output == "later" and ag.CALLS["lookup"] == 0


def test_negative_fork_step_counts_from_the_end():
    src = _record()
    _live(text_response(9, "end"))
    assert crashlab.fork(src, at_step=-1).fork_step == 2


def test_fork_without_patch_reruns_live():
    src = _record()
    _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "again"))
    br = crashlab.fork(src, at_step=0)
    assert br.output == "again" and [s.source for s in br.steps] == ["live", "cassette", "live"]
    assert br.patch_description == "no patch"


def test_prefix_divergence_raises_and_is_stored():
    src = _record()
    ag.KNOBS["system"] = "changed"  # the agent changed before the fork step
    with pytest.raises(crashlab.DivergenceError) as ei:
        crashlab.fork(src, at_step=2, patch=APPEND)
    assert ei.value.step_idx == 0 and ei.value.fields == ["system"]
    stored = crashlab.load(ei.value.replay_run_id)
    assert stored.status == "error" and stored.outcome_reason.startswith("fork diverged")


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"at_step": 3}, crashlab.ForkError, "out of range"),
        ({"at_step": -9}, crashlab.ForkError, "out of range"),
        ({"at_step": True}, crashlab.ForkError, "must be an int"),
        (
            {"at_step": 2, "patch": P.MessagePatch(step=0, index=0, content="x")},
            crashlab.ForkError,
            "before the fork step",
        ),
        (
            {"at_step": 0, "patch": P.ToolResultPatch(step=0, result=1)},
            P.PatchError,
            "not a tool call",
        ),
        (
            {"at_step": 0, "patch": P.MessagePatch(step=1, index=0, content="x")},
            P.PatchError,
            "not an LLM call",
        ),
        (
            {"at_step": 1, "patch": P.ToolResultPatch(step=1, tool="other", result=1)},
            P.PatchError,
            "calls 'lookup'",
        ),
        (
            {"at_step": 0, "patch": P.SystemPromptPatch(op="insert_line", after="zzz", text="x")},
            P.PatchError,
            "no line containing",
        ),
    ],
)
def test_fork_validates_before_running_anything(kwargs, error, match):
    src = _record()
    before = len(Store().list_runs(limit=0))
    with pytest.raises(error, match=match):
        crashlab.fork(src, **kwargs)
    assert len(Store().list_runs(limit=0)) == before and not ag.CLIENTS


def test_unreached_fork_point_and_unapplied_patches_are_flagged():
    src = _record()
    ag.KNOBS["stop_early"] = True  # the agent now ends after step 1
    br = crashlab.fork(src, at_step=2, patch=APPEND)
    msgs = [f.message for f in br.flags]
    assert any("never reached step 2" in m for m in msgs)
    assert any("patch 1 was never applied" in m for m in msgs)
    assert crashlab.load(br.run_id).flags  # stored


# --- tools ---------------------------------------------------------------------------


def test_new_tool_arguments_run_live_and_tools_live_mode():
    src = _record()
    _live(
        tool_call_response(1, "lookup", {"key": "xyz"}), text_response(2, "ok"), key_from_model=True
    )
    br = crashlab.fork(src, at_step=0)
    assert br.steps[1].source == "live" and br.steps[1].result == {"key": "xyz", "value": 3}
    assert ag.CALLS["lookup"] == 1
    _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "ok"))
    crashlab.fork(src, at_step=0, tools="live")
    assert ag.CALLS["lookup"] == 2  # same args, but tools="live"


def test_side_effect_tools_never_run_live_in_a_fork():
    src = _record(charge=True)  # chat, lookup, charge(5.0), chat
    assert src.steps[2].name == "charge"
    _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "ok"), charge=True)
    br = crashlab.fork(src, at_step=0)
    assert br.steps[2].source == "cassette" and ag.CALLS["charge"] == 0  # identical: served

    ag.reset()
    src2 = _record()  # no charge in the parent
    _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "ok"), charge=True)
    br2 = crashlab.fork(src2, at_step=0)
    assert br2.status == "error" and br2.error.type.endswith("SideEffectBlockedError")
    assert br2.steps[2].source == "blocked" and ag.CALLS["charge"] == 0
    _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "ok"), charge=True)
    with pytest.raises(crashlab.SideEffectBlockedError):
        crashlab.fork(src2, at_step=0, raise_errors=True)
    _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "ok"), charge=True)
    br3 = crashlab.fork(src2, at_step=0, allow_live_side_effects=True)
    assert br3.status == "ok" and ag.CALLS["charge"] == 1 and br3.steps[2].source == "live"


def _pay(times: int) -> None:
    with crashlab.record("pay"):
        for _ in range(times):
            ag.charge(5.0)


def test_an_identical_side_effect_is_served_only_once():
    _pay(1)
    src = crashlab.load("last")
    br = crashlab.fork(src, at_step=0, fn=lambda: _pay(2))
    assert [s.source for s in br.steps] == ["cassette", "blocked"]
    assert ag.CALLS["charge"] == 1  # only the original recording ran it


# --- each patch kind, end to end -----------------------------------------------------


def test_tool_result_patch_counterfactual_value_and_error():
    src = _record()
    _live(text_response(2, "value is 99"))
    br = crashlab.fork(
        src,
        at_step=1,
        patch=P.ToolResultPatch(step=1, tool="lookup", result={"key": "abc", "value": 99}),
    )
    assert br.steps[1].source == "patched" and br.steps[1].result == {"key": "abc", "value": 99}
    assert "99" in _fake().calls[0]["messages"][-1]["content"] and ag.CALLS["lookup"] == 0
    _live(text_response(2, "sorry"))
    br = crashlab.fork(src, at_step=1, patch=P.ToolResultPatch(step=1, error="backend down"))
    assert br.steps[1].error.type.endswith("PatchedToolError")
    assert "PatchedToolError: backend down" in _fake().calls[0]["messages"][-1]["content"]


def test_message_and_inject_patches_end_to_end():
    src = _record()
    _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "ok"))
    patch = P.MessagePatch(step=0, index=0, content="new task") + P.InjectMessagePatch(
        step=2, content="be precise"
    )
    br = crashlab.fork(src, at_step=0, patch=patch)
    first, second = _fake().calls
    assert first["messages"][1] == {"role": "user", "content": "new task"}
    assert second["messages"][1]["content"] == "new task"  # sticky
    assert second["messages"][-1] == {"role": "user", "content": "be precise"}
    assert len(second["messages"]) == 4
    assert br.steps[0].request["messages"][0]["parts"][0]["content"] == "new task"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Find a key.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
        },
    }
]


def _tools_agent(client):
    with crashlab.record("tools-agent") as run:
        r = client.chat.completions.create(
            model="m1", temperature=0, tools=TOOLS, messages=[{"role": "user", "content": "hi"}]
        )
        run.output = r.choices[0].message.content


def test_tool_description_schema_and_model_patches_end_to_end():
    _tools_agent(crashlab.wrap(FakeOpenAI([text_response(1, "first")])))
    src = crashlab.load("last")
    live = FakeOpenAI([text_response(2, "second")])
    patch = P.PatchSet.of(
        P.ToolDescriptionPatch(tool="lookup", description="Find a key. Keys are lowercase."),
        P.ToolSchemaPatch(tool="lookup", merge=True, parameters={"required": ["key"]}),
        P.ModelParamPatch(model="m2", params={"max_tokens": 50}, unset=["temperature"]),
    )
    br = crashlab.fork(src, at_step=0, patch=patch, fn=lambda: _tools_agent(crashlab.wrap(live)))
    sent = live.calls[0]
    fn = sent["tools"][0]["function"]
    assert fn["description"] == "Find a key. Keys are lowercase."
    assert fn["parameters"]["required"] == ["key"] and "key" in fn["parameters"]["properties"]
    assert sent["model"] == "m2" and sent["max_tokens"] == 50 and "temperature" not in sent
    assert br.steps[0].name == "m2" and br.output == "second"
    assert TOOLS[0]["function"]["description"] == "Find a key."  # user data untouched


# --- providers -----------------------------------------------------------------------


def test_fork_anthropic_create_and_stream_helper():
    def body(client):
        with crashlab.record("anth") as run:
            a = client.messages.create(
                model="claude-x",
                max_tokens=10,
                system="Be brief.",
                messages=[{"role": "user", "content": "hi"}],
            )
            with client.messages.stream(
                model="claude-x",
                max_tokens=10,
                system="Be brief.",
                messages=[{"role": "user", "content": "more"}],
            ) as s:
                b = s.get_final_message()
            run.output = [
                a.content[-1].input if a.content[-1].type == "tool_use" else a.content[-1].text,
                b.content[-1].text,
            ]

    body(crashlab.wrap(FakeAnthropic([a_tool_use(1, "get_order", {"id": "1"}), a_text(2, "s")])))
    src = crashlab.load("last")
    live = FakeAnthropic([a_text(4, "streamed live")])
    br = crashlab.fork(
        src,
        at_step=1,
        patch=P.SystemPromptPatch(op="append", text="Kind."),
        fn=lambda: body(crashlab.wrap(live)),
    )
    assert br.output == [{"id": "1"}, "streamed live"]
    assert [s.source for s in br.steps] == ["cassette", "patched"]
    assert live.calls[0]["system"] == "Be brief.\nKind."


def test_fork_openai_responses_stream_and_async():
    def responses(client):
        with crashlab.record("resp") as run:
            r = client.responses.create(model="gpt-x", instructions="Be brief.", input="hi")
            run.output = r["output"][0]["content"][0]["text"]

    responses(crashlab.wrap(FakeOpenAI([responses_text(1, "a")])))
    live = FakeOpenAI([responses_text(2, "b")])
    br = crashlab.fork(
        crashlab.load("last"),
        at_step=0,
        fn=lambda: responses(crashlab.wrap(live)),
        patch=P.SystemPromptPatch(op="replace", text="Be long."),
    )
    assert live.calls[0]["instructions"] == "Be long." and br.output == "b"

    def streamed(client):
        with crashlab.record("stream") as run:
            s = client.chat.completions.create(
                model="m",
                stream=True,
                messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            )
            run.output = "".join(c.choices[0]["delta"].get("content") or "" for c in s if c.choices)

    streamed(crashlab.wrap(FakeOpenAI([chunks_for_text("one")])))
    live = FakeOpenAI([chunks_for_text("two")])
    br = crashlab.fork(
        crashlab.load("last"),
        at_step=0,
        fn=lambda: streamed(crashlab.wrap(live)),
        patch=P.SystemPromptPatch(op="append", text="t"),
    )
    assert br.output == "two" and live.calls[0]["messages"][0]["content"] == "s\nt"
    assert br.steps[0].source == "patched" and br.steps[0].attributes["crashlab.stream"]

    async def amain(client):
        async with crashlab.record("async") as run:
            r = await client.chat.completions.create(
                model="m",
                messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            )
            run.output = r.choices[0].message.content

    asyncio.run(amain(crashlab.wrap(AsyncFakeOpenAI([text_response(1, "x")]))))
    live = AsyncFakeOpenAI([text_response(2, "y")])
    br = crashlab.fork(
        crashlab.load("last"),
        at_step=0,
        fn=lambda: amain(crashlab.wrap(live)),
        patch=P.ModelParamPatch(model="m9"),
    )
    assert br.output == "y" and live.calls[0]["model"] == "m9"


# --- refund_bot: the headline demo, offline ------------------------------------------

RESTORE_POLICY = fork_demo.RESTORE_POLICY


def _c03_refund():
    return [
        tool_call_response(1, "get_order", {"order_id": "C03"}),
        tool_call_response(2, "issue_refund", {"order_id": "C03", "amount": 120}),
        text_response(3, "Your refund of $120 has been issued."),
    ]


def test_policy_drop_fork_flips_fail_to_pass():
    refund.run_agent(
        "Please refund order C03",
        client=crashlab.wrap(FakeOpenAI(_c03_refund())),
        model="fake-model",
        faults=["policy_drop"],
    )
    src = crashlab.load("last")
    assert src.outcome == "fail" and "45 days ago" in src.outcome_reason
    refund.reset_db()
    refuse = crashlab.wrap(
        FakeOpenAI(
            [
                tool_call_response(1, "get_order", {"order_id": "C03"}),
                text_response(2, "Sorry, C03 is outside the 30-day refund window."),
            ]
        )
    )
    br = crashlab.fork(src, at_step=0, patch=RESTORE_POLICY, client=refuse)
    assert br.outcome == "pass" and not refund.REFUNDS
    assert "30 days ago" in br.steps[0].request["system"]
    diff = crashlab.diff(src, br)
    changed = {c.name for c in diff.checks if c.changed}
    assert changed == {"refund_within_policy"}
    assert diff.relation == "B is a fork of A at step 0"
    # the patched model still refunds: the identical refund is served (never re-run) and
    # still fails the check
    br2 = crashlab.fork(
        src, at_step=0, patch=RESTORE_POLICY, client=crashlab.wrap(FakeOpenAI(_c03_refund()))
    )
    assert br2.outcome == "fail" and br2.steps[3].source == "cassette" and not refund.REFUNDS


def test_a_blocked_refund_attempt_still_counts_for_the_checks():
    refund.run_agent(
        "Please refund order C03",
        client=crashlab.wrap(FakeOpenAI(_c03_refund())),
        model="fake-model",
        faults=["policy_drop"],
    )
    src = crashlab.load("last")
    half = crashlab.wrap(
        FakeOpenAI(
            [
                tool_call_response(1, "get_order", {"order_id": "C03"}),
                tool_call_response(2, "issue_refund", {"order_id": "C03", "amount": 60}),
                text_response(3, "Refunded half."),
            ]
        )
    )
    br = crashlab.fork(src, at_step=0, client=half)
    assert br.steps[3].source == "blocked" and br.outcome == "fail"
    assert "attempt blocked by crashlab" in br.outcome_reason


def test_fork_demo_script_counts_flips(capsys):
    refund.run_agent(
        "Please refund order C03",
        client=crashlab.wrap(FakeOpenAI(_c03_refund())),
        model="fake-model",
        faults=["policy_drop"],
    )
    src = crashlab.load("last")
    assert 'status is "delivered"' in RESTORE_POLICY.text and "30 days" in RESTORE_POLICY.text
    replies = iter([
        [tool_call_response(1, "get_order", {"order_id": "C03"}), text_response(2, "No.")],
        _c03_refund(),
    ])  # fmt: skip
    branches = fork_demo.fork_k(
        src, 2, RESTORE_POLICY, lambda: crashlab.wrap(FakeOpenAI(next(replies)))
    )
    assert [b.outcome for b in branches] == ["pass", "fail"]
    assert all(b.forked_from == src.run_id and b.fork_step == 0 for b in branches)
    out = capsys.readouterr().out
    assert "pass no refund" in out and "fail refunded 120" in out
    assert [fork_demo.tally(b) for b in branches] == ["pass", "fail"]
    boom = crashlab.fork(src, at_step=0, client=crashlab.wrap(FakeOpenAI([])))
    assert boom.status == "error" and fork_demo.tally(boom).startswith("error (")
    with pytest.raises(SystemExit):
        fork_demo.first_llm_step(src.model_copy(update={"steps": []}))


# --- CLI -----------------------------------------------------------------------------


class _Runner(CliRunner):
    def invoke(self, *a, **kw):
        kw.setdefault("env", {"COLUMNS": "220", "TERM": "xterm-256color", "NO_COLOR": "1"})
        res = super().invoke(*a, **kw)
        res.plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
        return res


def test_cli_fork_with_flags_then_diff_and_show():
    runner = _Runner()
    src = _record()
    _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "kind"))
    res = runner.invoke(
        app,
        [
            "fork",
            src.run_id,
            "--at",
            "0",
            "--system-append",
            "Be kind.",
            "--param",
            "temperature=0.5",
            "--model",
            "fake-2",
        ],
    )
    assert res.exit_code == 0, res.output
    assert (
        "+Be kind." in res.plain and "model params: model -> fake-2, temperature=0.5" in res.plain
    )
    m = re.search(r"fork run (\w+) \(fork of (\w+) at step 0\)", res.plain)
    assert m and m.group(2) == src.run_id
    fork_id = m.group(1)
    sent = _fake().calls[0]
    assert sent["model"] == "fake-2" and sent["temperature"] == 0.5
    assert "fork@0" in runner.invoke(app, ["ls"]).plain
    show = runner.invoke(app, ["show", fork_id]).plain
    assert f"forked from  {src.run_id} at step 0" in show and "chat ✎" in show
    assert 'patch        1. system prompt: append "Be kind."' in show
    assert "execute_tool ↺" in show

    d = runner.invoke(app, ["diff", fork_id])  # parent resolved automatically
    assert d.exit_code == 0, d.output
    assert "B is a fork of A at step 0" in d.plain
    assert "first divergence at step 0: different request (params, system)" in d.plain
    assert "≠ request" in d.plain and "+Be kind." in d.plain
    js = json.loads(runner.invoke(app, ["diff", src.run_id, fork_id, "--json"]).output)
    assert js["first_divergence"] == 0 and js["steps"][1]["status"] == "same"
    assert runner.invoke(app, ["diff", src.run_id]).exit_code == 2  # no parent, one run


def test_cli_fork_patch_file_dry_run_save_and_errors(tmp_path):
    runner = _Runner()
    src = _record()
    path = tmp_path / "fix.json"
    path.write_text(
        P.as_patchset(
            [P.SystemPromptPatch(op="substitute", old="brief", text="thorough")]
        ).to_json()
    )
    saved = tmp_path / "saved.json"
    res = runner.invoke(
        app,
        [
            "fork",
            src.run_id,
            "--at",
            "0",
            "--patch",
            str(path),
            "--dry-run",
            "--tool-result",
            "1",
            '{"value": 7}',
            "--save-patch",
            str(saved),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "-be brief" in res.plain and "+be thorough" in res.plain
    assert "-value: 3" in res.plain and "+value: 7" in res.plain
    assert 'tool result at step 1 -> {"value": 7}' in res.plain
    assert not ag.CLIENTS and len(Store().list_runs(limit=0)) == 1  # nothing ran
    assert [p.kind for p in P.load(saved)] == ["system_prompt", "tool_result"]
    bad = runner.invoke(app, ["fork", src.run_id, "--at", "0", "--system-insert-after", "zzz", "x"])
    assert bad.exit_code == 2 and "bad patch" in bad.plain
    oob = runner.invoke(app, ["fork", src.run_id, "--at", "7", "--system-append", "x"])
    assert oob.exit_code == 2 and "out of range" in oob.plain
    ag.KNOBS["system"] = "changed"
    div = runner.invoke(app, ["fork", src.run_id, "--at", "2", "--system-append", "x"])
    assert div.exit_code == 1 and "fork prefix diverged" in div.plain


def test_cli_message_inject_and_tool_description_flags():
    runner = _Runner()
    src = _record()
    _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "ok"))
    res = runner.invoke(
        app,
        [
            "fork",
            src.run_id,
            "--at",
            "0",
            "--message",
            "0",
            "0",
            "new task",
            "--inject-user",
            "2",
            "check twice",
        ],
    )
    assert res.exit_code == 0, res.output
    first, second = _fake().calls
    assert first["messages"][1]["content"] == "new task"
    assert second["messages"][-1]["content"] == "check twice"
    res = runner.invoke(
        app, ["fork", src.run_id, "--at", "0", "--tool-description", "lookup", "x", "--dry-run"]
    )
    assert res.exit_code == 2 and "not in the request" in res.plain  # mini sends no tools


def test_a_registered_fake_serves_a_new_side_effect_call():
    # verify on real agents: a fix that changes a side-effect argument ("widgets" for
    # "widget") was blocked in every trial; a fake lets the fork grade it, no real call
    src = _record()  # no charge in the parent
    seen = []
    crashlab.fake_tool("charge", lambda amount: seen.append(amount) or {"charged": 0, "fake": 1})
    try:
        _live(tool_call_response(1, "lookup", {"key": "abc"}), text_response(2, "ok"), charge=True)
        br = crashlab.fork(src, at_step=0)
    finally:
        crashlab.clear_fakes()
    assert br.status == "ok" and ag.CALLS["charge"] == 0 and seen == [5.0]
    step = br.steps[2]
    assert step.source == "fake" and step.result == {"charged": 0, "fake": 1}
    assert step.attributes["crashlab.tool.fake"] is True
    # record mode always runs the real tool, fakes or not
    crashlab.fake_tool("charge", lambda amount: {"fake": 1})
    try:
        _pay(1)
    finally:
        crashlab.clear_fakes()
    assert ag.CALLS["charge"] == 1 and crashlab.load("last").steps[0].source == "live"


def test_fake_argument_on_the_tool_and_async_fakes():
    async def fake_send(to: str) -> str:
        return f"fake:{to}"

    sent = []

    @crashlab.tool(side_effects=True, fake=fake_send)
    async def send(to: str) -> str:
        sent.append(to)
        return f"sent:{to}"

    async def main(n: int) -> list[str]:
        with crashlab.record("send"):
            return [await send(f"u{i}") for i in range(n)]

    asyncio.run(main(1))
    src = crashlab.load("last")
    br = crashlab.fork(src, at_step=0, fn=lambda: asyncio.run(main(2)))
    assert [s.source for s in br.steps] == ["cassette", "fake"]
    assert br.steps[1].result == "fake:u1" and sent == ["u0"]
