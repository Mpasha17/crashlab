"""Multi-agent runs: which agent made each step (``gen_ai.agent.name``).

Found testing an OpenAI Agents SDK handoff (triage -> refunds): the recorded chat steps
did not say which agent made them, so a failing refund could not be pinned on the
refunds agent's prompt vs the triage agent's routing."""

from __future__ import annotations

import asyncio

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.models import Step
from tests.fakes import (
    AsyncFakeOpenAI,
    FakeAnthropic,
    FakeOpenAI,
    a_text,
    text_response,
    tool_call_response,
)

pytestmark = pytest.mark.requires_providers("pydantic")


@crashlab.tool(side_effects=True)
def issue_refund(order_id: str, amount: float) -> dict:
    return {"refunded": amount}


@crashlab.tool()
def lookup(order_id: str) -> dict:
    return {"total": 15.0}


def _handoff(triage_client, refunds_client) -> str:
    """triage decides on a handoff, refunds does the work (one client per agent)."""
    r = triage_client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "refund A1"}]
    )
    assert r.choices[0].message.tool_calls[0].function.name == "transfer_to_refunds"
    r = refunds_client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "refund A1"}]
    )
    lookup("A1")
    r = refunds_client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}]
    )
    issue_refund("A1", 15.0)
    r = refunds_client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "y"}]
    )
    return r.choices[0].message.content


def _scripts():
    triage = FakeOpenAI([tool_call_response(1, "transfer_to_refunds", {})])
    refunds = FakeOpenAI([
        tool_call_response(2, "lookup", {"order_id": "A1"}),
        tool_call_response(3, "issue_refund", {"order_id": "A1", "amount": 15.0}),
        text_response(4, "refunded"),
    ])  # fmt: skip
    return triage, refunds


@crashlab.agent(name="support")
def support() -> str:
    triage, refunds = _scripts()
    return _handoff(crashlab.wrap(triage, agent="triage"), crashlab.wrap(refunds, agent="refunds"))


def test_wrap_agent_labels_chat_and_inherited_tool_steps():
    assert support() == "refunded"
    run = crashlab.load("last")
    assert [(s.operation, s.agent) for s in run.steps] == [
        ("chat", "triage"),
        ("chat", "refunds"),
        ("execute_tool", "refunds"),
        ("chat", "refunds"),
        ("execute_tool", "refunds"),
        ("chat", "refunds"),
    ]
    assert run.agents == ["triage", "refunds"]
    assert [s.idx for s in run.chat_steps(agent="triage")] == [0]
    assert [s.name for s in run.tool_calls(agent="refunds")] == ["lookup", "issue_refund"]
    assert run.tool_calls(agent="triage") == []


def test_labels_survive_replay_and_do_not_change_hashes():
    support()
    src = crashlab.load("last")
    crashlab.replay(src)
    rep = crashlab.load("last")
    assert rep.replay_of == src.run_id and rep.status == "ok"
    assert [s.agent for s in rep.steps] == [s.agent for s in src.steps]
    # the label is not part of the request: same hash as an unlabelled recording

    @crashlab.agent(name="support-unlabelled")
    def plain() -> str:
        triage, refunds = _scripts()
        return _handoff(crashlab.wrap(triage), crashlab.wrap(refunds))

    plain()
    bare = crashlab.load("last")
    assert all(s.agent is None for s in bare.steps)
    assert [s.request_hash for s in bare.steps] == [s.request_hash for s in src.steps]


def test_unlabelled_0_0_3_steps_read_as_no_agent():
    step = Step(run_id="r", idx=0, span_id="s", operation="chat", name="m", request_hash="h")
    assert step.agent is None


def test_agent_scope_labels_a_shared_client_and_nests():
    @crashlab.agent(name="graph")
    def graph() -> None:
        client = crashlab.wrap(FakeOpenAI([text_response(1, "a"), text_response(2, "b"),
                                           text_response(3, "c")]))  # fmt: skip
        with crashlab.agent_scope("supervisor"):
            client.chat.completions.create(model="m", messages=[])
            with crashlab.agent_scope("researcher"):
                client.chat.completions.create(model="m", messages=[])
        client.chat.completions.create(model="m", messages=[])

    graph()
    run = crashlab.load("last")
    assert [s.agent for s in run.steps] == ["supervisor", "researcher", None]


def test_agent_scope_in_async_tasks():
    @crashlab.agent(name="async-graph")
    def g() -> None:
        client = crashlab.wrap(AsyncFakeOpenAI([text_response(1, "a"), text_response(2, "b")]))

        async def worker(name: str) -> None:
            with crashlab.agent_scope(name):
                await asyncio.sleep(0.01 if name == "a" else 0)
                await client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": name}]
                )

        async def go():
            await asyncio.gather(worker("a"), worker("b"))

        asyncio.run(go())

    g()
    run = crashlab.load("last")
    by_agent = {s.agent: s.request["messages"][-1] for s in run.steps}
    assert set(by_agent) == {"a", "b"}
    for name, msg in by_agent.items():
        assert name in str(msg)


def test_wrap_agent_wins_over_scope_and_tool_scope_wins_over_inherited():
    @crashlab.agent(name="mixed")
    def m() -> None:
        c = crashlab.wrap(
            FakeOpenAI([tool_call_response(1, "lookup", {"order_id": "A1"})]), agent="refunds"
        )
        with crashlab.agent_scope("other"):
            c.chat.completions.create(model="m", messages=[])
            lookup("A1")

    m()
    run = crashlab.load("last")
    assert [s.agent for s in run.steps] == ["refunds", "other"]


def test_nested_agent_steps_carry_the_sub_agent_name():
    @crashlab.agent(name="researcher")
    def researcher(q: str) -> str:
        return lookup(q)["total"]

    @crashlab.agent(name="orchestrator")
    def orchestrator() -> float:
        return researcher("A1")

    orchestrator()
    run = crashlab.load("last")
    assert [(s.operation, s.agent) for s in run.steps] == [
        ("invoke_agent", "researcher"),
        ("execute_tool", "researcher"),
    ]


def test_wrap_agent_on_anthropic_and_rewrap():
    client = crashlab.wrap(FakeAnthropic([a_text(1, "hi")]), agent="writer")

    @crashlab.agent(name="anth")
    def a() -> None:
        client.messages.create(
            model="claude-fake", max_tokens=10, messages=[{"role": "user", "content": "x"}]
        )

    a()
    assert crashlab.load("last").steps[0].agent == "writer"
    # re-labelling an already wrapped client wraps the original under the new name
    other = crashlab.wrap(client, agent="editor")
    assert other is not client and other.unwrapped is client.unwrapped
    assert crashlab.wrap(client, agent="writer") is client


def test_with_options_keeps_the_agent_label():
    base = crashlab.wrap(FakeOpenAI([]), agent="triage")
    base._inner.with_options = lambda **kw: FakeOpenAI([text_response(1, "ok")])

    @crashlab.agent(name="opts")
    def o() -> None:
        base.with_options(timeout=5).chat.completions.create(model="m", messages=[])

    o()
    assert crashlab.load("last").steps[0].agent == "triage"


@pytest.mark.parametrize("bad", ["", "  ", 3])
def test_bad_agent_names_are_refused(bad):
    with pytest.raises(ValueError):
        crashlab.wrap(FakeOpenAI([]), agent=bad)
    with pytest.raises(ValueError), crashlab.agent_scope(bad):
        pass


@pytest.mark.requires_providers("litellm")
def test_litellm_module_refuses_per_client_agent():
    import litellm

    with pytest.raises(TypeError, match="agent_scope"):
        crashlab.wrap(litellm, agent="x")


def test_show_prints_agents(monkeypatch):
    from crashlab.cli import app

    support()
    monkeypatch.setenv("COLUMNS", "160")
    out = CliRunner().invoke(app, ["show", "last"]).output
    assert "triage → refunds" in out
    assert "@refunds" in out
    monkeypatch.setenv("COLUMNS", "80")
    out = CliRunner().invoke(app, ["show", "last"]).output
    assert "@refunds" in out


def test_show_agent_filter(monkeypatch):
    from crashlab.cli import app

    support()
    monkeypatch.setenv("COLUMNS", "160")
    out = CliRunner().invoke(app, ["show", "last", "--agent", "triage"]).output
    assert "showing 1 of 6 steps: agent triage" in out
    assert "lookup" not in out.split("showing")[0].split("agents")[-1]
    out = CliRunner().invoke(app, ["show", "last", "--agent", "billing", "--tools"]).output
    assert "agent billing (agents: triage, refunds)" in out
    out = CliRunner().invoke(app, ["show", "last", "--agent", "refunds", "--tools"]).output
    assert "showing 2 of 6 steps" in out


def test_diagnoser_view_names_the_agent_of_each_step():
    from crashlab.diagnose.compact import compact

    support()
    view = compact(crashlab.load("last"))
    lines = view.splitlines()
    assert any(ln.startswith("[0] chat") and ln.endswith("[agent: triage]") for ln in lines)
    assert "[2] tool lookup  [agent: refunds]" in lines

    @crashlab.agent(name="plain")
    def plain() -> None:
        crashlab.wrap(FakeOpenAI([text_response(1, "x")])).chat.completions.create(
            model="m", messages=[]
        )

    plain()
    assert "[agent:" not in compact(crashlab.load("last"))
