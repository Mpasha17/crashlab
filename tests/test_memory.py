# ruff: noqa: E501
"""Memory / state debugging: memory.* detectors, crashlab.snapshot, crashlab state / memory."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab import memory as mem
from crashlab.checks import ChecksConfig, evaluate, resolve_config
from crashlab.checks.detectors import KINDS
from crashlab.cli import app
from crashlab.models import Run, Step
from tests.fakes import FakeOpenAI, text_response

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
MEMORY_KINDS = (
    "memory.context_truncation",
    "memory.forgotten_fact",
    "memory.self_contradiction",
    "memory.stale_tool_result",
    "memory.unbounded_growth",
)


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _m(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "parts": [{"type": "text", "content": text}]}


def _chat(idx: int, msgs: list[tuple[str, str]], reply: str, *, agent=None, turn=None,
          tokens: int | None = None, system: str | None = "You are a helpful support agent.",
          tool_msgs: list[dict] | None = None) -> Step:  # fmt: skip
    attrs: dict[str, Any] = {"gen_ai.request.model": "fake-model"}
    if tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = tokens
    if agent:
        attrs["gen_ai.agent.name"] = agent
    if turn is not None:
        attrs["crashlab.turn"] = turn
    messages = [_m(r, t) for r, t in msgs] + (tool_msgs or [])
    return Step(
        run_id="r", idx=idx, span_id=f"c{idx}", operation="chat", name="fake-model",
        attributes=attrs, request={"system": system, "messages": messages},
        response={"messages": [{"role": "assistant", "parts": [{"type": "text", "content": reply}]}]},
        request_hash=f"h{idx}",
    )  # fmt: skip


def _tool(idx: int, name: str, args: dict, result: Any, *, side_effects: bool = False) -> Step:
    attrs: dict[str, Any] = {"gen_ai.tool.name": name}
    if side_effects:
        attrs["crashlab.tool.side_effects"] = True
    return Step(
        run_id="r", idx=idx, span_id=f"t{idx}", operation="execute_tool", name=name,
        attributes=attrs, request={"args": args}, response={"result": result},
        request_hash=f"h{idx}",
    )  # fmt: skip


def _run(steps: list[Step]) -> Run:
    return Run(run_id="r", name="mem", status="ok", steps=steps)


def _kinds(steps: list[Step], cfg: Any = None) -> list[str]:
    return [
        f.kind for f in evaluate(_run(steps), cfg or ChecksConfig()) if f.kind.startswith("memory.")
    ]


def _findings(steps: list[Step], kind: str, cfg: Any = None):
    return [f for f in evaluate(_run(steps), cfg or ChecksConfig()) if f.kind == kind]


def _conversation() -> list[Step]:
    """A normal 4-turn support chat with facts, history passed on every turn."""
    h: list[tuple[str, str]] = []
    steps: list[Step] = []
    script = [
        (
            "Hi, my name is Priya Sharma and my email is priya@example.com.",
            "Hello Priya! How can I help?",
        ),
        (
            "I want to return order A1234, it arrived broken.",
            "Sorry about that, Priya. I can start the return for order A1234.",
        ),
        (
            "Great, please do. Also my budget for a replacement is $200.",
            "Done. A replacement within your budget of $200 is possible.",
        ),
        (
            "Thanks, that is all.",
            "You're welcome, Priya. A confirmation went to priya@example.com.",
        ),
    ]
    for i, (user, reply) in enumerate(script):
        h.append(("user", user))
        steps.append(_chat(i, list(h), reply, turn=i + 1, tokens=100 + 60 * i))
        h.append(("assistant", reply))
    return steps


# --- a normal conversation never fires --------------------------------------------------


def test_normal_multi_turn_has_no_findings():
    assert _kinds(_conversation()) == []


def test_normal_conversation_with_tools_and_state_changes_has_no_findings():
    steps = [
        _chat(0, [("user", "What is the status of order A1234?")], "Let me check."),
        _tool(1, "get_order", {"order_id": "A1234"}, {"status": "pending", "total": 49.0}),
        _chat(2, [("user", "What is the status of order A1234?"), ("assistant", "Let me check.")],
              "Your order total is $49.00 and it is pending."),
        _tool(3, "cancel_order", {"order_id": "A1234"}, {"ok": True}, side_effects=True),
        _tool(4, "get_order", {"order_id": "A1234"}, {"status": "cancelled", "total": 49.0}),
        _chat(5, [("user", "What is the status of order A1234?"), ("assistant", "Let me check."),
                  ("assistant", "Your order total is $49.00 and it is pending.")],
              "Your order total is $49.00 and it is cancelled."),
    ]  # fmt: skip
    assert _kinds(steps) == []


# --- context truncation -----------------------------------------------------------------


def test_context_truncation_names_the_lost_turn_and_fact():
    u1 = "My email is priya@example.com and my order is A1234."
    steps = [
        _chat(0, [("user", u1)], "Got it."),
        _chat(1, [("user", u1), ("assistant", "Got it."), ("user", "Any update?")], "Checking."),
        _chat(2, [("assistant", "Got it."), ("user", "Any update?"), ("assistant", "Checking."),
                  ("user", "Hello??")], "Hi!"),
    ]  # fmt: skip
    (f,) = _findings(steps, "memory.context_truncation")
    assert f.steps == [1, 2] and f.severity == "error"
    assert "user turn 1" in f.message and "priya@example.com" in f.message
    assert f.evidence[0].step_idx == 1 and "priya@example.com" in f.evidence[0].quote
    assert f.evidence[0].field.startswith("request.messages[0]")


def test_context_truncation_reported_once_per_dropped_message():
    u1 = "I booked room 12 for two nights."
    steps = [
        _chat(0, [("user", u1)], "ok"),
        _chat(1, [("user", u1), ("assistant", "ok"), ("user", "and?")], "yes"),
        _chat(2, [("assistant", "ok"), ("user", "and?"), ("assistant", "yes"), ("user", "so?")], "a"),
        _chat(3, [("user", "and?"), ("assistant", "yes"), ("user", "so?"), ("assistant", "a"),
                  ("user", "then?")], "b"),
    ]  # fmt: skip
    found = _findings(steps, "memory.context_truncation")
    assert [f.steps for f in found] == [[1, 2], [2, 3]]  # the second drop is a new message


def test_a_summary_message_means_compaction_not_truncation():
    steps = [
        _chat(0, [("user", "first question about my invoice INV-9001")], "answer one"),
        _chat(1, [("user", "first question about my invoice INV-9001"),
                  ("assistant", "answer one"), ("user", "second")], "answer two"),
        _chat(2, [("assistant", "Summary of the conversation so far: invoice INV-9001."),
                  ("user", "second"), ("assistant", "answer two"), ("user", "third")], "three"),
    ]  # fmt: skip
    assert _kinds(steps) == []


def test_a_new_shorter_conversation_is_not_truncation():
    a = [_chat(0, [("user", "hi")], "hello"),
         _chat(1, [("user", "hi"), ("assistant", "hello"), ("user", "bye")], "bye")]  # fmt: skip
    b = [_chat(2, [("user", "hi")], "hello")]
    assert _kinds(a + b) == []


def test_context_truncation_ignores_other_agents_and_can_be_switched_off():
    u1 = "my order is A1234"
    steps = [
        _chat(0, [("user", u1)], "ok", agent="a"),
        _chat(1, [("user", u1), ("assistant", "ok"), ("user", "x")], "k", agent="a"),
        _chat(2, [("assistant", "ok"), ("user", "x"), ("assistant", "k"), ("user", "y")], "z", agent="a"),
        _chat(3, [("user", "unrelated")], "fine", agent="b"),
    ]  # fmt: skip
    assert _kinds(steps) == ["memory.context_truncation"]
    off = {"memory": {"context_truncation": False}}
    assert _kinds(steps, resolve_config(off)) == []


def test_turn_without_history_is_a_reset():
    steps = [
        _chat(0, [("user", "My name is Priya and my order is A1234.")], "Hi Priya.", turn=1),
        _chat(1, [("user", "Where is it?")], "Which order do you mean?", turn=2),
    ]
    (f,) = _findings(steps, "memory.context_truncation")
    assert f.key == "reset:2" and "turn 2" in f.message and "not passed on" in f.message
    assert _kinds(steps, resolve_config({"memory": {"context_truncation": {"reset": False}}}))[
        :1
    ] != ["memory.context_truncation"]


# --- forgotten facts --------------------------------------------------------------------


def test_extract_facts():
    facts = {
        (f.kind, f.label): f.value
        for f in mem.extract_facts(
            "My name is Priya Sharma, email priya@example.com, phone +1 415 555 0132, "
            "order #A1234, birthday March 3, budget $500, party of 4."
        )
    }
    assert facts[("name", "name")] == "Priya Sharma"
    assert facts[("email", "email")] == "priya@example.com"
    assert facts[("phone", "phone")].startswith("+1 415")
    assert facts[("id", "order")] == "A1234"
    assert facts[("date", "birthday")] == "March 3"
    assert facts[("money", "budget")] == "$500"
    assert facts[("count", "people")] == "4"
    assert mem.extract_facts("I'm looking for a refund, and I'm fine.") == []
    assert [f.kind for f in mem.extract_facts("Hello there", speaker="assistant")] == []
    assert mem.extract_facts("Hi Priya, thanks", speaker="assistant")[0].value == "Priya"


def test_forgotten_fact_asked_again():
    steps = [
        _chat(0, [("user", "Please refund order A1234, my email is priya@example.com.")], "On it."),
        _chat(1, [("user", "Please refund order A1234, my email is priya@example.com."),
                  ("assistant", "On it."), ("user", "Well?")],
              "Could you share your order number and email address?"),
    ]  # fmt: skip
    found = _findings(steps, "memory.forgotten_fact")
    assert {f.key for f in found} == {"ask:id:order:a1234", "ask:email:email:priya@example.com"}
    f = next(f for f in found if "order" in f.key)
    assert f.severity == "warn" and "turn 1" in f.message and f.steps == [0, 1]
    assert "order A1234" in f.evidence[0].quote
    assert "order number" in f.evidence[1].quote


def test_confirming_a_fact_is_not_asking_again():
    u = "My email is priya@example.com and my order is A1234."
    steps = [
        _chat(0, [("user", u)], "Thanks."),
        _chat(1, [("user", u), ("assistant", "Thanks."), ("user", "ok")],
              "Just to confirm, is your email address priya@example.com? And is order A1234 the one?"),
    ]  # fmt: skip
    assert _kinds(steps) == []


def test_asking_for_a_fact_the_user_never_gave_is_fine():
    steps = [
        _chat(0, [("user", "I need help with a refund.")], "Sure. What is your order number?"),
        _chat(1, [("user", "I need help with a refund."), ("assistant", "Sure. What is your order number?"),
                  ("user", "It is A1234.")], "Thanks. Could you also confirm your email?"),
    ]  # fmt: skip
    assert _kinds(steps) == []


def test_forgotten_fact_contradicted_name_and_order():
    u = "Hi, my name is Priya and my order is A1234."
    steps = [
        _chat(0, [("user", u)], "Hello Priya."),
        _chat(1, [("user", u), ("assistant", "Hello Priya."), ("user", "status?")],
              "Hi Rahul, I looked up order B9876 for you."),
    ]  # fmt: skip
    found = {f.key: f for f in _findings(steps, "memory.forgotten_fact")}
    assert set(found) == {"contradict:name:name:rahul", "contradict:id:order:b9876"}
    assert found["contradict:name:name:rahul"].severity == "error"
    assert "'Rahul'" in found["contradict:name:name:rahul"].message


def test_a_value_from_a_tool_result_is_not_a_contradiction():
    u = "My order is A1234, please check it."
    steps = [
        _chat(0, [("user", u)], "Checking."),
        _tool(1, "find_orders", {"q": "A1234"}, {"related": ["B9876"]}),
        _chat(2, [("user", u), ("assistant", "Checking.")], "Your order is fine; related order B9876 shipped."),
    ]  # fmt: skip
    assert _kinds(steps) == []


def test_forgotten_fact_ignore_kinds():
    u = "My email is priya@example.com."
    steps = [
        _chat(0, [("user", u)], "ok"),
        _chat(1, [("user", u), ("assistant", "ok"), ("user", "hi")], "What is your email address?"),
    ]
    assert _kinds(steps) == ["memory.forgotten_fact"]
    cfg = resolve_config({"memory": {"forgotten_fact": {"ignore_kinds": ["email"]}}})
    assert _kinds(steps, cfg) == []


# --- self contradiction -----------------------------------------------------------------


def test_self_contradiction_numbers_and_dates():
    steps = [
        _chat(0, [("user", "How much?")], "The total is $120.00. Delivery date is 2026-12-01."),
        _chat(1, [("user", "How much?"), ("assistant", "x"), ("user", "again?")],
              "The total is $150.00. The delivery date is 2026-12-05."),
    ]  # fmt: skip
    found = _findings(steps, "memory.self_contradiction")
    assert len(found) == 2 and all(f.steps == [0, 1] for f in found)
    assert any("$120.00" in f.message and "$150.00" in f.message for f in found)
    assert {e.step_idx for e in found[0].evidence} == {0, 1}


def test_self_contradiction_yes_no():
    steps = [
        _chat(0, [("user", "refund?")], "Your order is eligible for a refund."),
        _chat(
            1, [("user", "refund?"), ("user", "sure?")], "Your order is not eligible for a refund."
        ),
    ]
    (f,) = _findings(steps, "memory.self_contradiction")
    assert "eligible" in f.message
    can = [_chat(0, [("user", "a")], "I can cancel that for you."),
           _chat(1, [("user", "a"), ("user", "b")], "I cannot cancel that, sorry.")]  # fmt: skip
    assert _kinds(can) == ["memory.self_contradiction"]


def test_changed_value_backed_by_tool_or_user_is_not_a_contradiction():
    base = [_chat(0, [("user", "price?")], "The total is $120.00.")]
    with_tool = [*base, _tool(1, "get_quote", {}, {"total": 150.0}),
                 _chat(2, [("user", "price?"), ("user", "recheck")], "The total is $150.00.")]  # fmt: skip
    assert _kinds(with_tool) == []
    with_write = [*base, _tool(1, "apply_coupon", {}, {"ok": True}, side_effects=True),
                  _chat(2, [("user", "price?"), ("user", "recheck")], "The total is $90.00.")]  # fmt: skip
    assert _kinds(with_write) == []
    corrected = [*base, _chat(1, [("user", "price?"), ("user", "I added a second item, total $180.00 now?")],
                              "The total is $180.00.")]  # fmt: skip
    assert _kinds(corrected) == []


def test_hedged_and_repeated_statements_do_not_contradict():
    steps = [
        _chat(0, [("user", "q")], "The total is $120.00."),
        _chat(1, [("user", "q"), ("user", "r")], "The total is $120.00. If you add shipping the total would be $130.00."),
        _chat(2, [("user", "q"), ("user", "r"), ("user", "s")], "Is the total $140.00?"),
    ]  # fmt: skip
    assert _kinds(steps) == []


# --- stale tool results -----------------------------------------------------------------


def _stale_steps(reply: str) -> list[Step]:
    return [
        _chat(0, [("user", "status of A1")], "Looking."),
        _tool(1, "get_order", {"order_id": "A1"}, {"status": "pending", "eta": "2026-12-01"}),
        _tool(2, "update_order", {"order_id": "A1"}, {"ok": True}, side_effects=True),
        _tool(3, "get_order", {"order_id": "A1"}, {"status": "shipped", "eta": "2026-12-01"}),
        _chat(4, [("user", "status of A1"), ("assistant", "Looking.")], reply),
    ]


def test_stale_tool_result_detected():
    (f,) = _findings(_stale_steps("Your order is still pending."), "memory.stale_tool_result")
    assert (
        f.severity == "error" and f.steps == [1, 3, 4] and f.key == "stale:get_order:status:pending"
    )
    assert "shipped" in f.message
    assert [e.step_idx for e in f.evidence] == [1, 3, 4]
    assert f.evidence[0].quote == "pending" and f.evidence[1].quote == "shipped"


def test_fresh_tool_result_is_not_stale():
    assert _kinds(_stale_steps("Your order has shipped.")) == []
    assert _kinds(_stale_steps("It was pending earlier but has now shipped.")) == []


def test_stale_tool_result_ignores_volatile_fields_and_ignored_tools():
    steps = [
        _tool(0, "clock", {}, {"timestamp": "2026-11-27T10:00:00", "ok": "alive"}),
        _tool(1, "clock", {}, {"timestamp": "2026-11-27T10:05:00", "ok": "alive"}),
        _chat(2, [("user", "time?")], "It was 2026-11-27T10:00:00."),
    ]
    assert _kinds(steps) == []
    cfg = resolve_config({"memory": {"stale_tool_result": {"ignore_tools": ["get_order"]}}})
    assert _kinds(_stale_steps("Your order is still pending."), cfg) == []


# --- unbounded growth -------------------------------------------------------------------


def _growth(tokens: list[int]) -> list[Step]:
    hist: list[tuple[str, str]] = []
    steps = []
    for i, t in enumerate(tokens):
        hist.append(("user", f"q{i}"))
        steps.append(_chat(i, list(hist), f"a{i}", turn=i + 1, tokens=t))
        hist.append(("assistant", f"a{i}"))
    return steps


def test_superlinear_growth_is_flagged():
    (f,) = _findings(_growth([200, 260, 400, 700, 1300, 2400, 4400]), "memory.unbounded_growth")
    assert f.severity == "warn" and f.step == 6 and "grows faster every turn" in f.message


def test_linear_growth_stable_prompts_and_compaction_are_fine():
    assert _kinds(_growth([200 + 300 * i for i in range(10)])) == []
    assert _kinds(_growth([500] * 8)) == []
    assert _kinds(_growth([200, 400, 900, 2000, 500, 900, 1600, 2600])) == []  # compacted
    assert _kinds(_growth([200, 260, 400])) == []  # too few samples
    cfg = resolve_config({"memory": {"unbounded_growth": {"min_steps": 4}}})
    assert _kinds(_growth([200, 400, 1000, 2600]), cfg) == ["memory.unbounded_growth"]


# --- snapshots --------------------------------------------------------------------------


@crashlab.agent(name="snap-bot", ignore_args=("client",))
def _snap_agent(user: str, client) -> str:
    notes: dict[str, Any] = {"user": None, "cart": []}
    crashlab.snapshot("memory", notes)
    notes["user"] = {"name": "Priya", "api_key": "sk-" + "a1b2c3d4e5f6a7b8c9d0e1f2"}
    notes["cart"].append("book")
    crashlab.snapshot("memory", notes)
    r = client.chat.completions.create(
        model="fake-model", messages=[{"role": "user", "content": user}]
    )
    notes["cart"].append("pen")
    notes["plan"] = "x" * 5000
    crashlab.snapshot("memory", notes)
    crashlab.snapshot("other", [1, 2, 3])
    return r.choices[0].message.content


def _record_snapshots():
    _snap_agent("hi", crashlab.wrap(FakeOpenAI([text_response(1, "hello")])))
    return crashlab.load("last")


def test_snapshots_are_recorded_redacted_and_capped():
    run = _record_snapshots()
    snaps = mem.snapshots(run)
    assert [(s.name, s.idx) for s in snaps] == [
        ("memory", 0),
        ("memory", 1),
        ("memory", 3),
        ("other", 4),
    ]
    assert all(s.name == "crashlab.snapshot" for s in run.steps if s.operation == "custom")
    assert snaps[0].state == {"user": None, "cart": []}
    assert snaps[1].state["user"]["name"] == "Priya"
    assert "a1b2c3d4" not in json.dumps(snaps[1].state)
    assert len(snaps[2].state["plan"]) < 2100 and "chars]" in snaps[2].state["plan"]
    assert [s.name for s in mem.snapshots(run, "other")] == ["other"]
    assert run.usage.llm_calls == 1  # a snapshot is not an LLM call


def test_snapshot_outside_a_run_is_a_noop_and_validates_name():
    assert crashlab.snapshot("x", {"a": 1}) is False
    with pytest.raises(ValueError):
        crashlab.snapshot("  ", {})


def test_oversized_snapshot_keeps_only_a_preview():
    @crashlab.agent(name="big-bot")
    def big() -> str:
        crashlab.snapshot("big", {f"k{i}": "v" * 1900 for i in range(60)})
        return "ok"

    big()
    (sn,) = mem.snapshots(crashlab.load("last"))
    assert sn.truncated and sn.state["_truncated"] is True and sn.state["_chars"] > 20_000
    assert len(json.dumps(sn.state)) < 3000


def test_snapshot_survives_strict_replay():
    src = _record_snapshots()
    client = crashlab.wrap(FakeOpenAI([]))
    replayed = crashlab.replay(src, overrides={"client": client})
    assert [s.state for s in mem.snapshots(replayed)] == [s.state for s in mem.snapshots(src)]
    assert replayed.status == "ok"


def test_state_diff_key_level():
    old = {"user": {"name": "Priya", "tier": "free"}, "cart": ["book"], "n": 1}
    new = {"user": {"name": "Priya", "email": "p@x.io"}, "cart": ["book", "pen"], "n": 2}
    got = {(c.op, c.path): (c.old, c.new) for c in mem.state_diff(old, new)}
    assert got == {
        ("removed", "user.tier"): ("free", None),
        ("added", "user.email"): (None, "p@x.io"),
        ("added", "cart[1]"): (None, "pen"),
        ("changed", "n"): (1, 2),
    }
    assert mem.state_diff({"a": 1}, {"a": 1.0}) == []
    assert mem.state_diff([1], "x")[0].op == "changed"
    assert len(mem.state_diff({}, {str(i): i for i in range(50)}, limit=5)) == 5


def test_snapshot_growth_is_super_linear_unbounded_state():
    steps = []
    for i, n in enumerate([200, 300, 600, 1200, 2600, 5600, 12000]):
        steps.append(Step(run_id="r", idx=i, span_id=f"s{i}", operation="custom",
                          name="crashlab.snapshot", request={"name": "scratch"},
                          attributes={"crashlab.snapshot.name": "scratch", "crashlab.snapshot.chars": n * 4},
                          response={"state": {"x": "y"}}, request_hash=f"h{i}"))  # fmt: skip
    (f,) = _findings(steps, "memory.unbounded_growth")
    assert f.key == "state:scratch" and "scratch" in f.message


# --- recorded and simulated conversations ----------------------------------------------


def _scenario():
    return crashlab.Scenario(
        name="memory",
        turns=[
            crashlab.ScenarioTurn(user="My name is Priya and my order is A1234."),
            crashlab.ScenarioTurn(user="Please check my order."),
            crashlab.ScenarioTurn(user="Thanks."),
        ],
    )


def _chat_agent(use_history: bool):
    texts = iter(
        [
            "Hello Priya, got order A1234.",
            "Order A1234 is on its way.",
            "You're welcome.",
            "You're welcome.",
            "x",
            "y",
            "z",
        ]  # fmt: skip
    )
    client = crashlab.wrap(FakeOpenAI([text_response(i, next(texts)) for i in range(1, 5)]))

    def agent(user_msg: str, history: list) -> str:
        msgs = (
            [*history, {"role": "user", "content": user_msg}]
            if use_history
            else [{"role": "user", "content": user_msg}]
        )
        r = client.chat.completions.create(model="fake-model", messages=msgs)
        return r.choices[0].message.content

    return agent


def test_simulated_conversation_with_history_is_clean():
    result = crashlab.simulate(_chat_agent(True), _scenario())
    assert result.turns_executed == 3
    run = crashlab.load("last")
    assert mem.memory_findings(run) == []
    assert [f for f in run.findings if f.kind.startswith("memory.")] == []


def test_simulated_stateless_agent_loses_context_and_is_caught_live():
    crashlab.simulate(_chat_agent(False), _scenario())
    run = crashlab.load("last")
    found = mem.memory_findings(run)
    assert {f.kind for f in found} == {"memory.context_truncation"}
    assert [f.key for f in found] == ["reset:2", "reset:3"]
    # recording stored the same findings (the live engine ran the memory detectors)
    assert {f.kind for f in run.findings if f.kind.startswith("memory.")} == {
        "memory.context_truncation"
    }


# --- config / registry / diagnoser ------------------------------------------------------


def test_kinds_config_and_taxonomy_mapping():
    assert set(MEMORY_KINDS) <= set(KINDS)
    cfg = resolve_config({"memory": False})
    assert cfg.memory.enabled is False and cfg.enabled
    assert (
        resolve_config({"memory": {"unbounded_growth": False}}).memory.unbounded_growth.enabled
        is False
    )
    assert ChecksConfig().memory.unbounded_growth.min_steps == 6
    with pytest.raises(ValueError):
        ChecksConfig.model_validate({"memory": {"no_such_check": True}})
    from crashlab.diagnose.heuristics import _FINDING_RULES
    from crashlab.models import TaxonomyLabel

    assert all(k in _FINDING_RULES for k in MEMORY_KINDS)
    assert _FINDING_RULES["memory.forgotten_fact"][0] is TaxonomyLabel.TASK_SPEC_VIOLATION
    assert _FINDING_RULES["memory.stale_tool_result"][0] is TaxonomyLabel.TOOL_ERROR_MISHANDLED


def test_a_broken_step_shape_never_crashes_the_detectors():
    odd = Step(run_id="r", idx=0, span_id="s", operation="chat", name="m",
               request={"messages": ["plain string", {"role": None}, {"content": [1, {"text": 5}]}]},
               response={"messages": [None, "x"]}, request_hash="h")  # fmt: skip
    assert _kinds([odd, odd.model_copy(update={"idx": 1})]) == []


# --- judge confirmation -----------------------------------------------------------------


def _judge_fn(score: float):
    def fake(messages):
        assert any("memory.forgotten_fact" in m["content"] for m in messages)
        return json.dumps({"score": score, "reasoning": "checked", "evidence": []})

    return fake


def _forgotten_run() -> Run:
    u = "My email is priya@example.com."
    return _run([_chat(0, [("user", u)], "ok"),
                 _chat(1, [("user", u), ("assistant", "ok"), ("user", "hi")], "What is your email address?")])  # fmt: skip


def test_confirm_with_judge_confirms_or_dismisses():
    run = _forgotten_run()
    found = mem.memory_findings(run)
    assert [f.kind for f in found] == ["memory.forgotten_fact"]
    (yes,) = mem.confirm_with_judge(run, found, judge_fn=_judge_fn(0.0))
    (no,) = mem.confirm_with_judge(run, found, judge_fn=_judge_fn(1.0))
    assert yes["confirmed"] is True and yes["kind"] == "memory.forgotten_fact"
    assert no["confirmed"] is False and no["score"] == 1.0


# --- CLI --------------------------------------------------------------------------------


def test_help_is_ansi_clean_and_listed():
    for cmd, needle in (("memory", "--judge"), ("state", "--diff")):
        res = runner.invoke(app, [cmd, "--help"])
        out = _plain(res.output)
        assert res.exit_code == 0 and needle in out and "--json" in out
    top = _plain(runner.invoke(app, ["--help"]).output)
    assert "memory" in top and "state" in top and "Diagnose and fix" in top


def test_cli_state_timeline_diff_and_json():
    _record_snapshots()
    res = runner.invoke(app, ["state", "last"], env={"COLUMNS": "200"})
    out = _plain(res.output)
    assert res.exit_code == 0 and "state snapshots" in out and "memory" in out and "other" in out
    res = runner.invoke(
        app, ["state", "last", "--diff", "--name", "memory"], env={"COLUMNS": "200"}
    )
    out = _plain(res.output)
    assert "(first snapshot)" in out and "changed user" in out and "added cart[0]" in out
    assert "a1b2c3d4" not in out
    res = runner.invoke(app, ["state", "last", "--diff", "--json"])
    data = json.loads(res.output)
    assert data["ok"] and len(data["snapshots"]) == 4
    assert data["snapshots"][0]["first"] is True
    assert {c["path"] for c in data["snapshots"][1]["changes"]} >= {"user", "cart[0]"}


def test_cli_state_no_snapshots_and_missing_run():
    res = runner.invoke(app, ["state", "nope-not-a-run"])
    assert res.exit_code == 2
    res = runner.invoke(app, ["state", "nope-not-a-run", "--json"])
    assert res.exit_code == 2 and json.loads(res.output)["ok"] is False
    crashlab.simulate(_chat_agent(True), _scenario())
    res = runner.invoke(app, ["state", "last"])
    assert res.exit_code == 0 and "no state snapshots" in _plain(res.output)


def test_cli_memory_clean_findings_json_and_missing():
    crashlab.simulate(_chat_agent(True), _scenario())
    res = runner.invoke(app, ["memory", "last"])
    assert res.exit_code == 0 and "no memory problems" in _plain(res.output)
    crashlab.simulate(_chat_agent(False), _scenario())
    res = runner.invoke(app, ["memory", "last"], env={"COLUMNS": "220"})
    out = _plain(res.output)
    assert (
        res.exit_code == 1 and "memory.context_truncation" in out and "2 memory problem(s)" in out
    )
    assert "step 0" in out or "step 1" in out
    res = runner.invoke(app, ["memory", "last", "--json"])
    data = json.loads(res.output)
    assert res.exit_code == 1 and data["ok"] is False
    assert {f["kind"] for f in data["findings"]} == {"memory.context_truncation"}
    assert data["findings"][0]["evidence"][0]["quote"]
    res = runner.invoke(app, ["memory", "nope-not-a-run"])
    assert res.exit_code == 2


def test_cli_memory_judge(monkeypatch):
    crashlab.simulate(_chat_agent(False), _scenario())
    # no judge model configured: a clean error, exit 2
    res = runner.invoke(app, ["memory", "last", "--judge"])
    assert res.exit_code == 2 and "model" in _plain(res.output).lower()
    real = mem.confirm_with_judge

    def offline(run, findings, **kw):
        return real(
            run, findings, judge_fn=lambda m: json.dumps({"score": 1.0, "reasoning": "fine"}), **kw
        )

    monkeypatch.setattr(mem, "confirm_with_judge", offline)
    res = runner.invoke(app, ["memory", "last", "--judge"], env={"COLUMNS": "220"})
    out = _plain(res.output)
    assert res.exit_code == 0 and "dismissed" in out and "0 memory problem(s)" in out
    monkeypatch.setattr(
        mem, "confirm_with_judge",
        lambda run, findings, **kw: real(run, findings, judge_fn=lambda m: json.dumps({"score": 0.0, "reasoning": "real"}), **kw),
    )  # fmt: skip
    res = runner.invoke(app, ["memory", "last", "--judge", "--json"])
    data = json.loads(res.output)
    assert res.exit_code == 1 and all(v["confirmed"] for v in data["judge"])
