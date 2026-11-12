"""Runtime checks in multi-agent runs. Found on a live LangGraph research graph: the
writer agent repeated the researcher's final answer verbatim (a normal hand-over) and
step_repetition reported "the model gave the same reply 2 times in a row". With agent
labels, reply / request repetition is judged per agent; unlabelled runs are unchanged."""

from __future__ import annotations

import pytest

import crashlab
from tests.fakes import FakeOpenAI, text_response

pytestmark = pytest.mark.requires_providers("pydantic")

FACT = "Rust 1.90 was released on 18 September 2025."


def _two_agents(label: bool, same_agent: bool = False) -> crashlab.Run:
    @crashlab.agent(name="handover")
    def handover() -> None:
        a = "researcher"
        b = "researcher" if same_agent else "writer"
        r = FakeOpenAI([text_response(1, FACT)])
        w = FakeOpenAI([text_response(2, FACT)])
        r = crashlab.wrap(r, agent=a) if label else crashlab.wrap(r)
        w = crashlab.wrap(w, agent=b) if label else crashlab.wrap(w)
        r.chat.completions.create(model="m", messages=[{"role": "user", "content": "rust?"}])
        w.chat.completions.create(model="m", messages=[{"role": "user", "content": FACT}])

    handover()
    return crashlab.load("last")


def _repetition(run):
    return [f for f in run.findings if f.kind == "step_repetition"]


def test_writer_repeating_researcher_is_not_a_repeated_reply():
    assert _repetition(_two_agents(label=True)) == []


def test_same_agent_repeating_itself_is_still_flagged():
    found = _repetition(_two_agents(label=True, same_agent=True))
    assert found and "same reply 2 times" in found[0].message


def test_unlabelled_runs_keep_the_old_behaviour():
    found = _repetition(_two_agents(label=False))
    assert found and "same reply 2 times" in found[0].message


def test_identical_requests_from_two_agents_are_not_a_loop():
    @crashlab.agent(name="twins")
    def twins() -> None:
        msgs = [{"role": "user", "content": "summarise the ticket"}]
        for i, name in enumerate(("a", "b", "c")):
            c = crashlab.wrap(FakeOpenAI([text_response(i, f"reply {i}")]), agent=name)
            c.chat.completions.create(model="m", messages=msgs)

    twins()
    run = crashlab.load("last")
    assert not [f for f in _repetition(run) if "same LLM request" in f.message]

    @crashlab.agent(name="loop")
    def loop() -> None:
        msgs = [{"role": "user", "content": "summarise the ticket"}]
        c = crashlab.wrap(FakeOpenAI([text_response(i, f"r{i}") for i in range(3)]), agent="a")
        for _ in range(3):
            c.chat.completions.create(model="m", messages=msgs)

    loop()
    run = crashlab.load("last")
    assert [f for f in _repetition(run) if "same LLM request" in f.message]
