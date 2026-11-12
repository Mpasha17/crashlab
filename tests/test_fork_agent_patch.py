"""Patches that target one agent: ``SystemPromptPatch(..., agent="critic")`` edits only
that agent's LLM calls in a fork, even when the agents share a prompt text (``when=``
cannot tell them apart).

Live finding (AutoGen writer + critic, nov13): fixing the critic meant matching its
prompt text with ``when=``; agents built from one template have no distinguishing text."""

from __future__ import annotations

import pytest

import crashlab
from crashlab import patch as P
from tests.fakes import FakeOpenAI, text_response

SYS = "You are a helpful assistant."


def _ask(client, text):
    r = client.chat.completions.create(
        model="m", messages=[{"role": "system", "content": SYS}, {"role": "user", "content": text}]
    )
    return r.choices[0].message.content


@crashlab.agent(name="two-agents", ignore_args=["writer", "critic"])
def two_agents(q: str, writer=None, critic=None) -> str:
    draft = _ask(writer, q)
    return _ask(critic, draft)


def _clients(w: str, c: str):
    wf, cf = FakeOpenAI([text_response(1, w)]), FakeOpenAI([text_response(2, c)])
    return wf, cf, crashlab.wrap(wf, agent="writer"), crashlab.wrap(cf, agent="critic")


def _record():
    _, _, w, c = _clients("draft", "APPROVE")
    two_agents("haiku please", writer=w, critic=c)
    return crashlab.load("last")


def test_fork_applies_the_patch_only_to_the_named_agent():
    src = _record()
    wf, cf, w, c = _clients("draft", "too long")
    patch = P.SystemPromptPatch(op="append", text="Count syllables.", agent="critic")
    run = crashlab.fork(src, at_step=0, patch=patch, overrides={"writer": w, "critic": c})
    assert wf.calls[0]["messages"][0]["content"] == SYS
    assert cf.calls[0]["messages"][0]["content"] == SYS + "\nCount syllables."
    assert run.output == "too long"
    assert "[agent critic]" in run.patch_description


def test_preview_checks_the_agent_exists_from_the_fork_step():
    src = _record()
    patch = P.SystemPromptPatch(op="append", text="x", agent="critc")
    with pytest.raises(P.PatchError, match=r"no LLM call of agent 'critc'.*writer, critic"):
        crashlab.fork(src, at_step=0, patch=patch)
    late = P.SystemPromptPatch(op="append", text="x", agent="writer")
    with pytest.raises(P.PatchError, match="from step 1 on"):
        crashlab.fork(src, at_step=1, patch=late)
    diffs = P.preview_patches(P.as_patchset(P.SystemPromptPatch(op="append", text="x",
                                                                agent="critic")), src)  # fmt: skip
    assert [t for t, _ in diffs] == ["step 1 m request"]


def test_agent_field_round_trips_and_is_validated():
    ps = P.as_patchset({"kind": "system_prompt", "op": "append", "text": "x", "agent": "critic"})
    assert ps.patches[0].agent == "critic"
    assert '"agent":"critic"' in ps.to_json()
    assert "agent" not in P.as_patchset({"kind": "system_prompt", "text": "x"}).to_json()
    with pytest.raises(ValueError, match="researcher#2"):
        P.SystemPromptPatch(text="x", agent="researcher#2")
    with pytest.raises(ValueError, match="agent"):
        P.as_patchset({"kind": "tool_result", "step": 1, "result": 1, "agent": "critic"})
    with pytest.raises(ValueError, match="agent"):
        P.SystemPromptPatch(text="x", agent=" ")


def test_unlabelled_requests_are_not_edited_by_an_agent_patch():
    ps = P.as_patchset(P.SystemPromptPatch(op="append", text="x", agent="critic"))
    req = {"model": "m", "messages": [{"role": "system", "content": SYS}]}
    _, applied = ps.apply_chat("openai.chat.completions", 0, req)
    assert applied == []
    _, applied = ps.apply_chat("openai.chat.completions", 0, req, agent="critic")
    assert applied == [0]
