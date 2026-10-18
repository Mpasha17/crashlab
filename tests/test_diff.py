"""crashlab.diff: step alignment, statuses, volatile values, checks, relations."""

from __future__ import annotations

import crashlab
from crashlab.compare import compare_steps
from tests.fakes import FakeOpenAI, text_response, tool_call_response


@crashlab.check("diffbot")
def says_done(run: crashlab.Run) -> None:
    """The answer is 'done'."""
    assert run.output == "done", f"answered {run.output!r}"


def _run(script, *, system="s", extra=False, tool_args=None):
    client = crashlab.wrap(FakeOpenAI(script))

    @crashlab.tool()
    def lookup(key: str) -> dict:
        return {"key": key}

    with crashlab.record("diffbot") as run:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": "go"}]
        client.chat.completions.create(model="m", messages=msgs)
        lookup(**(tool_args or {"key": "a"}))
        out = client.chat.completions.create(model="m", messages=msgs)
        if extra:
            client.chat.completions.create(model="m", messages=msgs)
        run.output = out.choices[0].message.content
    return crashlab.load(run.run_id)


def test_identical_runs_with_different_ids_are_the_same():
    a = _run([tool_call_response(1, "lookup", {"key": "a"}), text_response(2, "done")])
    b = _run([tool_call_response(7, "lookup", {"key": "a"}), text_response(8, "done")])
    assert a.steps[0].response["id"] != b.steps[0].response["id"]  # volatile, ignored
    d = crashlab.diff(a, b)
    assert d.identical_steps and d.first_divergence is None
    assert d.summary() == "all 3 steps are the same"
    assert d.relation is None and not any(c.changed for c in d.checks)


def test_request_response_only_and_check_changes():
    a = _run([tool_call_response(1, "lookup", {"key": "a"}), text_response(2, "done")])
    b = _run(
        [tool_call_response(1, "lookup", {"key": "a"}), text_response(2, "nope")],
        system="t",
        extra=False,
    )
    d = crashlab.diff(a, b)
    assert [s.status for s in d.steps] == ["request", "same", "request"]
    assert d.steps[0].fields == ["system"]
    assert "-s" in d.steps[0].detail() and "+t" in d.steps[0].detail()
    assert d.summary().startswith("first divergence at step 0: different request (system)")
    (check,) = d.checks
    assert check.name == "says_done" and check.changed
    assert (check.a.status, check.b.status) == ("pass", "fail")

    c = _run([tool_call_response(1, "lookup", {"key": "a"}), text_response(2, "other")])
    d = crashlab.diff(a, c)
    assert [s.status for s in d.steps] == ["same", "same", "response"]
    assert d.steps[2].fields == ["messages"] and "+" in d.steps[2].detail()


def test_extra_and_missing_steps_and_different_calls():
    a = _run([tool_call_response(1, "lookup", {"key": "a"}), text_response(2, "done")])
    b = _run(
        [
            tool_call_response(1, "lookup", {"key": "a"}),
            text_response(2, "done"),
            text_response(3, "more"),
        ],
        extra=True,
    )
    d = crashlab.diff(a, b)
    assert [s.status for s in d.steps] == ["same", "same", "same", "only_b"]
    assert "only B has this step" in d.summary()
    assert crashlab.diff(b, a).steps[-1].status == "only_a"
    chat, tool = a.steps[0], a.steps[1]
    assert compare_steps(chat, tool, 0).status == "different"
    assert "A: chat" in compare_steps(chat, tool, 0).detail()


def test_relations():
    a = _run([tool_call_response(1, "lookup", {"key": "a"}), text_response(2, "done")])
    b = a.model_copy(update={"run_id": "B", "forked_from": a.run_id, "fork_step": 2})
    c = a.model_copy(update={"run_id": "C", "forked_from": a.run_id, "fork_step": 0})
    r = a.model_copy(update={"run_id": "R", "replay_of": a.run_id})
    assert crashlab.diff(a, b).relation == "B is a fork of A at step 2"
    assert crashlab.diff(b, a).relation == "A is a fork of B at step 2"
    assert crashlab.diff(b, c).relation == f"both are forks of {a.run_id}"
    assert crashlab.diff(a, r).relation == "B is a replay of A"
