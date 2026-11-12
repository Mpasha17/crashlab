"""empty_reply runtime check: the model ended its turn with no text and no tool call.

Live finding (LlamaIndex AgentWorkflow on Groq gpt-oss-120b): the research agent's last
turn was reasoning only ("...So we should hand off to WriteAgent"), with no handoff call
and no text. The workflow returned '' with status ok and no runtime check said so."""

from __future__ import annotations

from typing import Any

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage as Message
from openai.types.completion_usage import CompletionUsage

import crashlab
from crashlab.checks import ChecksConfig, evaluate, resolve_config
from tests.fakes import FakeOpenAI


@pytest.fixture(autouse=True)
def _no_pyproject(monkeypatch):
    monkeypatch.setenv("CRASHLAB_NO_PYPROJECT", "1")


def _chat(idx: int, response: dict[str, Any], **kw: Any) -> crashlab.Step:
    return crashlab.Step(
        run_id="r", idx=idx, span_id=f"s{idx}", operation="chat", name="m",
        request_hash="h", response=response, **kw,
    )  # fmt: skip


def _reply(*parts: dict[str, Any], finish: str | None = "stop", **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"messages": [{"role": "assistant", "parts": list(parts)}], **extra}
    if finish is not None:
        out["finish_reason"] = finish
    return out


def _empty(run: crashlab.Run, cfg: ChecksConfig | None = None) -> list[Any]:
    return [f for f in evaluate(run, cfg or ChecksConfig()) if f.kind == "empty_reply"]


REASONING = {"type": "reasoning", "content": "So we should hand off to WriteAgent."}


def test_reasoning_only_reply_is_flagged_with_the_reasoning_as_evidence():
    run = crashlab.Run(run_id="r", name="t", steps=[_chat(0, _reply(REASONING))])
    (f,) = _empty(run)
    assert f.severity == "warn" and f.step == 0 and f.steps == [0]
    assert "no text and no tool call" in f.message and "finish_reason=stop" in f.message
    assert f.evidence[0].field == "response.reasoning"
    assert "hand off to WriteAgent" in f.evidence[0].quote


@pytest.mark.parametrize(
    "response",
    [
        _reply(),
        _reply({"type": "text", "content": ""}),
        _reply({"type": "text", "content": "   \n"}),
        {"messages": [], "finish_reason": "stop"},
        {"messages": [{"role": "assistant", "content": None}], "finish_reason": "end_turn"},
    ],
)
def test_blank_replies_are_flagged(response):
    (f,) = _empty(crashlab.Run(run_id="r", name="t", steps=[_chat(0, response)]))
    assert f.evidence == []


@pytest.mark.parametrize(
    "step",
    [
        _chat(0, _reply({"type": "text", "content": "Python 3.14.0 shipped."})),
        _chat(0, _reply(REASONING, tool_calls=[{"id": "c1", "name": "handoff"}])),
        _chat(0, _reply({"type": "refusal", "content": "I can't help with that."})),
        _chat(0, {"messages": [{"role": "assistant", "content": "hi"}], "finish_reason": "stop"}),
        # cut off at the output limit: length_cutoff reports it, not empty_reply
        _chat(0, _reply(REASONING, finish="length")),
        # no finish reason recorded (older recordings, unknown providers): stay quiet
        _chat(0, _reply(REASONING, finish=None)),
        # the call failed: there is no reply to judge
        _chat(0, {}, error={"type": "APIError", "message": "boom"}),
        _chat(0, None),
    ],
)
def test_quiet_on_real_replies_tool_calls_cutoffs_and_errors(step):
    assert _empty(crashlab.Run(run_id="r", name="t", steps=[step])) == []


def test_an_interrupted_stream_is_not_called_empty():
    step = _chat(0, _reply(REASONING), attributes={"crashlab.stream": True,
                 "crashlab.stream.complete": False})  # fmt: skip
    assert _empty(crashlab.Run(run_id="r", name="t", steps=[step])) == []


def test_can_be_disabled():
    run = crashlab.Run(run_id="r", name="t", steps=[_chat(0, _reply(REASONING))])
    assert _empty(run, resolve_config({"empty_reply": False})) == []


def test_live_wrapped_client_empty_completion_is_flagged():
    empty = ChatCompletion(
        id="chatcmpl-1", model="m", object="chat.completion", created=0,
        choices=[Choice(index=0, message=Message(role="assistant", content=""),
                        finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )  # fmt: skip
    client = crashlab.wrap(FakeOpenAI([empty]))
    with crashlab.record("empty"):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    run = crashlab.load("last")
    (f,) = run.findings_of("empty_reply")
    assert f.step == 0 and run.status == "ok"


def test_heuristic_diagnosis_points_at_the_empty_turn():
    from crashlab.diagnose.heuristics import heuristic_raw

    tool = crashlab.Step(
        run_id="r", idx=1, span_id="s1", operation="execute_tool", name="lookup_release",
        request_hash="h", request={"args": {"project": "Python"}}, response={"result": "3.14"},
    )  # fmt: skip
    first = _chat(0, _reply(tool_calls=[{"id": "c1", "name": "lookup_release", "arguments": "{}"}]))
    run = crashlab.Run(
        run_id="r", name="t", steps=[first, tool, _chat(2, _reply(REASONING))], outcome="fail",
        checks=[crashlab.CheckResult(name="report_keeps_date", status="fail", message="lost")],
    )  # fmt: skip
    run.findings = evaluate(run, ChecksConfig())
    raw = heuristic_raw(run)
    assert raw["failing_step"] == 2 and "empty_reply" in raw["root_cause"], raw
