"""Streams closed early keep the partial output (recorded and replayed)."""

from __future__ import annotations

import pytest

import crashlab
from tests._transport import Script, httpx_module, sse
from tests.fakes import responses_text


@pytest.fixture
def openai_mod():
    return pytest.importorskip("openai")


def _client(mod, script):
    httpx = httpx_module()
    return mod.OpenAI(
        api_key="sk-test",
        base_url="http://localhost:11434/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(script.handler(httpx))),
    )


def _events(final, n):
    ev = [{"type": "response.created", "sequence_number": 0,
           "response": {**final, "status": "in_progress", "output": []}},
          {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
           "item": {"type": "message", "id": "msg_2", "role": "assistant", "status": "in_progress",
                    "content": []}}]  # fmt: skip
    ev += [{"type": "response.output_text.delta", "sequence_number": i + 2, "item_id": "msg_2",
            "output_index": 0, "content_index": 0, "delta": "word "} for i in range(n)]  # fmt: skip
    ev.append({"type": "response.completed", "sequence_number": n + 2, "response": final})
    return ev


@crashlab.agent(name="resp-early-close", ignore_args=["client"])
def read_three_deltas(q: str, client=None) -> str:
    stream = client.responses.create(model="m", input=q, stream=True)
    got = []
    for event in stream:
        if event.type == "response.output_text.delta":
            got.append(event.delta)
            if len(got) == 3:
                break
    stream.close()
    return "".join(got)


def test_responses_stream_closed_early_keeps_partial_text(openai_mod, crashlab_dir):
    final = responses_text(2, "word " * 20)
    script = Script([(sse(_events(final, 20), named=True), "text/event-stream")])
    out = read_three_deltas("hi", client=crashlab.wrap(_client(openai_mod, script)))
    assert out == "word word word "
    run = crashlab.load("last")
    step = run.steps[0]
    assert step.attributes["crashlab.stream.complete"] is False
    # the deltas that arrived are recorded, not an empty output
    assert step.response["messages"][0]["parts"][0]["content"] == "word word word "
    # and the replay serves them as text deltas again
    replayed = crashlab.replay(
        run, overrides={"client": crashlab.wrap(_client(openai_mod, Script([500])))}
    )
    assert replayed.output == out


def test_responses_assembler_partial_function_call():
    from crashlab.capture.streams import ResponsesAssembler

    asm = ResponsesAssembler()
    for ev in [
        {"type": "response.created",
         "response": {"id": "r", "output": [], "status": "in_progress"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "function_call", "id": "fc", "call_id": "c1", "name": "f",
                  "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"a": '},
    ]:  # fmt: skip
        asm._add(ev)
    item = asm.result()["output"][0]
    assert item["arguments"] == '{"a": ' and item["status"] == "incomplete"
