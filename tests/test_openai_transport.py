"""Exercise the real openai SDK (sync + async) through a mocked httpx transport."""

from __future__ import annotations

import json

import pytest

import crashlab

openai = pytest.importorskip("openai")
try:  # openai>=3 ships on httpx2; older releases use httpx
    import httpx2 as httpx
except ImportError:
    httpx = pytest.importorskip("httpx")

from examples.refund_bot import agent as refund  # noqa: E402


def _completion(n: int, message: dict, finish: str) -> dict:
    return {
        "id": f"chatcmpl-{n}",
        "object": "chat.completion",
        "created": 1760700000,
        "model": "gpt-test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
    }


def _tool_msg(n: int, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": f"call_{n}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }  # fmt: skip


SCRIPT = [
    _completion(1, _tool_msg(1, "get_order", {"order_id": "A17"}), "tool_calls"),
    _completion(2, _tool_msg(2, "issue_refund", {"order_id": "A17", "amount": 49}), "tool_calls"),
    _completion(3, {"role": "assistant", "content": "Refunded $49."}, "stop"),
]
API_KEY = "sk-test-abcdefghijklmnopqrstuvwxyz0123"


def _handler(seen: list):
    script = list(SCRIPT)

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json=script.pop(0))

    return handle


def test_real_openai_client_sync(crashlab_dir):
    seen: list = []
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url="http://localhost:11434/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler(seen))),
    )
    answer = refund.run_agent("Refund A17", client=crashlab.wrap(client), model="gpt-test")
    assert answer == "Refunded $49."
    assert seen[0].headers["authorization"] == f"Bearer {API_KEY}"
    run = crashlab.load("last")
    assert [s.operation for s in run.steps] == ["chat", "execute_tool"] * 2 + ["chat"]
    assert run.steps[0].attributes["server.address"] == "localhost"
    assert run.steps[0].attributes["gen_ai.response.id"] == "chatcmpl-1"
    assert run.steps[0].response["tool_calls"][0]["name"] == "get_order"
    assert run.usage.total_tokens == 165
    assert run.env.get("openai") == openai.__version__
    on_disk = "".join(p.read_text() for p in crashlab_dir.rglob("*.json*"))
    assert API_KEY not in on_disk


async def test_real_openai_client_async(crashlab_dir):
    seen: list = []
    client = openai.AsyncOpenAI(
        api_key=API_KEY,
        base_url="http://localhost:11434/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler(seen))),
    )
    wrapped = crashlab.wrap(client)
    async with crashlab.record("async-openai", input={"q": "x"}) as run:
        resp = await wrapped.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "x"}]
        )
        run.output = resp.choices[0].message.tool_calls[0].function.name
    loaded = crashlab.load(run.run_id)
    assert loaded.output == "get_order"
    assert loaded.steps[0].response["finish_reason"] == "tool_calls"
    assert loaded.steps[0].attributes["gen_ai.usage.output_tokens"] == 5


def test_real_openai_http_error_recorded(crashlab_dir):
    def handle(request):
        return httpx.Response(500, json={"error": {"message": "boom", "type": "server_error"}})

    client = openai.OpenAI(
        api_key=API_KEY,
        base_url="http://localhost:1/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    with pytest.raises(openai.InternalServerError):
        refund.run_agent("Refund A17", client=crashlab.wrap(client), model="gpt-test")
    run = crashlab.load("last")
    assert run.status == "error"
    assert run.steps[0].error.status_code == 500
    assert run.steps[0].error.type == "openai.InternalServerError"
