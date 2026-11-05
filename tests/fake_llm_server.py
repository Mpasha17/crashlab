"""Deterministic fake OpenAI chat-completions endpoint (httpx MockTransport) with SSE
streaming. Brain: last msg user -> parallel tool calls; last msg tool -> final text."""

import json

import httpx

TOOLS_FOR = {
    "default": [
        ("get_price", {"item": "widget"}),
        ("get_price", {"item": "gadget"}),
        ("place_order", {"item": "widget", "quantity": 2}),
    ]
}


def _reply(body):
    msgs = body["messages"]
    last = msgs[-1]
    names = {t["function"]["name"] for t in body.get("tools") or []}
    if last["role"] == "user" and names:
        calls = [(n, a) for n, a in TOOLS_FOR["default"] if n in names] or [(sorted(names)[0], {})]
        turn = sum(m["role"] == "user" for m in msgs)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{turn}_{i}",
                    "type": "function",
                    "function": {"name": n, "arguments": json.dumps(a)},
                }
                for i, (n, a) in enumerate(calls)
            ],
        }
    tool_txt = " ".join(str(m.get("content")) for m in msgs if m["role"] == "tool")[-200:]
    return {"role": "assistant", "content": f"turn {len(msgs)} done: {tool_txt}"}


def _completion(msg):
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


def _sse(msg):
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    out = []

    def chunk(delta, fin=None):
        out.append({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]})

    chunk({"role": "assistant", "content": ""})
    if msg.get("tool_calls"):
        for i, c in enumerate(msg["tool_calls"]):
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["function"]["name"], "arguments": ""},
                        }
                    ]
                }
            )
            a = c["function"]["arguments"]
            for j in range(0, len(a), 7):
                chunk({"tool_calls": [{"index": i, "function": {"arguments": a[j : j + 7]}}]})
        chunk({}, "tool_calls")
    else:
        t = msg["content"]
        for j in range(0, len(t), 9):
            chunk({"content": t[j : j + 9]})
        chunk({}, "stop")
    out.append(
        {
            **base,
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
    )
    return "".join(f"data: {json.dumps(c)}\n\n" for c in out) + "data: [DONE]\n\n"


def handler(req):
    body = json.loads(req.content)
    msg = _reply(body)
    if body.get("stream"):
        return httpx.Response(200, text=_sse(msg), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=_completion(msg))


def transport():
    return httpx.MockTransport(handler)
