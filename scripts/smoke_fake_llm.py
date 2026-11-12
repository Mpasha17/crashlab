"""Deterministic fake OpenAI chat-completions server for scripts/smoke.sh (stdlib only).

Plays a compliant refund agent: looks the order up, refunds it unless the system prompt
carries the 30-day rule and the order is outside it, then answers. Writes its port to the
file given as argv[1] and serves until killed. No keys, no network beyond 127.0.0.1."""

from __future__ import annotations

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WINDOW_RULE = "at most 30 days ago"


def _brain(body: dict) -> dict:
    msgs = body["messages"]
    system = " ".join(str(m.get("content")) for m in msgs if m["role"] == "system")
    user = " ".join(str(m.get("content")) for m in msgs if m["role"] == "user")
    tools = [m for m in msgs if m["role"] == "tool"]
    order_id = (re.search(r"\b([A-Z]\d\d)\b", user) or re.search(r"(A17)", "A17")).group(1)
    turn = len(msgs)

    def call(name: str, args: dict) -> dict:
        fn = {"name": name, "arguments": json.dumps(args)}
        tc = {"id": f"call_{turn}", "type": "function", "function": fn}
        return {"role": "assistant", "content": None, "tool_calls": [tc]}

    if not tools:
        return call("get_order", {"order_id": order_id})
    if len(tools) == 1:
        try:
            order = json.loads(tools[0]["content"])
        except (TypeError, ValueError):
            return {"role": "assistant", "content": "The order service is down, try later."}
        days = order.get("days_since_delivery")
        late = order.get("status") != "delivered" or (days or 0) > 30
        if late and WINDOW_RULE in system:
            return {"role": "assistant", "content": f"Sorry, order {order_id} can't be refunded."}
        return call("issue_refund", {"order_id": order_id, "amount": order.get("total", 0)})
    return {"role": "assistant", "content": f"Done: your refund for {order_id} was issued."}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: object) -> None:  # quiet
        pass

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        msg = _brain(body)
        finish = "tool_calls" if msg.get("tool_calls") else "stop"
        out = json.dumps({
            "id": f"chatcmpl-smoke{time.monotonic_ns()}", "object": "chat.completion",
            "created": int(time.time()), "model": body.get("model", "smoke-model"),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }).encode()  # fmt: skip
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    with open(sys.argv[1], "w") as f:
        f.write(str(server.server_address[1]))
    server.serve_forever()
