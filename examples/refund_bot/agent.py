"""refund-bot: a small OpenAI-compatible tool-calling agent loop instrumented with crashlab.

Run it (from the repo root) against any OpenAI-compatible endpoint:

    OPENAI_API_KEY=sk-...            python -m examples.refund_bot "Refund order A17, it broke"
    OPENAI_BASE_URL=http://localhost:11434/v1 MODEL=qwen2.5:7b \
                                     python -m examples.refund_bot "Refund order A17, it broke"

Set FAULT_HTML_502=1 (or pass --fault-html-502) to make get_order return an HTML 502 page
instead of JSON, the classic "garbage tool result" failure.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import crashlab

SYSTEM_PROMPT = """You are the refund assistant for ShopCo customer support.

Refund policy:
1. Always look up the order with get_order before deciding anything.
2. Only refund orders whose status is "delivered" and that were delivered at most 30 days ago.
3. Never refund more than the order total.
4. Issue at most one refund per order.

Use the tools to act, then reply to the customer in one or two short sentences."""

# --- mock in-memory order DB --------------------------------------------------------

ORDERS: dict[str, dict[str, Any]] = {
    "A17": {"order_id": "A17", "item": "Ceramic teapot", "total": 49.0, "currency": "USD",
            "status": "delivered", "days_since_delivery": 5},
    "B22": {"order_id": "B22", "item": "Desk lamp", "total": 89.5, "currency": "USD",
            "status": "shipped", "days_since_delivery": None},
    "C03": {"order_id": "C03", "item": "Running shoes", "total": 120.0, "currency": "USD",
            "status": "delivered", "days_since_delivery": 45},
}  # fmt: skip
REFUNDS: list[dict[str, Any]] = []

HTML_502 = """<!DOCTYPE html>
<html><head><title>502 Bad Gateway</title></head>
<body><center><h1>502 Bad Gateway</h1></center><hr><center>nginx/1.25.3</center></body>
</html>"""


def fault_html_502() -> bool:
    return os.environ.get("FAULT_HTML_502", "").lower() in ("1", "true", "yes", "on")


def reset_db() -> None:
    REFUNDS.clear()


# --- tools --------------------------------------------------------------------------


@crashlab.tool()
def get_order(order_id: str) -> dict[str, Any] | str:
    """Look up an order by id. Returns the order as JSON."""
    if fault_html_502():
        return HTML_502
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise KeyError(f"order {order_id!r} not found")
    return dict(order)


@crashlab.tool(side_effects=True)
def issue_refund(order_id: str, amount: float) -> dict[str, Any]:
    """Issue a refund for an order. This moves real money."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise KeyError(f"order {order_id!r} not found")
    if amount <= 0 or amount > order["total"]:
        raise ValueError(f"refund amount {amount} must be > 0 and <= order total {order['total']}")
    refund = {"refund_id": f"R{len(REFUNDS) + 1:04d}", "order_id": order["order_id"],
              "amount": float(amount), "status": "issued"}  # fmt: skip
    REFUNDS.append(refund)
    return refund


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order by id. Returns the order as JSON.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "e.g. A17"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund for an order. Moves real money.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number", "description": "Refund amount in USD"},
                },
                "required": ["order_id", "amount"],
            },
        },
    },
]
TOOL_FUNCS = {"get_order": get_order, "issue_refund": issue_refund}


# --- client -------------------------------------------------------------------------


def make_client() -> Any:
    """OpenAI client from OPENAI_BASE_URL / OPENAI_API_KEY, wrapped by crashlab."""
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL") or None
    api_key = os.environ.get("OPENAI_API_KEY") or ("ollama" if base_url else None)
    return crashlab.wrap(OpenAI(base_url=base_url, api_key=api_key))


def default_model() -> str:
    return os.environ.get("MODEL", "gpt-4o-mini")


# --- the agent loop (plain OpenAI tool calling; nothing crashlab-specific inside) -------


def _assistant_message(msg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out


def _run_tool(name: str, arguments: str) -> str:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"ERROR: arguments are not valid JSON: {exc}"
    try:
        result = fn(**args)
    except Exception as exc:  # the model sees the error and can recover
        return f"ERROR: {type(exc).__name__}: {exc}"
    return result if isinstance(result, str) else json.dumps(result)


@crashlab.agent(name="refund-bot", ignore_args=("client",))
def run_agent(
    user_msg: str, *, client: Any = None, model: str | None = None, max_turns: int = 8
) -> str:
    client = client or make_client()
    model = model or default_model()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0
        )
        msg = resp.choices[0].message
        messages.append(_assistant_message(msg))
        if not msg.tool_calls:
            return msg.content or ""
        for tc in msg.tool_calls:
            content = _run_tool(tc.function.name, tc.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    return "Sorry, I could not finish this request."


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="refund-bot crashlab demo")
    parser.add_argument(
        "message", nargs="?", default="Hi, please refund order A17, it arrived broken."
    )
    parser.add_argument("--fault-html-502", action="store_true", help="get_order returns HTML 502")
    parser.add_argument("--model", default=None, help="model name (default: $MODEL or gpt-4o-mini)")
    ns = parser.parse_args(argv)
    if ns.fault_html_502:
        os.environ["FAULT_HTML_502"] = "1"
    answer = run_agent(ns.message, model=ns.model)
    print(answer)
    if REFUNDS:
        print(f"[refunds issued: {REFUNDS}]")
    print("Recorded. Inspect with: crashlab ls && crashlab show last")


if __name__ == "__main__":
    main()
