"""refund-bot: a small tool-calling agent loop instrumented with crashlab.

Run it (from the repo root) against any OpenAI-compatible endpoint or Anthropic:

    OPENAI_API_KEY=sk-...            python -m examples.refund_bot "Refund order A17, it broke"
    OPENAI_BASE_URL=http://localhost:11434/v1 MODEL=qwen2.5:7b \
                                     python -m examples.refund_bot "Refund order A17, it broke"
    PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... \
                                     python -m examples.refund_bot "Refund order A17, it broke"

Faults (env var = 1, or the matching --fault-... flag):

- ``FAULT_HTML_502``: get_order returns an HTML 502 page instead of JSON.
- ``FAULT_SCHEMA_DRIFT``: get_order returns valid JSON, but ``total`` is renamed
  ``amount_total`` (an upstream API changed its schema).
- ``FAULT_POLICY_DROP``: the refund-window rule (policy line 2) is missing from the
  system prompt. Try it with an out-of-window order: "Please refund order C03".

The active faults are part of the recorded run input, so ``crashlab replay`` reproduces a
faulty run without the env vars. ``@crashlab.check`` oracles at the bottom grade every run
against the real order database: pass / fail instead of "unknown".
"""

from __future__ import annotations

import argparse
import contextvars
import json
import os
import re
from typing import Any

import crashlab

POLICY_LINES = [
    "Always look up the order with get_order before deciding anything.",
    'Only refund orders whose status is "delivered" and that were delivered at most 30 days ago.',
    "Never refund more than the order total.",
    "Issue at most one refund per order.",
]
REFUND_WINDOW_DAYS = 30
POLICY_DROP_LINE = 1  # FAULT_POLICY_DROP removes the delivery / 30-day window rule


def system_prompt(faults: frozenset[str] = frozenset()) -> str:
    lines = [ln for i, ln in enumerate(POLICY_LINES)
             if not ("policy_drop" in faults and i == POLICY_DROP_LINE)]  # fmt: skip
    policy = "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))
    return (
        "You are the refund assistant for ShopCo customer support.\n\n"
        f"Refund policy:\n{policy}\n\n"
        "Use the tools to act, then reply to the customer in one or two short sentences."
    )


SYSTEM_PROMPT = system_prompt()

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

# --- faults -------------------------------------------------------------------------

FAULTS = ("html_502", "schema_drift", "policy_drop")
_active_faults: contextvars.ContextVar[frozenset[str] | None] = contextvars.ContextVar(
    "refund_bot_faults", default=None
)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def faults_from_env() -> list[str]:
    return [f for f in FAULTS if _env_on(f"FAULT_{f.upper()}")]


def active_faults() -> frozenset[str]:
    faults = _active_faults.get()
    return faults if faults is not None else frozenset(faults_from_env())


def fault_html_502() -> bool:
    return "html_502" in active_faults()


def reset_db() -> None:
    REFUNDS.clear()


# --- tools --------------------------------------------------------------------------


@crashlab.tool()
def get_order(order_id: str) -> dict[str, Any] | str:
    """Look up an order by id. Returns the order as JSON."""
    faults = active_faults()
    if "html_502" in faults:
        return HTML_502
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise KeyError(f"order {order_id!r} not found")
    if "schema_drift" in faults:  # same data, but the upstream renamed a field
        return {("amount_total" if k == "total" else k): v for k, v in order.items()}
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


ANTHROPIC_TOOLS = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOLS
]


# --- client -------------------------------------------------------------------------

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "litellm": "ollama_chat/qwen2.5:3b",  # local, no API key; or groq/..., gemini/...
}


def default_provider() -> str:
    provider = os.environ.get("PROVIDER", "openai").strip().lower()
    if provider not in DEFAULT_MODELS:
        raise SystemExit(f"PROVIDER must be one of {sorted(DEFAULT_MODELS)}, not {provider!r}")
    return provider


class LiteLLMClient:
    """``client.chat.completions.create(...)`` on top of ``litellm.completion``, so the
    OpenAI-style loop below also runs on any LiteLLM model (``ollama_chat/qwen2.5:3b``,
    ``ollama/llama3.2:3b``, ``groq/...``). crashlab records it via ``crashlab.wrap(litellm)``."""

    def __init__(self) -> None:
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no fetch at import
        import litellm

        self.litellm = crashlab.wrap(litellm)

        class _Completions:
            @staticmethod
            def create(**kwargs: Any) -> Any:
                return self.litellm.completion(**kwargs)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def make_client(provider: str | None = None) -> Any:
    """Client wrapped by crashlab. openai: OPENAI_BASE_URL / OPENAI_API_KEY (any
    OpenAI-compatible server, e.g. Ollama at http://localhost:11434/v1); anthropic:
    ANTHROPIC_API_KEY (ANTHROPIC_BASE_URL optional); litellm: whatever the model's
    provider needs (Ollama: nothing; OLLAMA_API_BASE if not on localhost)."""
    provider = provider or default_provider()
    if provider == "anthropic":
        from anthropic import Anthropic

        return crashlab.wrap(Anthropic())
    if provider == "litellm":
        return LiteLLMClient()
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL") or None
    api_key = os.environ.get("OPENAI_API_KEY") or ("ollama" if base_url else None)
    return crashlab.wrap(OpenAI(base_url=base_url, api_key=api_key))


def default_model(provider: str | None = None) -> str:
    return os.environ.get("MODEL") or DEFAULT_MODELS[provider or default_provider()]


def client_provider(client: Any) -> str:
    """'anthropic' for Anthropic-style clients (messages.create), else 'openai'."""
    inner = getattr(client, "unwrapped", client)
    if isinstance(inner, LiteLLMClient):
        return "litellm"
    if type(inner).__module__.split(".")[0] == "anthropic":
        return "anthropic"
    if getattr(inner, "chat", None) is None and getattr(inner, "messages", None) is not None:
        return "anthropic"
    return "openai"


# --- the agent loops (plain SDK tool calling; nothing crashlab-specific inside) --------


def _tool_call_dict(tc: Any) -> dict[str, Any]:
    # model_dump keeps provider extras, e.g. Gemini's extra_content.google.thought_signature,
    # which Gemini 3 requires to be sent back with the next request.
    if hasattr(tc, "model_dump"):
        return tc.model_dump(exclude_none=True)
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


def _assistant_message(msg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [_tool_call_dict(tc) for tc in msg.tool_calls]
    return out


def _run_tool(name: str, arguments: str | dict[str, Any]) -> str:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return f"ERROR: unknown tool {name!r}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments are not valid JSON: {exc}"
    try:
        result = fn(**arguments)
    except Exception as exc:  # the model sees the error and can recover
        return f"ERROR: {type(exc).__name__}: {exc}"
    return result if isinstance(result, str) else json.dumps(result)


def _openai_loop(client: Any, model: str, user_msg: str, max_turns: int, system: str) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
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


def _anthropic_loop(client: Any, model: str, user_msg: str, max_turns: int, system: str) -> str:
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]
    for _ in range(max_turns):
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=ANTHROPIC_TOOLS,
        )
        blocks: list[dict[str, Any]] = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        messages.append({"role": "assistant", "content": blocks})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            return "".join(b.text for b in resp.content if b.type == "text")
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": _run_tool(b.name, b.input),
                    }
                    for b in tool_uses
                ],
            }
        )
    return "Sorry, I could not finish this request."


@crashlab.agent(name="refund-bot", ignore_args=("client",))
def run_agent(
    user_msg: str,
    *,
    client: Any = None,
    model: str | None = None,
    provider: str | None = None,
    max_turns: int = 8,
    faults: list[str] | None = None,
) -> str:
    """Answer one customer message. ``faults`` (default: from FAULT_* env vars) selects
    injected failures; pass it explicitly so it is recorded in the run input."""
    unknown = set(faults or ()) - set(FAULTS)
    if unknown:
        raise ValueError(f"unknown faults {sorted(unknown)}; choose from {FAULTS}")
    active = frozenset(faults) if faults is not None else frozenset(faults_from_env())
    if client is None:
        provider = provider or default_provider()
        client = make_client(provider)
    provider = provider or client_provider(client)
    model = model or default_model(provider)
    loop = _anthropic_loop if provider == "anthropic" else _openai_loop
    token = _active_faults.set(active)
    try:
        return loop(client, model, user_msg, max_turns, system_prompt(active))
    finally:
        _active_faults.reset(token)


# --- outcome oracles (graded against the real order DB, not what the tools returned) ----

STEP_BUDGET = 8  # the happy path takes 5 steps: chat, get_order, chat, issue_refund, chat


def _blocked(step: crashlab.Step) -> bool:
    """A refund the agent decided to make, but crashlab blocked (a fork never runs
    side-effect tools live). It still counts: the decision is what we grade."""
    return step.error is not None and step.error.type.endswith("SideEffectBlockedError")


def _refunds(run: crashlab.Run) -> list[crashlab.Step]:
    """Refunds issued, plus refund attempts blocked by crashlab during a fork."""
    return [s for s in run.tool_calls("issue_refund") if s.error is None or _blocked(s)]


def _note(step: crashlab.Step) -> str:
    return " (attempt blocked by crashlab during the fork)" if _blocked(step) else ""


def _order(order_id: Any) -> dict[str, Any] | None:
    return ORDERS.get(str(order_id or "").strip().upper())


@crashlab.check("refund-bot")
def refund_amount_equals_order_total(run: crashlab.Run) -> None:
    """Every refund issued is for exactly the order total."""
    for step in _refunds(run):
        order = _order(step.args.get("order_id"))
        assert order is not None, f"step {step.idx}: refund for unknown order {step.args}"
        amount = float(step.args.get("amount") or 0)
        assert abs(amount - order["total"]) < 0.005, (
            f"step {step.idx}: refunded {amount:g} for {order['order_id']}, "
            f"but the order total is {order['total']:g}{_note(step)}"
        )


@crashlab.check("refund-bot")
def refund_within_policy(run: crashlab.Run) -> None:
    """Refunds only after a lookup, only for delivered orders inside the refund window,
    at most one per order and never above the total."""
    looked_up: set[str] = set()
    refunded: set[str] = set()
    for step in run.steps:
        if step.operation != "execute_tool":
            continue
        oid = str(step.args.get("order_id") or "").strip().upper()
        if step.name == "get_order":
            looked_up.add(oid)
        if step.name != "issue_refund" or (step.error is not None and not _blocked(step)):
            continue
        note = _note(step)
        order = _order(oid)
        assert order is not None, f"step {step.idx}: refund for unknown order {oid!r}{note}"
        assert oid in looked_up, (
            f"step {step.idx}: refunded {oid} without looking it up first{note}"
        )
        assert order["status"] == "delivered", (
            f"step {step.idx}: refunded {oid}, but its status is {order['status']!r}{note}"
        )
        days = order["days_since_delivery"]
        assert days is not None and days <= REFUND_WINDOW_DAYS, (
            f"step {step.idx}: refunded {oid}, delivered {days} days ago "
            f"(window: {REFUND_WINDOW_DAYS} days){note}"
        )
        assert float(step.args.get("amount") or 0) <= order["total"] + 0.005, (
            f"step {step.idx}: refunded more than the order total for {oid}{note}"
        )
        assert oid not in refunded, f"step {step.idx}: second refund for {oid}{note}"
        refunded.add(oid)


# "has been issued", "I've processed", "was refunded" ...: the reply says the refund happened
_CLAIM = re.compile(
    r"\b(?:has|have|had)\s+(?:now\s+)?(?:been\s+)?(?:successfully\s+)?"
    r"(?:issued|processed|refunded|initiated|completed|credited|approved)\b"
    r"|\b(?:I|we)(?:'ve|\u2019ve)?\s+(?:just\s+|now\s+)?(?:issued|processed|refunded|credited)\b"
    r"|\b(?:was|were|is)\s+(?:now\s+)?(?:successfully\s+)?(?:issued|processed|refunded|credited)\b",
    re.IGNORECASE,
)
_NEGATED = re.compile(
    r"\b(?:not|never|no|unable|cannot|can't|couldn't|won't|haven't|hasn't|wasn't)\b"
    r"|n[\u2019']t\b|^\s*(?:if|once|when|after|until)\b",
    re.IGNORECASE,
)


def _reply_texts(step: crashlab.Step) -> list[str]:
    msgs = (step.response or {}).get("messages") or []
    parts = [p for m in msgs for p in m.get("parts", []) if p.get("type") == "text"]
    return [str(p.get("content") or p.get("text") or "") for p in parts]


def _refund_claims(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [
        s.strip()
        for s in sentences
        if "refund" in s.lower() and _CLAIM.search(s) and not _NEGATED.search(s)
    ]


@crashlab.check("refund-bot")
def no_unbacked_refund_claim(run: crashlab.Run) -> None:
    """The final reply never says a refund was issued unless issue_refund actually ran
    (small local models like to promise actions they never took)."""
    if _refunds(run):
        return
    chats = run.chat_steps()
    idx = chats[-1].idx if chats else len(run.steps) - 1
    final = " ".join(_reply_texts(chats[-1])).strip() if chats else ""
    if not final and isinstance(run.output, str):
        final = run.output
    claims = _refund_claims(final)
    assert not claims, (
        f"step {idx}: the reply says {claims[0]!r}, but issue_refund was never called"
    )


@crashlab.check("refund-bot")
def within_step_budget(run: crashlab.Run) -> None:
    """The run stays within the step budget (no lookup loops)."""
    steps = len(run.steps) or run.usage.steps
    assert steps <= STEP_BUDGET, f"{steps} steps, budget is {STEP_BUDGET}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="refund-bot crashlab demo")
    parser.add_argument(
        "message", nargs="?", default="Hi, please refund order A17, it arrived broken."
    )
    for fault in FAULTS:
        flag = "--fault-" + fault.replace("_", "-")
        parser.add_argument(
            flag, action="store_true", help=f"inject {fault} (or FAULT_{fault.upper()}=1)"
        )
    parser.add_argument("--model", default=None, help="model name (default: $MODEL)")
    parser.add_argument(
        "--provider", choices=sorted(DEFAULT_MODELS), default=None, help="default: $PROVIDER"
    )
    ns = parser.parse_args(argv)
    faults = [f for f in FAULTS if getattr(ns, "fault_" + f) or f in faults_from_env()]
    # resolve everything that shapes the LLM requests here, so it lands in the recorded
    # input and `crashlab replay` does not depend on env vars
    provider = ns.provider or default_provider()
    model = ns.model or default_model(provider)
    answer = run_agent(ns.message, model=model, provider=provider, faults=faults)
    print(answer)
    if REFUNDS:
        print(f"[refunds issued: {REFUNDS}]")
    run = crashlab.load("last")
    print(f"[outcome: {run.outcome}" + (f" - {run.outcome_reason}]" if run.outcome_reason else "]"))
    print("Recorded. Inspect with: crashlab show last   Replay offline: crashlab replay last")


if __name__ == "__main__":
    main()
