"""Hand-built runs for search / cluster / show tests: written through the real Store, so
they read back exactly like recordings (index + cassette), with fixed ids and times."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from crashlab.models import CheckResult, Diagnosis, ErrorInfo, Finding, Run, Step, Usage
from crashlab.store import Store

T0 = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)
_ALPH = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def run_id(n: int) -> str:
    """Sortable 26-char ids: a larger n is newer."""
    s = ""
    v = n
    for _ in range(6):
        v, r = divmod(v, 32)
        s = _ALPH[r] + s
    return "01FIXTURE" + "0" * 11 + s


def chat(rid: str, idx: int, user: str | None, reply: str | None = None, *,
         tool_call: tuple[str, dict] | None = None, model: str = "gpt-4o-mini",
         history: list[dict] | None = None, tokens: tuple[int, int] = (100, 10),
         cost: float | None = 0.0001, duration_ms: float = 120.0,
         at: datetime | None = None, error: ErrorInfo | None = None) -> Step:  # fmt: skip
    msgs = list(history or [])
    if user is not None:
        msgs.append({"role": "user", "parts": [{"type": "text", "content": user}]})
    parts: list[dict[str, Any]] = []
    if reply is not None:
        parts.append({"type": "text", "content": reply})
    if tool_call is not None:
        parts.append({"type": "tool_call", "id": f"call_{idx}", "name": tool_call[0],
                      "arguments": tool_call[1]})  # fmt: skip
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": model,
        "gen_ai.response.model": model,
        "gen_ai.usage.input_tokens": tokens[0],
        "gen_ai.usage.output_tokens": tokens[1],
        "crashlab.usage.total_tokens": sum(tokens),
    }
    if cost is not None:
        attrs["crashlab.usage.cost_usd"] = cost
    return Step(
        run_id=rid, idx=idx, span_id=f"{idx:016x}", operation="chat", name=model,
        attributes=attrs,
        request={"system": "You are a refund assistant. Never refund twice.", "messages": msgs},
        response=None if error else {"messages": [{"role": "assistant", "parts": parts}]},
        error=error, request_hash=f"h{idx}", duration_ms=duration_ms,
        started_at=at or T0,
    )  # fmt: skip


def tool(rid: str, idx: int, name: str, args: dict, result: Any = None, *,
         error: ErrorInfo | None = None, duration_ms: float = 5.0,
         at: datetime | None = None) -> Step:  # fmt: skip
    return Step(
        run_id=rid, idx=idx, span_id=f"{idx:016x}", operation="execute_tool", name=name,
        attributes={"gen_ai.tool.name": name}, request={"args": args},
        response=None if error else {"result": result}, error=error,
        request_hash=f"t{idx}", duration_ms=duration_ms, started_at=at or T0,
    )  # fmt: skip


def save(store: Store, run: Run, steps: list[Step]) -> Run:
    run.usage = Usage(
        steps=len(steps),
        llm_calls=sum(s.operation == "chat" for s in steps),
        tool_calls=sum(s.operation == "execute_tool" for s in steps),
        errors=sum(s.error is not None for s in steps),
    )
    store.begin_run(run)
    for s in steps:
        store.add_step(s)
    store.end_run(run)
    return run


def make_run(store: Store, n: int, name: str = "refund-bot", *, status: str = "ok",
             outcome: str = "pass", error: ErrorInfo | None = None,
             checks: list[CheckResult] | None = None, findings: list[Finding] | None = None,
             diagnosis_label: str | None = None, diagnosis_step: int | None = None,
             steps: list[Step] | None = None, hours_ago: float | None = None,
             user: str = "Refund my order A17") -> Run:  # fmt: skip
    rid = run_id(n)
    started = T0 + timedelta(minutes=n)
    if hours_ago is not None:
        started = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    run = Run(
        run_id=rid, name=name, status=status, outcome=outcome, error=error,  # type: ignore[arg-type]
        checks=checks or [], findings=findings or [], started_at=started,
        ended_at=started + timedelta(seconds=2), input={"user_msg": user},
    )  # fmt: skip
    if diagnosis_label:
        run.diagnosis = Diagnosis(
            run_id=rid,
            label=diagnosis_label,
            failing_step=diagnosis_step,  # type: ignore[arg-type]
            root_cause=f"root cause for {diagnosis_label.lower()}",
        )
    if steps is None:
        steps = [
            chat(rid, 0, user, tool_call=("get_order", {"order_id": "A17"}), at=started),
            tool(
                rid,
                1,
                "get_order",
                {"order_id": "A17"},
                {"status": "delivered", "total": 49},
                at=started + timedelta(milliseconds=130),
            ),  # fmt: skip
            chat(
                rid,
                2,
                None,
                "Done, refund issued.",
                history=[{"role": "user", "parts": [{"type": "text", "content": user}]}],
                at=started + timedelta(milliseconds=140),
            ),  # fmt: skip
        ]
    else:
        steps = [s.model_copy(update={"run_id": rid}) for s in steps]
    return save(store, run, steps)
