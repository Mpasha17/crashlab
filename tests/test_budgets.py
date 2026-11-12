"""Budget oracles (crashlab.budgets) and the per-call / missing-usage budget limits of
the built-in budget detector. Offline, from fixture steps."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

import crashlab
from crashlab.budgets import summarize_usage, within_budget
from crashlab.checks import evaluate, resolve_config
from crashlab.models import ErrorInfo, Run, Step, Usage

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _no_pyproject(monkeypatch):
    monkeypatch.setenv("CRASHLAB_NO_PYPROJECT", "1")


def chat(idx: int, inp: int | None = 100, out: int | None = 20, *, model: str = "gpt-4o-mini",
         ms: float = 500.0, source: str = "live", cost: float | None = None,
         error: bool = False) -> Step:  # fmt: skip
    attrs: dict[str, Any] = {"gen_ai.request.model": model, "gen_ai.provider.name": "openai"}
    if inp is not None:
        attrs["gen_ai.usage.input_tokens"] = inp
    if out is not None:
        attrs["gen_ai.usage.output_tokens"] = out
    if cost is not None:
        attrs["crashlab.usage.cost_usd"] = cost
    return Step(
        run_id="R", idx=idx, span_id=f"s{idx}", operation="chat", name=model,
        attributes=attrs, request={"messages": []}, response={"messages": []},
        request_hash=f"h{idx}", started_at=T0 + timedelta(seconds=idx), duration_ms=ms,
        source=source, error=ErrorInfo(type="APIError", message="boom") if error else None,
    )  # fmt: skip


def tool(idx: int, ms: float = 10.0, source: str = "live") -> Step:
    return Step(
        run_id="R", idx=idx, span_id=f"s{idx}", operation="execute_tool", name="lookup",
        request={"args": {}}, response={"result": {"ok": 1}}, request_hash=f"t{idx}",
        started_at=T0 + timedelta(seconds=idx), duration_ms=ms, source=source,
    )  # fmt: skip


def run_of(*steps: Step, wall_ms: float | None = 3000.0, **kw: Any) -> Run:
    end = T0 + timedelta(seconds=3)
    return Run(run_id="R", name="bot", steps=list(steps), started_at=T0, status="ok",
               ended_at=end, usage=Usage(wall_time_ms=wall_ms), **kw)  # fmt: skip


def test_all_limits_hold_returns_none():
    run = run_of(chat(0, cost=0.001), tool(1), chat(2, cost=0.002))
    assert within_budget(run, max_cost_usd=0.01, max_tokens=1000, max_seconds=10,
                         max_call_seconds=1, max_steps=3, max_llm_calls=2,
                         max_tool_calls=1, max_call_input_tokens=100) is None  # fmt: skip
    u = summarize_usage(run)
    assert (u.total_tokens, u.input_tokens, u.output_tokens) == (240, 200, 40)
    assert u.cost_usd == pytest.approx(0.003) and u.wall_seconds == 3.0


def test_each_limit_exceeded_is_named():
    run = run_of(chat(0, 100, cost=0.004), tool(1, ms=2500), chat(2, 900, cost=0.004))
    msg = within_budget(run, max_cost_usd=0.005, max_tokens=500, max_input_tokens=800,
                        max_output_tokens=10, max_call_input_tokens=500, max_seconds=2,
                        max_call_seconds=2, max_steps=2, max_llm_calls=1,
                        max_tool_calls=0)  # fmt: skip
    assert msg is not None and msg.startswith("budget: ")
    for part in ("3 > max_steps 2", "2 > max_llm_calls 1", "1 > max_tool_calls 0",
                 "1,040 tokens > max_tokens 500", "1,000 input tokens > max_input_tokens 800",
                 "40 output tokens > max_output_tokens 10",
                 "900 input tokens at step 2, grew from 100 > max_call_input_tokens 500",
                 "$0.0080 > max_cost_usd $0.0050", "3.0s > max_seconds 2.0s",
                 "2.50s at step 1 > max_call_seconds 2.00s"):  # fmt: skip
        assert part in msg, part


def test_cost_from_price_table_when_not_stored():
    # gpt-4o-mini is in the built-in table; 1M input tokens cost its input price
    run = run_of(chat(0, inp=1_000_000, out=0))
    price = crashlab.pricing.price_for("gpt-4o-mini", "openai", T0)
    assert price is not None
    assert summarize_usage(run).cost_usd == pytest.approx(price.input)
    assert within_budget(run, max_cost_usd=price.input * 2) is None
    assert "max_cost_usd" in (within_budget(run, max_cost_usd=price.input / 2) or "")


def test_missing_token_usage_fails_by_default_and_says_so():
    run = run_of(chat(0), chat(1, inp=None, out=None))
    msg = within_budget(run, max_tokens=10_000)
    assert msg is not None and "cannot check: token usage not recorded for chat step 1" in msg
    assert within_budget(run, max_tokens=10_000, unknown="pass") is None
    # still a real failure when the known part already exceeds the limit
    assert "120 tokens > max_tokens 50" in (within_budget(run, max_tokens=50, unknown="pass")
                                           or "")  # fmt: skip
    # limits that do not need token counts are unaffected
    assert within_budget(run, max_llm_calls=5) is None


def test_failed_request_without_usage_is_not_missing():
    run = run_of(chat(0), chat(1, inp=None, out=None, error=True))
    assert summarize_usage(run).tokens_missing == []
    assert within_budget(run, max_tokens=10_000, max_cost_usd=1.0) is None


def test_unpriced_model_cost_is_unknown_not_zero():
    run = run_of(chat(0, model="mystery-model-xyz"))
    u = summarize_usage(run)
    assert u.cost_usd is None and u.unpriced_models == ["mystery-model-xyz"]
    msg = within_budget(run, max_cost_usd=1.0)
    assert msg and "cost unknown for chat step 0" in msg and "mystery-model-xyz" in msg
    assert within_budget(run, max_cost_usd=1.0, unknown="pass") is None


def test_time_limits_on_replays_and_forks():
    for kw in ({"replay_of": "X"}, {"forked_from": "X", "fork_step": 1}):
        run = run_of(chat(0, source="cassette"), chat(1), **kw)
        # default: time is judged on the live recording, not on a replay / fork
        assert within_budget(run, max_seconds=0.001) is None, kw
        msg = within_budget(run, max_seconds=100, replay_time="fail")
        assert msg and "wall time not measurable" in msg, kw
    # a served step's duration is not a live latency; live steps after a fork still count
    run = run_of(chat(0, ms=9000, source="cassette"), chat(1, ms=100), forked_from="X")
    assert within_budget(run, max_call_seconds=1) is None
    run = run_of(chat(0, ms=100, source="cassette"), chat(1, ms=3000), forked_from="X")
    assert "3.00s at step 1 > max_call_seconds" in (within_budget(run, max_call_seconds=1)
                                                    or "")  # fmt: skip
    run = run_of(chat(0, ms=9000, source="cassette"), replay_of="X")
    assert within_budget(run, max_call_seconds=1) is None
    msg = within_budget(run, max_call_seconds=1, replay_time="fail")
    assert "no live LLM or tool call" in (msg or "")
    # tokens and cost of served steps are still the recorded usage
    run = run_of(chat(0, inp=900, source="cassette"), replay_of="X")
    assert "920 tokens > max_tokens 100" in (within_budget(run, max_tokens=100) or "")


CLIENT: list[Any] = []


@crashlab.agent(name="budget-replay")
def _replay_agent() -> str:
    r = CLIENT[0].chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    return r.choices[0].message.content or ""


@pytest.mark.requires_providers("openai")
def test_replay_and_exported_style_check_agree_with_the_recording(tmp_path):
    """A recorded run that passed its budget check also passes it on strict replay (what
    an exported regression test asserts), time limits included."""
    from crashlab.verify.oracles import unregister
    from tests.fakes import FakeOpenAI, text_response

    @crashlab.check("budget-replay")
    def in_budget(run):
        return within_budget(run, max_tokens=1000, max_seconds=60, max_call_seconds=30)

    try:
        CLIENT[:] = [crashlab.wrap(FakeOpenAI([text_response(1, "hi")]))]
        _replay_agent()
        rec = crashlab.load("last")
        assert [c.status for c in rec.checks] == ["pass"]
        replayed = crashlab.replay(rec)
        assert replayed.replay_of == rec.run_id
        assert [c.status for c in replayed.checks] == ["pass"], replayed.checks
        assert summarize_usage(replayed).total_tokens == summarize_usage(rec).total_tokens
    finally:
        CLIENT.clear()
        unregister("budget-replay")


def test_running_run_has_no_wall_time():
    run = run_of(chat(0), wall_ms=None)
    run.ended_at = None
    assert "the run has not ended" in (within_budget(run, max_seconds=5) or "")
    run = run_of(chat(0), wall_ms=None)  # ended: falls back to ended_at - started_at
    assert summarize_usage(run).wall_seconds == 3.0


def test_run_without_steps_uses_stored_totals():
    run = Run(run_id="R", name="b", status="ok", usage=Usage(
        llm_calls=2, total_tokens=500, input_tokens=400, output_tokens=100, cost_usd=0.01,
        wall_time_ms=1000))  # fmt: skip
    assert within_budget(run, max_tokens=600, max_cost_usd=0.02, max_seconds=2) is None
    assert "500 tokens > max_tokens 400" in (within_budget(run, max_tokens=400) or "")
    bare = Run(run_id="R", name="b", status="ok", usage=Usage(llm_calls=1, wall_time_ms=5))
    assert "some chat calls" in (within_budget(bare, max_tokens=1) or "")


def test_bad_arguments_raise():
    run = run_of(chat(0))
    with pytest.raises(ValueError):
        within_budget(run, max_tokens=-1)
    with pytest.raises(ValueError):
        within_budget(run, unknown="skip")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        within_budget(run, replay_time="no")  # type: ignore[arg-type]


@pytest.mark.requires_providers("openai")
def test_as_a_registered_check_end_to_end(tmp_path):
    from crashlab.verify.oracles import unregister
    from tests.fakes import FakeOpenAI, text_response

    @crashlab.check("budget-bot")
    def small(run):
        return within_budget(run, max_tokens=25)

    try:
        client = crashlab.wrap(FakeOpenAI([text_response(1, "hi"), text_response(2, "hi")]))
        with crashlab.record("budget-bot", store_dir=tmp_path / "s") as run:
            for _ in range(2):
                client.chat.completions.create(model="m", messages=[{"role": "user",
                                                                     "content": "x"}])  # fmt: skip
        stored = crashlab.Store(tmp_path / "s").load_run(run.run_id)
        (res,) = stored.checks
        assert res.status == "fail" and "tokens > max_tokens 25" in (res.message or "")
        assert stored.outcome == "fail"
    finally:
        unregister("budget-bot")


# --- built-in budget detector: per-call limits and missing usage --------------------


def evaluate_budget(run: Run, checks: dict[str, Any]) -> list[Any]:
    return [f for f in evaluate(run, resolve_config(checks)) if f.kind == "budget"]


def test_detector_per_call_input_tokens():
    run = run_of(chat(0, 100), tool(1), chat(2, 5000), chat(3, 6000))
    (f,) = evaluate_budget(run, {"max_call_input_tokens": 4000})
    assert f.kind == "budget" and f.severity == "critical" and f.steps == [2, 3]
    assert "5,000 input tokens > max_call_input_tokens 4,000" in f.message


def test_detector_per_call_seconds_live_only():
    run = run_of(chat(0, ms=100), tool(1, ms=4000), chat(2, ms=9000, source="cassette"))
    (f,) = evaluate_budget(run, {"budget": {"max_call_seconds": 2}})
    assert f.severity == "critical" and f.steps == [1]
    assert "slow tool lookup: 4.00s > max_call_seconds 2.00s" in f.message


def test_detector_missing_tokens_is_info_lower_bound():
    run = run_of(chat(0), chat(1, inp=None, out=None))
    found = evaluate_budget(run, {"max_tokens": 10_000})
    assert [(f.severity, f.key) for f in found] == [("info", "max_tokens:missing")]
    assert "lower bound" in found[0].message
    # no limit set: nothing to say
    assert evaluate_budget(run, {}) == []


def test_old_stored_check_config_still_loads():
    old = resolve_config({"max_tokens": 5}).model_dump(mode="json")
    for k in ("max_call_input_tokens", "max_call_seconds"):
        old["budget"].pop(k)
    cfg = resolve_config(None, base=json.loads(json.dumps(old)))
    assert cfg.budget.max_tokens == 5 and cfg.budget.max_call_seconds is None
