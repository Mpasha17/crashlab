"""Cost estimation: built-in table, LiteLLM's map, custom prices, unknown -> None."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import crashlab
from crashlab import pricing
from tests.fakes import FakeOpenAI, refund_script


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("CRASHLAB_NO_PYPROJECT", "1")
    monkeypatch.setattr(pricing, "_litellm_map", {})  # no LiteLLM map unless a test sets one
    pricing.clear_prices()
    yield
    pricing.clear_prices()


def _chat(model: str, host: str | None = "api.openai.com", when: str = "2026-10-18", **usage):
    attrs = {"gen_ai.request.model": model, "gen_ai.response.model": model}
    if host:
        attrs["server.address"] = host
    attrs.update({k: v for k, v in usage.items() if not k.startswith("_")})
    return crashlab.Step(
        run_id="r", idx=0, span_id="s", operation="chat", name=model, request_hash="h",
        attributes=attrs, response={"messages": []},
        started_at=datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
    )  # fmt: skip


def _usage(inp: int, out: int, cached: int = 0, total: int | None = None, **extra) -> dict:
    u = {
        "gen_ai.usage.input_tokens": inp,
        "gen_ai.usage.output_tokens": out,
        "crashlab.usage.total_tokens": total if total is not None else inp + out,
    }
    if cached:
        u["gen_ai.usage.cache_read.input_tokens"] = cached
    u.update(extra)
    return u


def test_openai_builtin_with_cached_input():
    # gpt-4o-mini: 0.15 in / 0.60 out / 0.075 cached per 1M; input_tokens include cached
    step = _chat("gpt-4o-mini", **_usage(1000, 100, cached=400))
    assert pricing.step_cost(step) == pytest.approx((600 * 0.15 + 400 * 0.075 + 100 * 0.60) / 1e6)


def test_dated_snapshot_names_resolve():
    a = pricing.step_cost(_chat("gpt-4o-mini-2024-07-18", **_usage(1000, 0)))
    assert a == pytest.approx(1000 * 0.15 / 1e6)


def test_anthropic_cache_counted_separately():
    step = _chat(
        "claude-haiku-4-5", host="api.anthropic.com",
        **_usage(100, 10, cached=1000, total=100 + 10 + 1000 + 200,
                 **{"gen_ai.usage.cache_creation.input_tokens": 200}),
    )  # fmt: skip
    expected = (100 * 1.00 + 1000 * 0.10 + 200 * 1.25 + 10 * 5.00) / 1e6
    assert pricing.step_cost(step) == pytest.approx(expected)


def test_groq_and_gemini_by_host_and_date():
    groq = _chat("openai/gpt-oss-120b", host="api.groq.com", **_usage(278, 52, cached=256))
    assert pricing.step_cost(groq) == pytest.approx((22 * 0.15 + 256 * 0.075 + 52 * 0.60) / 1e6)
    host = "generativelanguage.googleapis.com"
    # thinking tokens are in total but not in output: billed as output
    before = _chat("gemini-3.8-flash", host=host, when="2026-12-31", **_usage(100, 10, total=150))
    after = _chat("gemini-3.8-flash", host=host, when="2027-01-01", **_usage(100, 10, total=150))
    assert pricing.step_cost(before) == pytest.approx((100 * 0.75 + 50 * 3.75) / 1e6)
    assert pricing.step_cost(after) == pytest.approx((100 * 1.50 + 50 * 7.50) / 1e6)


@pytest.mark.parametrize("host", ["localhost:11434", "127.0.0.1", "ollama.local"])
def test_local_servers_are_free(host):
    assert pricing.step_cost(_chat("qwen2.5:3b", host=host, **_usage(100, 10))) == 0.0


def test_litellm_prefixes():
    lite = {"gen_ai.provider.name": "litellm"}
    assert pricing.step_cost(_chat("ollama_chat/qwen2.5:3b", host=None, **_usage(5, 5))) == 0.0
    step = _chat("groq/openai/gpt-oss-120b", host=None, **_usage(1000, 0), **lite)
    assert pricing.step_cost(step) == pytest.approx(1000 * 0.15 / 1e6)


def test_unknown_model_is_none_never_guessed():
    assert pricing.step_cost(_chat("gpt-99-turbo", **_usage(100, 10))) is None
    assert pricing.step_cost(_chat("gpt-4o-mini", host="api.together.xyz", **_usage(1, 1))) is None


def test_no_usage_reported_is_none_failed_call_is_zero():
    assert pricing.step_cost(_chat("gpt-4o-mini")) is None
    failed = _chat("gpt-4o-mini").model_copy(
        update={"error": crashlab.models.ErrorInfo(type="APIError", message="boom")}
    )
    assert pricing.step_cost(failed) == 0.0


def test_litellm_cost_map_used_when_builtin_misses(monkeypatch):
    monkeypatch.setattr(
        pricing, "_litellm_map",
        {"groq/llama-9-70b": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6,
                              "litellm_provider": "groq"},
         "llama-9-70b": {"input_cost_per_token": 9e-6, "output_cost_per_token": 9e-6,
                         "litellm_provider": "together_ai"}},
    )  # fmt: skip
    step = _chat("llama-9-70b", host="api.groq.com", **_usage(1000, 1000))
    assert pricing.step_cost(step) == pytest.approx((1000 * 1.0 + 1000 * 2.0) / 1e6)
    assert pricing.price_for("llama-9-70b", "groq").source == "litellm"


def test_set_price_and_pyproject_prices(tmp_path, monkeypatch):
    pricing.set_price("my-model", input=1.0, output=2.0)
    assert pricing.step_cost(_chat("my-model", host="api.x.com", **_usage(1e6, 0))) == 1.0
    pricing.clear_prices()
    monkeypatch.delenv("CRASHLAB_NO_PYPROJECT")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.crashlab.prices."house-model"]\ninput = 3.0\noutput = 4.0\n'
    )
    monkeypatch.chdir(tmp_path)
    assert pricing.step_cost(_chat("house-model", host="h", **_usage(0, 1e6))) == 4.0


def test_run_cost_recorded_and_unknown_spreads():
    run_id = None
    refund_bot = __import__("examples.refund_bot.agent", fromlist=["run_agent"])
    pricing.set_price("m1", input=1.0, output=2.0)
    refund_bot.run_agent("Refund A17", client=crashlab.wrap(FakeOpenAI(refund_script())),
                         model="m1")  # fmt: skip
    run = crashlab.load("last")
    run_id = run.run_id
    # fake.local is a local host: the recorded cost is $0 even with a price set
    assert run.usage.cost_usd == 0.0
    assert all(s.attributes.get("crashlab.usage.cost_usd") == 0.0
               for s in run.steps if s.operation == "chat")  # fmt: skip
    old = run.model_copy(deep=True)
    old.usage.cost_usd = None
    for s in old.steps:
        s.attributes.pop("crashlab.usage.cost_usd", None)
        s.attributes["server.address"] = "api.example.com"
    assert pricing.run_cost(old) == pytest.approx((403 * 1.0 + 40 * 2.0) / 1e6)
    pricing.clear_prices()
    assert pricing.run_cost(old) is None  # one unknown call makes the run total unknown
    assert run_id
