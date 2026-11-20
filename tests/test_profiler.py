"""Cost, token and latency profiling (``crashlab profile``)."""

from __future__ import annotations

import json
import re
from datetime import timedelta

from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.models import Run, Step, Usage
from crashlab.pricing import price_for
from crashlab.profiler import DEFAULT_COMPARE_MODELS, profile_run
from crashlab.store import Store
from tests.debug_fixtures import T0, chat, save, tool

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _run_with_overlap(store: Store) -> Run:
    """Two LLM calls that share a long unchanged prefix, plus a slow tool."""
    rid = "01PROFILE00000000000000001"
    system = "You are a refund assistant. Never refund twice."
    user = {"role": "user", "parts": [{"type": "text", "content": "Refund my order A17"}]}
    first = chat(
        rid, 0, "Refund my order A17", tool_call=("get_order", {"order_id": "A17"}),
        tokens=(200, 20), cost=0.001, duration_ms=400.0, at=T0,
    )  # fmt: skip
    first.request["system"] = system
    first.request["tools"] = [{"name": "get_order"}]
    mid = tool(
        rid, 1, "get_order", {"order_id": "A17"}, {"status": "delivered", "total": 49},
        duration_ms=250.0, at=T0 + timedelta(milliseconds=400),
    )  # fmt: skip
    second = chat(
        rid, 2, None, "Done, refund issued.",
        history=[user, {"role": "tool", "name": "get_order", "content": "ok"}],
        tokens=(280, 15), cost=0.002, duration_ms=300.0,
        at=T0 + timedelta(milliseconds=650),
    )  # fmt: skip
    # the second prompt starts with the same system + first user message
    second.request["system"] = system
    second.request["messages"] = [
        user,
        {"role": "assistant", "parts": [{"type": "tool_call", "name": "get_order"}]},
        {"role": "tool", "name": "get_order", "content": "ok"},
    ]
    second.request["tools"] = [{"name": "get_order"}]
    other = Step(
        run_id=rid, idx=3, span_id="0000000000000003", operation="custom",
        name="postprocess", request_hash="p", duration_ms=50.0,
        started_at=T0 + timedelta(milliseconds=950),
    )  # fmt: skip
    run = Run(
        run_id=rid,
        name="refund-bot",
        status="ok",
        outcome="pass",
        started_at=T0,
        ended_at=T0 + timedelta(milliseconds=1200),
        usage=Usage(wall_time_ms=1200, input_tokens=480, output_tokens=35, cost_usd=0.003),
    )
    steps = [first, mid, second, other]
    saved = save(store, run, steps)
    # save() persists the cassette but does not attach steps on the in-memory run
    saved.steps = steps
    return saved


def test_profile_breaks_down_time_tokens_cost_and_tools(crashlab_dir):
    run = _run_with_overlap(Store(crashlab_dir))
    # save() overwrites usage; restore the wall-clock total the profiler should prefer
    run.usage.wall_time_ms = 1200
    run.usage.input_tokens = 480
    run.usage.output_tokens = 35
    run.usage.cost_usd = 0.003
    profile = profile_run(run)

    assert profile.run_id == run.run_id
    assert profile.total_duration_ms == 1200
    assert profile.llm_duration_ms == 700  # 400 + 300
    assert profile.tool_duration_ms == 250
    assert profile.other_duration_ms == 250  # 50 custom + 200 leftover wall
    assert profile.llm_pct + profile.tool_pct + profile.other_pct == 100
    assert abs(profile.llm_pct - 700 / 12) < 0.2  # 58.33
    assert profile.input_tokens == 480
    assert profile.output_tokens == 35
    assert profile.cost_usd == 0.003

    by_idx = {s.idx: s for s in profile.steps}
    assert by_idx[0].category == "llm" and by_idx[0].input_tokens == 200
    assert by_idx[1].category == "tool" and by_idx[1].name == "get_order"
    assert by_idx[3].category == "other"
    assert profile.slowest_steps[0].idx == 0  # 400ms chat
    assert profile.costliest_steps[0].idx == 2  # $0.002
    assert profile.tools[0].name == "get_order"
    assert profile.tools[0].calls == 1
    assert profile.tools[0].total_duration_ms == 250


def test_profile_estimates_redundant_prompt_prefix_tokens(crashlab_dir):
    run = _run_with_overlap(Store(crashlab_dir))
    profile = profile_run(run)
    cache = profile.cache
    assert cache.consecutive_pairs == 1
    pair = cache.pairs[0]
    assert pair.from_idx == 0 and pair.to_idx == 2
    assert pair.prefix_chars > 0
    assert cache.estimated_redundant_tokens > 0
    # the shared system prompt + user turn is a large fraction of the second prompt
    assert pair.estimated_redundant_tokens <= 280
    assert cache.estimated_redundant_tokens == pair.estimated_redundant_tokens


def test_profile_what_if_defaults_and_custom_models(crashlab_dir):
    run = _run_with_overlap(Store(crashlab_dir))
    profile = profile_run(run)
    names = [s.model for s in profile.what_if]
    assert list(DEFAULT_COMPARE_MODELS) == names
    # gpt-4o-mini is in the built-in table: 200+280 in, 20+15 out
    mini = next(s for s in profile.what_if if s.model == "gpt-4o-mini")
    price = price_for("gpt-4o-mini", "openai")
    assert price is not None and mini.priced
    expected = (480 * price.input + 35 * price.output) / 1e6
    assert mini.estimated_cost_usd == round(expected, 8)

    custom = profile_run(run, compare_models=["gpt-4o-mini", "definitely-not-a-model"])
    assert [s.model for s in custom.what_if] == ["gpt-4o-mini", "definitely-not-a-model"]
    unknown = custom.what_if[1]
    assert unknown.priced is False
    assert unknown.estimated_cost_usd is None
    assert unknown.note == "no list price"


def test_profile_cli_help_strips_ansi_and_json(crashlab_dir):
    help_out = _plain(runner.invoke(app, ["profile", "--help"]).output)
    assert "Break down a run" in help_out
    assert "--compare-models" in help_out
    assert "\x1b[" not in help_out

    _run_with_overlap(Store(crashlab_dir))
    # wall_time was overwritten by save(); the CLI loads from the store as saved
    res = runner.invoke(app, ["profile", "last", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["run_id"]
    assert "llm_pct" in payload and "cache" in payload and "what_if" in payload
    assert payload["cache"]["consecutive_pairs"] == 1

    res = runner.invoke(app, ["profile", "last", "--compare-models", "gpt-4o-mini,gpt-5"])
    assert res.exit_code == 0, res.output
    out = _plain(res.output)
    assert "gpt-4o-mini" in out and "gpt-5" in out
    assert "cache opportunity" in out
    assert "LLM" in out and "tools" in out

    missing = runner.invoke(app, ["profile", "no-such-run"])
    assert missing.exit_code == 2
    empty = runner.invoke(app, ["profile", "last", "--compare-models", " , "])
    assert empty.exit_code == 2
