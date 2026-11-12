"""HTML report: per-step timing / cost waterfall, from recorded data only."""

from __future__ import annotations

import re
from datetime import timedelta

from crashlab.models import ErrorInfo, Run
from crashlab.report_html import render_html
from crashlab.store import Store
from tests.debug_fixtures import T0, chat, make_run, tool


def _section(html: str) -> str:
    m = re.search(r"<h2>Step timing and cost</h2>(.*?)<h2>Step timeline</h2>", html, re.S)
    assert m, "timing section missing"
    return m.group(1)


def _run(crashlab_dir, steps, **kw):
    s = Store()
    return s.load_run(make_run(s, 1, steps=steps, **kw).run_id)


def test_waterfall_offsets_durations_tokens_and_cost(crashlab_dir):
    ms = timedelta(milliseconds=1)
    run = _run(crashlab_dir, [
        chat("x", 0, "hi", tool_call=("lookup", {}), at=T0, duration_ms=800, cost=0.002,
             tokens=(1000, 50)),
        tool("x", 1, "lookup", {}, {"ok": 1}, at=T0 + 800 * ms, duration_ms=200),
        chat("x", 2, None, "done", at=T0 + 1000 * ms, duration_ms=1000, cost=0.001,
             tokens=(1200, 20)),
    ])  # fmt: skip
    sec = _section(render_html(run))
    rows = re.findall(r"<tr><td>(\d+)</td>.*?</tr>", sec, re.S)
    assert rows == ["0", "1", "2"]
    assert "+800 ms" in sec and "+1,000 ms" in sec and "1,000.0 ms" in sec
    assert "1000 / 50" in sec and "$0.0020" in sec
    # bars are positioned on the 2,000 ms span: step 1 starts at 40% and is 10% wide
    assert 'class="tool" style="left:40.00%;width:10.00%"' in sec
    assert 'style="left:50.00%;width:50.00%"' in sec
    assert "span 2,000 ms" in sec and "LLM calls 1,800 ms" in sec and "tools 200 ms" in sec
    assert "slowest: step 2 (chat gpt-4o-mini, 1,000 ms)" in sec
    assert "2,200 / 70 tokens" in sec and "est. cost $0.0030" in sec
    assert "no known price" not in sec and "<script" not in render_html(run)


def test_unknown_price_is_marked_not_invented(crashlab_dir):
    run = _run(crashlab_dir, [
        chat("x", 0, "hi", "yo", model="my-private-model", cost=None, duration_ms=50),
    ])  # fmt: skip
    sec = _section(render_html(run))
    assert '<td class="num">?</td>' in sec
    assert "est. cost $0.0000 + 1 call(s) with no known price" in sec
    assert "no price in crashlab" in sec


def test_error_steps_and_replayed_sources_are_labelled(crashlab_dir):
    run = _run(crashlab_dir, [
        tool("x", 0, "get", {}, error=ErrorInfo(type="E", message="m"), duration_ms=10),
    ])  # fmt: skip
    sec = _section(render_html(run))
    assert 'class="err"' in sec and "<td>+0 ms</td>" not in sec
    run.steps[0] = run.steps[0].model_copy(update={"source": "cassette"})
    assert "not of the original call" in _section(render_html(run))


def test_no_timing_data_says_so(crashlab_dir):
    run = _run(crashlab_dir, [chat("x", 0, "hi", "yo", duration_ms=0)])
    sec = _section(render_html(run))
    assert "No step durations were recorded" in sec and 'class="wf"' not in sec
    empty = Run(run_id="01X", name="n")
    assert "No steps recorded." in _section(render_html(empty))


def test_wall_time_is_shown_when_recorded(crashlab_dir):
    run = _run(crashlab_dir, [chat("x", 0, "hi", "yo", duration_ms=5)])
    run.usage.wall_time_ms = 1234.0
    assert "run wall time 1,234 ms" in _section(render_html(run))


def test_recorded_names_are_escaped(crashlab_dir):
    run = _run(crashlab_dir, [tool("x", 0, "<img src=x>", {}, {}, duration_ms=3)])
    sec = _section(render_html(run))
    assert "<img src=x>" not in sec and "&lt;img src=x&gt;" in sec
