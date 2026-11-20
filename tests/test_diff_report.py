"""HTML / Markdown diff reports and ``crashlab diff --fail-on-regression``."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.compare import diff
from crashlab.diff_report import has_regression, render_diff_html, render_diff_markdown
from crashlab.models import CheckResult, ErrorInfo, Finding, Run, Usage
from crashlab.store import Store
from tests.debug_fixtures import chat, make_run, tool

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _run(
    rid: str,
    *,
    name: str = "bot",
    status: str = "ok",
    outcome: str = "pass",
    steps: list | None = None,
    checks: list[CheckResult] | None = None,
    findings: list[Finding] | None = None,
    tokens: tuple[int, int] = (100, 20),
    cost: float | None = 0.001,
    wall: float | None = 1500,
    output: str = "done",
    tool_name: str = "lookup",
    extra: bool = False,
    error: ErrorInfo | None = None,
) -> Run:
    if steps is None:
        steps = [
            chat(
                rid,
                0,
                "go",
                tool_call=(tool_name, {"key": "a"}),
                tokens=tokens,
                cost=cost,
                duration_ms=200,
            ),  # fmt: skip
            tool(rid, 1, tool_name, {"key": "a"}, {"ok": True}, duration_ms=10),
            chat(
                rid,
                2,
                None,
                output,
                history=[
                    {
                        "role": "user",
                        "parts": [
                            {"type": "text", "content": "go"},
                        ],
                    }
                ],
                tokens=tokens,
                cost=cost,
                duration_ms=300,
            ),  # fmt: skip
        ]
        if extra:
            steps.append(chat(rid, 3, None, "more", duration_ms=50, cost=cost, tokens=tokens))
    n_chat = sum(s.operation == "chat" for s in steps)
    run = Run(
        run_id=rid, name=name, status=status, outcome=outcome,  # type: ignore[arg-type]
        checks=checks or [], findings=findings or [], output=output, error=error,
        usage=Usage(
            steps=len(steps), llm_calls=n_chat,
            tool_calls=sum(s.operation == "execute_tool" for s in steps),
            input_tokens=tokens[0] * n_chat, output_tokens=tokens[1] * n_chat,
            total_tokens=(tokens[0] + tokens[1]) * n_chat,
            cost_usd=None if cost is None else cost * n_chat, wall_time_ms=wall,
        ),
    )  # fmt: skip
    run.steps = steps
    return run


def test_render_diff_html_header_pills_and_steps():
    a = _run("runA")
    b = _run(
        "runB",
        outcome="fail",
        wall=2000,
        cost=0.002,
        tokens=(150, 30),
        checks=[CheckResult(name="says_done", status="fail", message="nope")],
        findings=[Finding(kind="loop", severity="warn", message="repeated call", step=1)],
        extra=True,
        tool_name="search",
    )
    a.checks = [CheckResult(name="says_done", status="pass")]
    d = diff(a, b)
    html = render_diff_html(d, a, b)
    assert html.startswith("<!doctype html>")
    for needle in (
        "crashlab diff",
        "runA",
        "runB",
        "Status",
        "Duration",
        "Cost",
        "Tokens",
        "Findings",
        "Trajectory",
        "edit distance",
        "similarity",
        "Regressions and improvements",
        "outcome pass → fail",
        "check says_done pass → fail",
        "new warn finding: loop",
        'id="steps"',
        "cost Δ",
        "latency Δ",
        "Diverged",
        "data-filter",
        "only B",
    ):
        assert needle in html, needle
    assert "search" in html
    assert d.tool_edit_distance >= 1


def test_render_diff_html_escapes_script_and_is_self_contained():
    evil = "<script>alert(1)</script>"
    a = _run("A", name=evil, output=evil)
    b = _run(
        "B",
        name='<img src=x onerror="alert(1)">',
        outcome="fail",
        output=evil,
        checks=[CheckResult(name="<b>check</b>", status="fail", message=evil)],
        findings=[Finding(kind=evil, severity="error", message="<script>boom</script>", step=0)],
    )
    html = render_diff_html(diff(a, b), a, b)
    assert "<script>alert(1)</script>" not in html
    assert "<script>boom</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;script&gt;boom&lt;/script&gt;" in html
    assert "<img src=x" not in html and "&lt;img" in html
    assert "<b>check</b>" not in html and "&lt;b&gt;check&lt;/b&gt;" in html
    # one inline filter script; no external assets or injected tags
    assert html.count("<script>") == 1
    assert not re.search(r"<(?:link|iframe|img)\b", html, re.I)
    assert "https://" not in html and "http://" not in html
    assert "@import" not in html and "url(" not in html and "cdn." not in html.lower()


def test_render_diff_markdown_pr_table():
    a = _run("AAA")
    b = _run(
        "BBB",
        outcome="fail",
        checks=[CheckResult(name="says_done", status="fail")],
        findings=[Finding(kind="loop", severity="error", message="stuck", step=1)],
    )
    a.checks = [CheckResult(name="says_done", status="pass")]
    md = render_diff_markdown(diff(a, b), a, b)
    assert md.startswith("# crashlab diff:")
    assert "| Metric | A | B | Δ |" in md
    for col in ("Outcome", "Status", "Checks", "Findings", "Cost", "Tokens", "Latency"):
        assert f"| {col} |" in md, col
    assert "Trajectory edit distance" in md
    assert "## Regressions" in md
    assert "outcome pass → fail" in md
    assert "check says_done pass → fail" in md
    assert "new error finding: loop" in md
    assert "## Improvements" in md
    assert "None" in md.split("## Improvements", 1)[1]


def test_has_regression_status_findings_and_checks():
    clean = _run("A")
    assert has_regression(diff(clean, _run("B"))) is False

    failed = _run("B", outcome="fail")
    assert has_regression(diff(clean, failed)) is True

    errored = _run("B", status="error", error=ErrorInfo(type="RuntimeError", message="boom"))
    assert has_regression(diff(clean, errored)) is True

    warned = _run("B", findings=[Finding(kind="loop", severity="warn", message="repeat", step=1)])
    assert has_regression(diff(clean, warned)) is True

    info_only = _run("B", findings=[Finding(kind="note", severity="info", message="fyi", step=0)])
    assert has_regression(diff(clean, info_only)) is False

    flipped = _run("B", checks=[CheckResult(name="says_done", status="fail")])
    clean.checks = [CheckResult(name="says_done", status="pass")]
    assert has_regression(diff(clean, flipped)) is True

    improved = _run("C", outcome="pass")
    bad = _run(
        "A2",
        outcome="fail",
        status="error",
        error=ErrorInfo(type="RuntimeError", message="x"),
        checks=[CheckResult(name="says_done", status="fail")],
        findings=[Finding(kind="loop", severity="warn", message="repeat", step=1)],
    )
    improved.checks = [CheckResult(name="says_done", status="pass")]
    assert has_regression(diff(bad, improved)) is False


def test_cli_html_markdown_and_fail_on_regression(crashlab_dir, tmp_path):
    store = Store()
    a = make_run(store, 1, outcome="pass", checks=[CheckResult(name="says_done", status="pass")])
    b = make_run(
        store, 2, outcome="fail",
        checks=[CheckResult(name="says_done", status="fail", message="nope")],
        findings=[Finding(kind="loop", severity="warn", message="repeat", step=1)],
    )  # fmt: skip
    c = make_run(store, 3, outcome="pass", checks=[CheckResult(name="says_done", status="pass")])
    html = tmp_path / "nested" / "diff.html"
    md = tmp_path / "out.md"
    failed = runner.invoke(
        app,
        [
            "diff", a.run_id, b.run_id, "--html", str(html), "--markdown", str(md),
            "--fail-on-regression",
        ],
    )  # fmt: skip
    assert failed.exit_code == 1, failed.output
    assert html.is_file() and html.read_text().startswith("<!doctype html>")
    text = md.read_text()
    assert "| Outcome |" in text and "pass → fail" in text

    ok = runner.invoke(app, ["diff", a.run_id, c.run_id, "--fail-on-regression"])
    assert ok.exit_code == 0, ok.output

    stdout_md = runner.invoke(app, ["diff", a.run_id, b.run_id, "--markdown", "-"])
    assert stdout_md.exit_code == 0, stdout_md.output
    assert "# crashlab diff:" in stdout_md.output
    assert "| Metric | A | B | Δ |" in stdout_md.output

    js = runner.invoke(
        app, ["diff", a.run_id, b.run_id, "--json", "--html", str(tmp_path / "x.html")]
    )
    assert js.exit_code == 0, js.output
    payload = json.loads(js.output)
    assert payload["a"] == a.run_id and payload["b"] == b.run_id
    assert "tool_edit_distance" in payload


def test_diff_help_lists_report_flags():
    res = runner.invoke(app, ["diff", "--help"])
    assert res.exit_code == 0, res.output
    out = _plain(res.output)
    assert "--html" in out
    assert "--markdown" in out
    assert "--fail-on-regression" in out
    assert "--json" in out
    assert "--full" in out
