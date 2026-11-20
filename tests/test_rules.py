"""Declarative policy rules (crashlab.assertions / crashlab rules). Offline only."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.assertions import (
    RULE_TYPES,
    RuleError,
    discover_rules,
    evaluate_rule,
    evaluate_rules,
    load_rules,
    parse_rules,
    parse_simple_yaml,
    register_rules,
    rules_to_checks,
    validate_rules,
)
from crashlab.cli import app
from crashlab.models import Finding, Run, Step, Usage
from crashlab.store import Store
from crashlab.suite import eval_runs
from crashlab.verify import apply_outcome, run_checks

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
ENV = {"COLUMNS": "220", "TERM": "xterm-256color", "NO_COLOR": "1"}
runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def _clean_oracles():
    from crashlab.verify.oracles import _registry

    before = dict(_registry)
    yield
    _registry.clear()
    _registry.update(before)


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def chat(
    run_id: str,
    idx: int,
    text: str = "ok",
    *,
    tokens: int = 100,
    cost: float = 0.001,
    tools: list | None = None,
    agent: str | None = None,
) -> Step:
    attrs = {
        "gen_ai.request.model": "fake-model",
        "gen_ai.usage.input_tokens": tokens,
        "gen_ai.usage.output_tokens": 10,
        "crashlab.usage.cost_usd": cost,
    }
    if agent:
        attrs["gen_ai.agent.name"] = agent
    response = {
        "messages": [{"role": "assistant", "content": text}],
        "tool_calls": tools or [],
    }
    return Step(
        run_id=run_id,
        idx=idx,
        span_id=f"s{idx}",
        operation="chat",
        name="fake-model",
        attributes=attrs,
        request={"messages": [{"role": "user", "content": "hi"}]},
        response=response,
        request_hash=f"h{idx}",
        started_at=T0 + timedelta(seconds=idx),
        duration_ms=100.0,
    )


def tool(
    run_id: str, idx: int, name: str, args: dict | None = None, *, agent: str | None = None
) -> Step:
    attrs = {"gen_ai.tool.name": name}
    if agent:
        attrs["gen_ai.agent.name"] = agent
    return Step(
        run_id=run_id,
        idx=idx,
        span_id=f"t{idx}",
        operation="execute_tool",
        name=name,
        attributes=attrs,
        request={"args": args or {}},
        response={"result": "ok"},
        request_hash=f"th{idx}",
        started_at=T0 + timedelta(seconds=idx),
        duration_ms=20.0,
    )


def make_run(
    rid: str = "RUNRULE0000000000000000001",
    *,
    name: str = "bot",
    output: object = "Your refund of $49.00 is on the way.",
    tools: list[tuple] | None = None,
    findings: list[Finding] | None = None,
    tokens: int = 100,
    cost: float = 0.01,
    duration_ms: float = 1500.0,
    extra_steps: list[Step] | None = None,
) -> Run:
    steps: list[Step] = []
    idx = 0
    for item in tools or [
        ("get_order", {"order_id": "A17"}),
        ("issue_refund", {"order_id": "A17", "amount": 49}),
    ]:
        tname, args, *rest = item if not isinstance(item, str) else (item, {})
        agent = rest[0] if rest else None
        steps.append(chat(rid, idx, tools=[{"id": f"c{idx}", "name": tname}], agent=agent))
        idx += 1
        steps.append(tool(rid, idx, tname, args, agent=agent))
        idx += 1
    steps.append(chat(rid, idx, text=output if isinstance(output, str) else "ok"))
    if extra_steps:
        steps.extend(extra_steps)
    n_steps = len(steps)
    return Run(
        run_id=rid,
        name=name,
        status="ok",
        outcome="unknown",
        output=output,
        started_at=T0,
        ended_at=T0 + timedelta(milliseconds=duration_ms),
        findings=list(findings or []),
        steps=steps,
        usage=Usage(
            input_tokens=tokens,
            output_tokens=10,
            total_tokens=tokens + 10,
            cost_usd=cost,
            steps=n_steps,
            llm_calls=sum(1 for s in steps if s.operation == "chat"),
            tool_calls=len(tools or [("get_order", {}), ("issue_refund", {})]),
            wall_time_ms=duration_ms,
        ),
    )


def put(store: Store, run: Run) -> Run:
    store.begin_run(run)
    for step in run.steps:
        store.add_step(step)
    store.end_run(run)
    return store.load_run(run.run_id)


# --- parsing -------------------------------------------------------------------------


def test_parse_json_toml_yaml_and_bare_list(tmp_path):
    payload = {
        "rules": [
            {"type": "must_call", "tools": ["get_order"], "id": "need-order"},
            {"type": "max_cost_usd", "max": 0.05},
        ]
    }
    (tmp_path / "rules.json").write_text(json.dumps(payload), "utf-8")
    (tmp_path / "rules.toml").write_text(
        '[[rules]]\ntype = "must_call"\ntools = ["get_order"]\nid = "need-order"\n\n'
        '[[rules]]\ntype = "max_cost_usd"\nmax = 0.05\n',
        "utf-8",
    )
    (tmp_path / "rules.yaml").write_text(
        "rules:\n  - type: must_call\n    tools: [get_order]\n    id: need-order\n"
        "  - type: max_cost_usd\n    max: 0.05\n",
        "utf-8",
    )
    for name in ("rules.json", "rules.toml", "rules.yaml"):
        rules = load_rules(tmp_path / name)
        assert [r.id for r in rules] == ["need-order", "max_cost_usd:0.05"]
        assert rules[0].type == "must_call"
    bare = parse_rules([{"type": "must_not_call", "tool": "delete_user"}])
    assert bare[0].type == "must_not_call" and bare[0].id == "must_not_call:delete_user"


def test_hyphenated_type_and_custom_metadata():
    rules = parse_rules(
        {
            "type": "must-call",
            "tool": "get_order",
            "id": "need-it",
            "severity": "error",
            "message": "look up the order first",
            "name": "refund-bot",
            "agent": "refund-bot",
        }
    )
    rule = rules[0]
    assert rule.type == "must_call"
    assert rule.severity == "error" and rule.message == "look up the order first"
    assert rule.name == "refund-bot" and rule.agent == "refund-bot"


def test_unknown_type_and_bad_schema():
    with pytest.raises(RuleError, match="unknown rule type"):
        parse_rules([{"type": "explode"}])
    with pytest.raises(RuleError, match="missing 'tool'"):
        parse_rules([{"type": "must_call"}])
    with pytest.raises(RuleError, match="missing 'max'"):
        parse_rules([{"type": "max_cost_usd"}])
    with pytest.raises(RuleError, match="invalid regex"):
        parse_rules([{"type": "reply_matches", "pattern": "("}])
    with pytest.raises(RuleError, match="duplicate rule id"):
        parse_rules(
            [
                {"type": "must_call", "tool": "a", "id": "x"},
                {"type": "must_not_call", "tool": "b", "id": "x"},
            ]
        )
    with pytest.raises(RuleError, match="rules"):
        parse_rules({"hello": 1})


def test_builtin_yaml_parser_handles_rule_lists():
    text = """
---
# comment
rules:
  - type: must_call
    tools:
      - get_order
      - issue_refund
    min_calls: 1
    name: refund-bot
  - type: tool_arg_matches
    tool: issue_refund
    constraint: "amount <= 500"
  - type: json_valid_reply
    required_keys: [status]
    properties:
      status: string
      amount: number
  - {type: max_cost_usd, max: 0.05}
"""
    data = parse_simple_yaml(text)
    rules = parse_rules(data)
    assert [r.type for r in rules] == [
        "must_call",
        "tool_arg_matches",
        "json_valid_reply",
        "max_cost_usd",
    ]
    assert rules[0].params["tools"] == ["get_order", "issue_refund"]
    assert rules[0].name == "refund-bot"
    assert rules[1].params["constraint"] == "amount <= 500"
    assert rules[2].params["required_keys"] == ["status"]
    assert rules[2].params["properties"]["amount"] == "number"
    assert rules[3].params["max"] == 0.05


def test_assertions_does_not_import_yaml_at_module_level():
    source = Path(crashlab.assertions.__file__).read_text("utf-8")
    # only comments / function-local imports may mention the yaml package
    for i, line in enumerate(source.splitlines(), 1):
        code = line.split("#", 1)[0]
        if (
            code
            and not code[:1].isspace()
            and re.match(r"^(import yaml|from yaml\b)", code.strip())
        ):
            raise AssertionError(f"top-level yaml import at line {i}: {line}")
    assert "import yaml" in source  # the lazy import inside _yaml_loads must still exist


def test_builtin_yaml_used_when_pyyaml_missing(tmp_path, monkeypatch):
    import builtins
    import sys

    path = tmp_path / "rules.yaml"
    path.write_text("- type: must_call\n  tool: get_order\n", "utf-8")
    orig_import = builtins.__import__

    def hook(name, *a, **k):
        if name == "yaml":
            raise ImportError("hidden")
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", hook)
    sys.modules.pop("yaml", None)
    rules = load_rules(path)
    assert rules[0].type == "must_call"


# --- discovery -----------------------------------------------------------------------


def test_auto_discover_crashlab_rules_and_pyproject(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path, rules = discover_rules(tmp_path)
    assert path is None and rules == []
    assert load_rules("auto", start=tmp_path) == []
    assert load_rules("off") == []
    assert load_rules(False) == []

    rules_dir = tmp_path / ".crashlab"
    rules_dir.mkdir()
    (rules_dir / "rules.json").write_text(
        json.dumps([{"type": "must_call", "tool": "get_order", "id": "from-json"}]), "utf-8"
    )
    found, loaded = discover_rules(tmp_path)
    assert found == rules_dir / "rules.json"
    assert loaded[0].id == "from-json"

    (tmp_path / "pyproject.toml").write_text(
        '[tool.crashlab.rules]\nrules = [{type = "max_steps", max = 3, id = "from-pp"}]\n',
        "utf-8",
    )
    # json still wins when both exist
    assert discover_rules(tmp_path)[1][0].id == "from-json"
    (rules_dir / "rules.json").unlink()
    found, loaded = discover_rules(tmp_path)
    assert found == tmp_path / "pyproject.toml"
    assert loaded[0].id == "from-pp"

    array_form = tmp_path / "nested"
    array_form.mkdir()
    (array_form / "pyproject.toml").write_text(
        '[[tool.crashlab.rules]]\ntype = "max_tokens"\nmax = 100\nid = "arr"\n', "utf-8"
    )
    loaded = load_rules(array_form / "pyproject.toml")
    assert loaded[0].id == "arr" and loaded[0].type == "max_tokens"


def test_missing_file_and_validate_auto(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuleError, match="not found"):
        load_rules(tmp_path / "nope.yaml")
    with pytest.raises(RuleError, match="no rules file found"):
        validate_rules("auto", start=tmp_path)


# --- evaluation ----------------------------------------------------------------------


def test_must_call_and_must_not_call():
    run = make_run()
    results = evaluate_rules(
        run,
        parse_rules(
            [
                {"type": "must_call", "tools": ["get_order", "issue_refund"]},
                {"type": "must_call", "tool": "wire_money", "id": "need-wire", "min_calls": 1},
                {"type": "must_not_call", "tools": ["delete_user"]},
                {"type": "must_not_call", "tool": "issue_refund", "id": "no-refund"},
            ]
        ),
    )
    by = {r.name: r for r in results}
    assert by["must_call:get_order,issue_refund"].status == "pass"
    assert by["need-wire"].status == "fail" and "wire_money" in (by["need-wire"].message or "")
    assert by["must_not_call:delete_user"].status == "pass"
    assert by["no-refund"].status == "fail"


def test_tool_order_and_max_tool_calls():
    run = make_run()
    assert (
        evaluate_rule(
            parse_rules([{"type": "tool_order", "tools": ["get_order", "issue_refund"]}])[0], run
        ).status
        == "pass"
    )
    bad = make_run(tools=[("issue_refund", {"amount": 1}), ("get_order", {"order_id": "A"})])
    result = evaluate_rule(
        parse_rules([{"type": "tool_order", "tools": ["get_order", "issue_refund"]}])[0], bad
    )
    assert result.status == "fail" and "must appear before" in (result.message or "")

    assert (
        evaluate_rule(parse_rules([{"type": "max_tool_calls", "max": 10}])[0], run).status == "pass"
    )
    assert (
        evaluate_rule(parse_rules([{"type": "max_tool_calls", "max": 1}])[0], run).status == "fail"
    )
    assert (
        evaluate_rule(
            parse_rules([{"type": "max_tool_calls", "tool": "issue_refund", "max": 1}])[0], run
        ).status
        == "pass"
    )
    assert (
        evaluate_rule(
            parse_rules([{"type": "max_tool_calls", "tools": {"get_order": 0}}])[0], run
        ).status
        == "fail"
    )


def test_budget_and_duration_and_steps():
    run = make_run(cost=0.02, tokens=100, duration_ms=1500)
    assert evaluate_rules(
        run,
        parse_rules(
            [
                {"type": "max_cost_usd", "max": 0.05, "id": "cost-ok"},
                {"type": "max_cost_usd", "max": 0.001, "id": "cost-bad"},
                {"type": "max_tokens", "max": 10_000, "id": "tok-ok"},
                {"type": "max_tokens", "max": 10, "id": "tok-bad"},
                {"type": "max_duration_ms", "max": 5000, "id": "dur-ok"},
                {"type": "max_duration_ms", "max": 100, "id": "dur-bad"},
                {"type": "max_steps", "max": 50, "id": "steps-ok"},
                {"type": "max_steps", "max": 1, "id": "steps-bad"},
            ]
        ),
    )
    by = {
        r.name: r.status
        for r in evaluate_rules(
            run,
            parse_rules(
                [
                    {"type": "max_cost_usd", "max": 0.05, "id": "cost-ok"},
                    {"type": "max_cost_usd", "max": 0.001, "id": "cost-bad"},
                    {"type": "max_tokens", "max": 10_000, "id": "tok-ok"},
                    {"type": "max_tokens", "max": 10, "id": "tok-bad"},
                    {"type": "max_duration_ms", "max": 5000, "id": "dur-ok"},
                    {"type": "max_duration_ms", "max": 100, "id": "dur-bad"},
                    {"type": "max_steps", "max": 50, "id": "steps-ok"},
                    {"type": "max_steps", "max": 1, "id": "steps-bad"},
                ]
            ),
        )
    }
    assert by["cost-ok"] == by["tok-ok"] == by["dur-ok"] == by["steps-ok"] == "pass"
    assert by["cost-bad"] == by["tok-bad"] == by["dur-bad"] == by["steps-bad"] == "fail"


def test_reply_matches_and_contains():
    run = make_run(output="Your refund of $49.00 is on the way.")
    rules = parse_rules(
        [
            {"type": "reply_contains", "text": "REFUND", "id": "has-refund"},
            {"type": "reply_contains", "text": "REFUND", "ignore_case": False, "id": "case"},
            {"type": "reply_not_contains", "text": "deleted", "id": "no-del"},
            {"type": "reply_matches", "pattern": r"\$\d+", "id": "dollar"},
            {"type": "reply_not_matches", "pattern": r"(?i)deleted", "id": "no-del-re"},
            {"type": "reply_matches", "pattern": "never-this", "id": "miss"},
        ]
    )
    by = {r.name: r.status for r in evaluate_rules(run, rules)}
    assert by["has-refund"] == by["no-del"] == by["dollar"] == by["no-del-re"] == "pass"
    assert by["case"] == by["miss"] == "fail"


def test_tool_arg_matches_and_forbidden():
    run = make_run()
    ok = parse_rules(
        [{"type": "tool_arg_matches", "tool": "issue_refund", "constraint": "amount <= 500"}]
    )[0]
    assert evaluate_rule(ok, run).status == "pass"
    assert (
        evaluate_rule(
            parse_rules(
                [{"type": "tool_arg_matches", "tool": "issue_refund", "arg": "amount", "max": 10}]
            )[0],
            run,
        ).status
        == "fail"
    )
    assert (
        evaluate_rule(
            parse_rules(
                [
                    {
                        "type": "tool_arg_matches",
                        "tool": "issue_refund",
                        "args": {"amount": {"min": 1, "max": 100}},
                    }
                ]
            )[0],
            run,
        ).status
        == "pass"
    )
    assert (
        evaluate_rule(
            parse_rules(
                [
                    {
                        "type": "tool_arg_forbidden",
                        "tool": "issue_refund",
                        "arg": "amount",
                        "min": 100,
                    }
                ]
            )[0],
            run,
        ).status
        == "pass"
    )
    assert (
        evaluate_rule(
            parse_rules(
                [
                    {
                        "type": "tool_arg_forbidden",
                        "tool": "issue_refund",
                        "constraint": "amount <= 500",
                    }
                ]
            )[0],
            run,
        ).status
        == "fail"
    )
    missing = evaluate_rule(
        parse_rules([{"type": "tool_arg_matches", "tool": "wire", "arg": "amount", "max": 1}])[0],
        run,
    )
    assert missing.status == "fail" and "was not called" in (missing.message or "")


def test_json_valid_reply_required_keys_and_properties():
    run = make_run(output='{"status": "ok", "amount": 49}')
    rules = parse_rules(
        [
            {
                "type": "json_valid_reply",
                "required_keys": ["status", "amount"],
                "properties": {"status": "string", "amount": "number"},
                "id": "json-ok",
            }
        ]
    )
    assert evaluate_rule(rules[0], run).status == "pass"
    fenced = make_run(output='Here you go:\n```json\n{"status": "ok"}\n```')
    assert (
        evaluate_rule(
            parse_rules([{"type": "json_valid_reply", "required_keys": ["status"]}])[0], fenced
        ).status
        == "pass"
    )
    obj = make_run(output={"status": "ok", "amount": "nope"})
    result = evaluate_rule(
        parse_rules(
            [
                {
                    "type": "json_valid_reply",
                    "properties": {"amount": {"type": "number"}},
                    "id": "types",
                }
            ]
        )[0],
        obj,
    )
    assert result.status == "fail" and "number" in (result.message or "")
    bad = evaluate_rule(
        parse_rules([{"type": "json_valid_reply"}])[0], make_run(output="not json at all")
    )
    assert bad.status == "fail"


def test_no_findings_above():
    clean = make_run()
    dirty = make_run(
        findings=[
            Finding(kind="budget", severity="critical", message="too many steps", step=2),
            Finding(kind="loop", severity="warn", message="repeated", step=1),
        ]
    )
    rule = parse_rules([{"type": "no_findings_above", "threshold": "error"}])[0]
    assert evaluate_rule(rule, clean).status == "pass"
    result = evaluate_rule(rule, dirty)
    assert result.status == "fail" and "budget:critical" in (result.message or "")
    warn = parse_rules([{"type": "no_findings_above", "threshold": "warn"}])[0]
    assert evaluate_rule(warn, dirty).status == "fail"
    # info floor would also catch warn; critical only catches the budget finding
    hard = parse_rules([{"type": "no_findings_above", "threshold": "critical"}])[0]
    assert evaluate_rule(hard, dirty).status == "fail"
    only_warn = make_run(findings=[Finding(kind="loop", severity="warn", message="x")])
    assert evaluate_rule(hard, only_warn).status == "pass"


def test_name_and_agent_filters():
    run = make_run(
        name="refund-bot",
        tools=[
            ("get_order", {"order_id": "A"}, "refund-bot"),
            ("search", {"q": "x"}, "researcher"),
        ],
    )
    skipped = parse_rules(
        [{"type": "must_call", "tool": "get_order", "name": "other", "id": "skip"}]
    )[0]
    assert evaluate_rule(skipped, run).status == "pass"
    scoped = parse_rules(
        [{"type": "must_not_call", "tool": "search", "agent": "refund-bot", "id": "no-search"}]
    )[0]
    # search was called by researcher, not refund-bot
    assert evaluate_rule(scoped, run).status == "pass"
    must = parse_rules(
        [{"type": "must_call", "tool": "search", "agent": "refund-bot", "id": "need-search"}]
    )[0]
    assert evaluate_rule(must, run).status == "fail"
    custom = parse_rules(
        [
            {
                "type": "must_call",
                "tool": "wire",
                "id": "wired",
                "message": "must wire the money",
                "name": "refund-bot",
            }
        ]
    )[0]
    result = evaluate_rule(custom, run)
    assert result.status == "fail" and result.message and "must wire the money" in result.message


def test_rules_to_checks_and_run_checks_and_register():
    run = make_run()
    rules = parse_rules(
        [
            {"type": "must_call", "tool": "get_order", "id": "need-order"},
            {"type": "must_call", "tool": "wire", "id": "need-wire"},
        ]
    )
    specs = rules_to_checks(rules)
    assert [s.name for s in specs] == ["need-order", "need-wire"]
    results = run_checks(run, specs)
    assert results[0].status == "pass" and results[1].status == "fail"
    apply_outcome(run, results)
    assert run.outcome == "fail"
    register_rules(parse_rules([{"type": "must_not_call", "tool": "delete_user", "id": "no-del"}]))
    registered = run_checks(run)
    assert any(r.name == "no-del" and r.status == "pass" for r in registered)


def test_all_rule_types_are_documented_constants():
    assert "must_call" in RULE_TYPES and "json_valid_reply" in RULE_TYPES
    assert len(RULE_TYPES) == 16


# --- suite / check integration -------------------------------------------------------


def test_eval_runs_applies_rules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store()
    put(store, make_run("RUNEVALR00000000000000001"))
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps([{"type": "must_call", "tool": "wire_money", "id": "need-wire"}]), "utf-8"
    )
    report = eval_runs(store, rules=path)
    assert report.failed == 1
    assert "need-wire" in report.results[0].failing_checks
    ok = eval_runs(store, rules=[{"type": "must_call", "tool": "get_order"}])
    assert ok.passed == 1
    skipped = eval_runs(store, rules="off")
    assert skipped.passed == 1


# --- CLI -----------------------------------------------------------------------------


def test_rules_help_strips_ansi_and_lists_subcommands():
    res = runner.invoke(app, ["rules", "--help"], env=ENV)
    out = _plain(res.output)
    assert res.exit_code == 0, res.output
    assert "validate" in out and "test" in out
    res = runner.invoke(app, ["rules", "validate", "--help"], env=ENV)
    out = _plain(res.output)
    assert res.exit_code == 0 and "--json" in out
    res = runner.invoke(app, ["rules", "test", "--help"], env=ENV)
    out = _plain(res.output)
    assert res.exit_code == 0 and "--all" in out and "--name" in out and "--json" in out
    res = runner.invoke(app, ["--help"], env=ENV)
    assert "rules" in _plain(res.output)
    res = runner.invoke(app, ["check", "--help"], env=ENV)
    assert "--rules" in _plain(res.output)
    res = runner.invoke(app, ["eval", "--help"], env=ENV)
    assert "--rules" in _plain(res.output)


def test_rules_validate_and_test_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store()
    put(store, make_run("RUNCLIR0000000000000000001"))
    good = tmp_path / "ok.yaml"
    good.write_text(
        "rules:\n  - type: must_call\n    tool: get_order\n    id: need-order\n", "utf-8"
    )
    res = runner.invoke(app, ["rules", "validate", str(good)], env=ENV)
    assert res.exit_code == 0, res.output
    out = _plain(res.output)
    assert "need-order" in out and "1 rule" in out
    res = runner.invoke(app, ["rules", "validate", str(good), "--json"], env=ENV)
    payload = json.loads(res.output)
    assert payload["ok"] is True and payload["count"] == 1

    bad = tmp_path / "bad.yaml"
    bad.write_text("rules:\n  - type: nope\n", "utf-8")
    res = runner.invoke(app, ["rules", "validate", str(bad), "--json"], env=ENV)
    assert res.exit_code == 1
    assert json.loads(res.output)["ok"] is False

    res = runner.invoke(app, ["rules", "test", str(good), "last"], env=ENV)
    assert res.exit_code == 0, res.output
    assert "pass" in _plain(res.output)

    fail_rules = tmp_path / "fail.yaml"
    fail_rules.write_text(
        "rules:\n  - type: must_call\n    tool: wire\n    id: need-wire\n", "utf-8"
    )
    res = runner.invoke(app, ["rules", "test", str(fail_rules), "--all", "--json"], env=ENV)
    assert res.exit_code == 1, res.output
    body = json.loads(res.output)
    assert body["ok"] is False
    assert body["results"][0]["checks"][0]["status"] == "fail"


def test_check_and_eval_cli_rules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store()
    put(store, make_run("RUNCHK00000000000000000001"))
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps([{"type": "must_call", "tool": "wire", "id": "need-wire"}]), "utf-8"
    )
    res = runner.invoke(
        app, ["check", "last", "--rules", str(rules), "--json", "--no-save"], env=ENV
    )
    assert res.exit_code == 1, res.output
    payload = json.loads(res.output)
    names = [c["name"] for c in payload["checks"]]
    assert "need-wire" in names
    assert payload["outcome"] == "fail"

    res = runner.invoke(app, ["eval", "--rules", str(rules), "--json"], env=ENV)
    assert res.exit_code == 1, res.output
    report = json.loads(res.output)
    assert report["failed"] == 1 or report["results"][0]["status"] == "fail"

    res = runner.invoke(app, ["eval", "--rules", "off", "--json"], env=ENV)
    assert res.exit_code == 0, res.output
