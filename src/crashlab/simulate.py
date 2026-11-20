"""Multi-turn conversation and persona simulation.

Drive a live agent through a scripted scenario (Python objects or a JSON / TOML
/ YAML file) inside one recorded :func:`crashlab.record` run. Per-turn
expectations catch constraint forgetting, pushback sycophancy and tool
regressions; ``@crashlab.check`` oracles and declarative rules grade the full
run at the end.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .assertions import Rule, RuleError, _loads, _read_text, evaluate_rules, load_rules
from .capture import record
from .capture._util import entrypoint_of
from .context import current, turn_scope
from .models import CheckResult, Finding, Run
from .store import Store
from .verify.oracles import CheckSpec, apply_outcome, run_checks

__all__ = [
    "Scenario",
    "ScenarioTurn",
    "SimulateError",
    "SimulationResult",
    "TurnResult",
    "load_scenario",
    "parse_scenario",
    "simulate",
]

_HISTORY_NAMES = frozenset({"history", "messages", "chat_history", "conversation", "transcript"})
_TURN_KEYS = frozenset(
    {
        "user",
        "when_reply_matches",
        "expect_tools",
        "forbid_tools",
        "expect_reply_matches",
        "forbid_reply_matches",
    }
)
_SCENARIO_KEYS = frozenset({"name", "turns", "stop_on_reply", "max_turns", "tags", "rules_path"})


class SimulateError(ValueError):
    """A scenario file could not be read, or a simulation could not start."""


@dataclass
class ScenarioTurn:
    """One user turn, with optional expectations and a conditional guard."""

    user: str | None = None
    when_reply_matches: str | None = None
    expect_tools: list[str] | None = None
    forbid_tools: list[str] | None = None
    expect_reply_matches: str | None = None
    forbid_reply_matches: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if value is not None:
                data[key] = value
        return data


@dataclass
class Scenario:
    """A named multi-turn script to run against an agent."""

    name: str
    turns: list[ScenarioTurn] = field(default_factory=list)
    stop_on_reply: str | None = None
    max_turns: int = 10
    tags: list[str] | None = None
    rules_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "turns": [t.to_dict() for t in self.turns],
            "max_turns": self.max_turns,
        }
        if self.stop_on_reply is not None:
            data["stop_on_reply"] = self.stop_on_reply
        if self.tags is not None:
            data["tags"] = list(self.tags)
        if self.rules_path is not None:
            data["rules_path"] = self.rules_path
        return data


@dataclass
class TurnResult:
    """Outcome of one executed or skipped scenario turn."""

    turn: int
    user: str | None
    reply: str | None
    passed: bool
    skipped: bool = False
    tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "turn": self.turn,
            "user": self.user,
            "reply": self.reply,
            "passed": self.passed,
            "skipped": self.skipped,
            "tools": list(self.tools),
            "errors": list(self.errors),
        }
        if self.skip_reason is not None:
            data["skip_reason"] = self.skip_reason
        return data


@dataclass
class SimulationResult:
    """How a scenario ran: per-turn results, final checks and findings."""

    passed: bool
    run_id: str
    turns_executed: int
    turn_results: list[TurnResult] = field(default_factory=list)
    check_results: list[CheckResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "run_id": self.run_id,
            "turns_executed": self.turns_executed,
            "turn_results": [t.to_dict() for t in self.turn_results],
            "check_results": [c.model_dump() for c in self.check_results],
            "findings": [f.model_dump(mode="json") for f in self.findings],
            "errors": list(self.errors),
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"{status}: {self.turns_executed} turn(s) executed, "
            f"{_n_failed_turns(self.turn_results)} failed, "
            f"{_n_failed_checks(self.check_results)} check(s) failed"
            + (f", {len(self.errors)} error(s)" if self.errors else "")
        ]
        if self.run_id:
            lines[0] += f"  run {self.run_id}"
        for turn in self.turn_results:
            if turn.skipped:
                reason = f" ({turn.skip_reason})" if turn.skip_reason else ""
                lines.append(f"  turn {turn.turn}: skipped{reason}")
                continue
            detail = f" — {turn.errors[0]}" if turn.errors else ""
            mark = "pass" if turn.passed else "fail"
            lines.append(f"  turn {turn.turn}: {mark}{detail}")
        for check in self.check_results:
            if check.status == "pass":
                continue
            msg = f": {check.message}" if check.message else ""
            lines.append(f"  check {check.name}: {check.status}{msg}")
        for err in self.errors:
            lines.append(f"  error: {err}")
        return "\n".join(lines)


def _n_failed_turns(turns: Sequence[TurnResult]) -> int:
    return sum(1 for t in turns if not t.passed and not t.skipped)


def _n_failed_checks(checks: Sequence[CheckResult]) -> int:
    return sum(1 for c in checks if c.status != "pass")


def _as_store(store: Store | str | Path | None) -> Store:
    if isinstance(store, Store):
        return store
    if store is None:
        return Store()
    return Store(store)


def _compile(pattern: str | None, loc: str) -> str | None:
    if pattern is None:
        return None
    if not isinstance(pattern, str) or not pattern:
        raise SimulateError(f"{loc}: expected a non-empty regex string")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise SimulateError(f"{loc}: invalid regex {pattern!r}: {exc}") from exc
    return pattern


def _str_list(value: Any, loc: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            raise SimulateError(f"{loc}: expected a non-empty tool name")
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item:
                raise SimulateError(f"{loc}: expected a list of non-empty strings")
            out.append(item)
        return out
    raise SimulateError(f"{loc}: expected a string or list of strings")


def _tags(value: Any, loc: str) -> list[str] | None:
    if value is None:
        return None
    return _str_list(value, loc)


def _parse_turn(item: Any, index: int, source: str) -> ScenarioTurn:
    loc = f"{source} turn {index}"
    if not isinstance(item, dict):
        raise SimulateError(f"{loc}: expected a mapping, got {type(item).__name__}")
    unknown = set(item) - _TURN_KEYS
    if unknown:
        raise SimulateError(f"{loc}: unknown field(s) {sorted(unknown)}")
    user = item.get("user")
    if user is not None and not isinstance(user, str):
        raise SimulateError(f"{loc}: user must be a string")
    return ScenarioTurn(
        user=user,
        when_reply_matches=_compile(item.get("when_reply_matches"), f"{loc}.when_reply_matches"),
        expect_tools=_str_list(item.get("expect_tools"), f"{loc}.expect_tools"),
        forbid_tools=_str_list(item.get("forbid_tools"), f"{loc}.forbid_tools"),
        expect_reply_matches=_compile(
            item.get("expect_reply_matches"), f"{loc}.expect_reply_matches"
        ),
        forbid_reply_matches=_compile(
            item.get("forbid_reply_matches"), f"{loc}.forbid_reply_matches"
        ),
    )


def parse_scenario(data: Any, *, source: str = "<memory>") -> Scenario:
    """Validate a parsed document (a scenario mapping, or ``{scenario: ...}``)."""
    if not isinstance(data, dict):
        raise SimulateError(f"{source}: expected a mapping, got {type(data).__name__}")
    body = data.get("scenario") if "scenario" in data and "turns" not in data else data
    if not isinstance(body, dict):
        raise SimulateError(f"{source}: expected a scenario mapping")
    unknown = set(body) - _SCENARIO_KEYS
    if unknown:
        raise SimulateError(f"{source}: unknown field(s) {sorted(unknown)}")
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SimulateError(f"{source}: name is required")
    raw_turns = body.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise SimulateError(f"{source}: turns must be a non-empty list")
    turns = [_parse_turn(item, i, source) for i, item in enumerate(raw_turns, 1)]
    max_turns = body.get("max_turns", 10)
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 0:
        raise SimulateError(f"{source}: max_turns must be a non-negative integer")
    stop = _compile(body.get("stop_on_reply"), f"{source}.stop_on_reply")
    rules_path = body.get("rules_path")
    if rules_path is not None and not isinstance(rules_path, str):
        raise SimulateError(f"{source}: rules_path must be a string")
    return Scenario(
        name=name.strip(),
        turns=turns,
        stop_on_reply=stop,
        max_turns=max_turns,
        tags=_tags(body.get("tags"), f"{source}.tags"),
        rules_path=rules_path,
    )


def _validate_scenario(scenario: Scenario, *, source: str = "<memory>") -> Scenario:
    if not isinstance(scenario.name, str) or not scenario.name.strip():
        raise SimulateError(f"{source}: name is required")
    if not isinstance(scenario.turns, list) or not scenario.turns:
        raise SimulateError(f"{source}: turns must be a non-empty list")
    if (
        not isinstance(scenario.max_turns, int)
        or isinstance(scenario.max_turns, bool)
        or scenario.max_turns < 0
    ):
        raise SimulateError(f"{source}: max_turns must be a non-negative integer")
    for i, turn in enumerate(scenario.turns, 1):
        if not isinstance(turn, ScenarioTurn):
            raise SimulateError(f"{source} turn {i}: expected ScenarioTurn")
        loc = f"{source} turn {i}"
        _compile(turn.when_reply_matches, f"{loc}.when_reply_matches")
        _compile(turn.expect_reply_matches, f"{loc}.expect_reply_matches")
        _compile(turn.forbid_reply_matches, f"{loc}.forbid_reply_matches")
    _compile(scenario.stop_on_reply, f"{source}.stop_on_reply")
    return scenario


def load_scenario(source: str | Path | Scenario | dict[str, Any]) -> Scenario:
    """Load a scenario from a JSON, TOML or YAML file, or a mapping / object."""
    if isinstance(source, Scenario):
        return _validate_scenario(source, source="<memory>")
    if isinstance(source, dict):
        return parse_scenario(source, source="<memory>")
    path = Path(source).expanduser()
    if not path.is_file():
        raise SimulateError(f"scenario file not found: {path}")
    try:
        raw = _loads(path, _read_text(path))
    except RuleError as exc:
        raise SimulateError(str(exc)) from exc
    return parse_scenario(raw, source=str(path))


def _coerce_scenario(scenario: Scenario | dict[str, Any] | str | Path) -> Scenario:
    return load_scenario(scenario)


def _wants_history(fn: Callable[..., Any]) -> bool:
    """True when ``fn`` should be called as ``fn(user_msg, history)``."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name not in {"self", "cls"}
    ]
    keyword_only = [p for p in sig.parameters.values() if p.kind is p.KEYWORD_ONLY]
    for param in (*params[1:], *keyword_only):
        if param.name in _HISTORY_NAMES:
            return True
    if len(params) >= 2:
        required = [p for p in params if p.default is p.empty]
        if len(required) >= 2:
            return True
    return any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values()) and len(params) < 2


def _invoke_agent(
    fn: Callable[..., Any], user_msg: str, history: list[dict[str, str]] | None
) -> Any:
    if history is None:
        return fn(user_msg)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(user_msg, history)
    keyword_only = [p for p in sig.parameters.values() if p.kind is p.KEYWORD_ONLY]
    for param in keyword_only:
        if param.name in _HISTORY_NAMES:
            return fn(user_msg, **{param.name: list(history)})
    return fn(user_msg, list(history))


def _chat_text(step: Any) -> str:
    parts: list[str] = []
    response = step.response or {}
    for msg in response.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") and msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str) and item:
                    parts.append(item)
    return "\n".join(parts)


def _reply_from(value: Any, steps: Sequence[Any]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "reply", "output", "message"):
            item = value.get(key)
            if isinstance(item, str):
                return item
    if value is not None and not isinstance(value, (bytes, bytearray)):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
        if text and text not in {"null", "None", "{}"}:
            return text
    for step in reversed(list(steps)):
        if getattr(step, "operation", None) != "chat" or step.error is not None:
            continue
        text = _chat_text(step)
        if text.strip():
            return text
    return ""


def _tool_names(steps: Sequence[Any]) -> list[str]:
    return [s.name for s in steps if getattr(s, "operation", None) == "execute_tool"]


def _search(pattern: str, text: str | None) -> bool:
    return re.search(pattern, text or "") is not None


def _eval_turn(turn: ScenarioTurn, index: int, reply: str, tools: list[str]) -> TurnResult:
    errors: list[str] = []
    if turn.expect_tools:
        missing = [name for name in turn.expect_tools if name not in tools]
        if missing:
            errors.append(f"expected tool(s) not called: {', '.join(missing)}")
    if turn.forbid_tools:
        hit = [name for name in turn.forbid_tools if name in tools]
        if hit:
            errors.append(f"called forbidden tool(s): {', '.join(hit)}")
    if turn.expect_reply_matches and not _search(turn.expect_reply_matches, reply):
        errors.append(f"reply did not match /{turn.expect_reply_matches}/")
    if turn.forbid_reply_matches and _search(turn.forbid_reply_matches, reply):
        errors.append(f"reply matched forbidden /{turn.forbid_reply_matches}/")
    return TurnResult(
        turn=index,
        user=turn.user,
        reply=reply,
        passed=not errors,
        tools=list(tools),
        errors=errors,
    )


def _record_checks(checks: Any) -> Any:
    """Runtime-check config for :func:`crashlab.record` (dict / ChecksConfig / False)."""
    if checks is None or checks is False or isinstance(checks, dict):
        return checks
    if type(checks).__name__ == "ChecksConfig":
        return checks
    return None


def _oracle_specs(checks: Any) -> list[CheckSpec] | None:
    if checks is None or checks is False or isinstance(checks, dict):
        return None
    if type(checks).__name__ == "ChecksConfig":
        return None
    items = checks if isinstance(checks, (list, tuple)) else [checks]
    specs: list[CheckSpec] = []
    for item in items:
        if isinstance(item, CheckSpec):
            specs.append(item)
        elif callable(item):
            specs.append(CheckSpec(name=getattr(item, "__name__", "check"), agent=None, fn=item))
        else:
            raise SimulateError(
                "checks= expected a ChecksConfig, dict, False, or check function(s)"
            )
    return specs


def _coerce_rules(rules: Any, scenario: Scenario) -> list[Rule]:
    source = rules if rules is not None else scenario.rules_path
    if source is None:
        return []
    if isinstance(source, (list, tuple)) and source and all(isinstance(r, Rule) for r in source):
        return list(source)
    try:
        return load_rules(source)
    except RuleError as exc:
        raise SimulateError(str(exc)) from exc


def simulate(
    agent_fn: Callable[..., Any],
    scenario: Scenario | dict[str, Any] | str | Path,
    *,
    store: Store | str | Path | None = None,
    checks: Any = None,
    rules: Any = None,
    max_turns: int | None = None,
    tags: Sequence[str] | None = None,
) -> SimulationResult:
    """Run ``scenario`` against ``agent_fn`` inside one recorded run.

    ``agent_fn`` is called as ``agent_fn(user_msg)`` or ``agent_fn(user_msg, history)``
    depending on its signature (a second parameter named ``history`` / ``messages``,
    or two required positional parameters). Each turn is tagged with
    ``crashlab.turn`` (1-indexed). Per-turn ``expect_*`` / ``forbid_*`` fields are
    checked against the steps recorded in that turn; ``@crashlab.check`` oracles and
    declarative ``rules`` then grade the full run.
    """
    if not callable(agent_fn):
        raise SimulateError("agent_fn must be callable")
    if inspect.iscoroutinefunction(agent_fn) or inspect.isasyncgenfunction(agent_fn):
        raise SimulateError("simulate() expects a synchronous agent_fn")
    script = _coerce_scenario(scenario)
    if max_turns is not None and (
        not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 0
    ):
        raise SimulateError("max_turns must be a non-negative integer")
    limit = script.max_turns if max_turns is None else max_turns
    st = _as_store(store)
    loaded_rules = _coerce_rules(rules, script)
    extra_specs = _oracle_specs(checks)
    run_tags: list[str] = []
    for item in (*(script.tags or ()), *(tags or ())):
        if item not in run_tags:
            run_tags.append(item)
    history: list[dict[str, str]] = []
    use_history = _wants_history(agent_fn)
    turn_results: list[TurnResult] = []
    errors: list[str] = []
    executed = 0
    last_reply: str | None = None
    run_id = ""
    try:
        entry = entrypoint_of(agent_fn)
    except Exception:
        entry = None
    with record(
        script.name,
        input={"scenario": script.name, "turns": [t.user for t in script.turns]},
        entrypoint=entry,
        tags=run_tags,
        store_dir=st.root,
        checks=_record_checks(checks),
    ) as run:
        run_id = run.run_id
        try:
            for index, turn in enumerate(script.turns, 1):
                if executed >= limit:
                    break
                if turn.when_reply_matches and (
                    last_reply is None or not _search(turn.when_reply_matches, last_reply)
                ):
                    turn_results.append(
                        TurnResult(
                            turn=index,
                            user=turn.user,
                            reply=None,
                            passed=True,
                            skipped=True,
                            skip_reason="when_reply_matches did not hit",
                        )
                    )
                    continue
                if turn.user is None:
                    turn_results.append(
                        TurnResult(
                            turn=index,
                            user=None,
                            reply=None,
                            passed=True,
                            skipped=True,
                            skip_reason="no user message",
                        )
                    )
                    continue
                ctx = current()
                start = len(ctx.steps) if ctx is not None else 0
                executed += 1
                try:
                    with turn_scope(index):
                        raw = _invoke_agent(agent_fn, turn.user, history if use_history else None)
                except Exception as exc:
                    steps = current().steps[start:] if current() is not None else []
                    result = TurnResult(
                        turn=index,
                        user=turn.user,
                        reply=None,
                        passed=False,
                        tools=_tool_names(steps),
                        errors=[f"{type(exc).__name__}: {exc}"],
                    )
                    turn_results.append(result)
                    errors.append(f"turn {index}: {type(exc).__name__}: {exc}")
                    break
                steps = current().steps[start:] if current() is not None else []
                reply = _reply_from(raw, steps)
                last_reply = reply
                run.output = reply
                history.append({"role": "user", "content": turn.user})
                history.append({"role": "assistant", "content": reply})
                turn_results.append(_eval_turn(turn, index, reply, _tool_names(steps)))
                if script.stop_on_reply and _search(script.stop_on_reply, reply):
                    break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    check_results: list[CheckResult] = []
    findings: list[Finding] = []
    finished: Run | None = None
    if run_id:
        try:
            finished = st.load_run(run_id)
        except Exception:
            finished = None
    if finished is not None:
        findings = list(finished.findings)
        check_results = list(finished.checks)
        if extra_specs:
            check_results.extend(run_checks(finished, extra_specs))
        if loaded_rules:
            check_results.extend(evaluate_rules(finished, loaded_rules))
        if check_results:
            apply_outcome(finished, check_results)
            with contextlib.suppress(Exception):
                st.update_run(finished)
        findings = list(finished.findings)

    passed = (
        _n_failed_turns(turn_results) == 0
        and _n_failed_checks(check_results) == 0
        and not errors
        and (finished is None or finished.status != "error")
    )
    return SimulationResult(
        passed=passed,
        run_id=run_id,
        turns_executed=executed,
        turn_results=turn_results,
        check_results=check_results,
        findings=findings,
        errors=errors,
    )
