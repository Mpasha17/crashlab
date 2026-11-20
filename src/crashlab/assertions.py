"""Declarative policy assertions: rule files that grade a recorded run.

A rule file is JSON, TOML or YAML (``.crashlab/rules.json`` / ``.toml`` / ``.yaml``,
or ``[tool.crashlab.rules]`` in ``pyproject.toml``). Each rule compiles to a
:class:`~crashlab.models.CheckResult` and a ``@crashlab.check``-compatible
:class:`~crashlab.verify.oracles.CheckSpec`, so the same file works with
``crashlab.check``, ``crashlab.suite.eval_runs``, ``crashlab verify`` and the
``crashlab rules`` / ``crashlab check --rules`` / ``crashlab eval --rules`` commands.

PyYAML is optional (``crashlab[yaml]``). It is never imported at module import time;
when it is not installed a small built-in parser still reads simple YAML rule lists.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import SEVERITY_RANK, CheckResult, Run, Step
from .verify.oracles import CheckSpec

__all__ = [
    "RULE_TYPES",
    "Rule",
    "RuleError",
    "discover_rules",
    "discover_rules_path",
    "evaluate_rule",
    "evaluate_rules",
    "load_rules",
    "parse_rules",
    "parse_simple_yaml",
    "register_rules",
    "rules_to_checks",
    "validate_rules",
]


class RuleError(ValueError):
    """A rule file could not be read, or a rule failed schema validation."""


_RULE_TYPES = (
    "must_call",
    "must_not_call",
    "tool_order",
    "max_tool_calls",
    "max_cost_usd",
    "max_tokens",
    "max_duration_ms",
    "max_steps",
    "reply_matches",
    "reply_not_matches",
    "reply_contains",
    "reply_not_contains",
    "tool_arg_matches",
    "tool_arg_forbidden",
    "json_valid_reply",
    "no_findings_above",
)
RULE_TYPES: tuple[str, ...] = _RULE_TYPES
_TYPE_SET = set(_RULE_TYPES)
_TYPE_ALIASES = {t.replace("_", "-"): t for t in _RULE_TYPES}
_META = frozenset({"type", "id", "severity", "message", "agent", "name"})
_OFF = frozenset({"off", "none", "false", "0"})
_AUTO = frozenset({"auto", ""})
_SEVERITIES = ("info", "warn", "error", "critical")
_RULE_FILES = ("rules.json", "rules.toml", "rules.yaml", "rules.yml")
_OPS = ("<=", ">=", "==", "!=", "<", ">")
_OP_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*(<=|>=|==|!=|<|>)\s*(.+?)\s*$")
_JSON_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.S | re.I)
_JSON_TYPES = {
    "string": lambda v: isinstance(v, str),
    "str": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "list": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


@dataclass(frozen=True)
class Rule:
    """One declarative policy rule, already validated."""

    type: str
    id: str
    params: dict[str, Any] = field(default_factory=dict)
    agent: str | None = None
    name: str | None = None
    severity: str | None = None
    message: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "type": self.type, **self.params}
        if self.agent is not None:
            data["agent"] = self.agent
        if self.name is not None:
            data["name"] = self.name
        if self.severity is not None:
            data["severity"] = self.severity
        if self.message is not None:
            data["message"] = self.message
        return data


def _read_text(path: Path) -> str:
    try:
        return path.read_text("utf-8")
    except OSError as exc:
        raise RuleError(f"cannot read {path}: {exc}") from exc


def _toml_loads(text: str) -> Any:
    if sys.version_info >= (3, 11):
        import tomllib

        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise RuleError(f"invalid TOML: {exc}") from exc
    try:
        import tomli
    except ImportError:
        try:
            return _parse_simple_toml(text)
        except RuleError:
            raise
        except Exception as exc:
            raise RuleError(
                f"invalid TOML (install tomli or use Python 3.11+ / a JSON file): {exc}"
            ) from exc
    try:
        return tomli.loads(text)
    except tomli.TOMLDecodeError as exc:
        raise RuleError(f"invalid TOML: {exc}") from exc


def _yaml_loads(text: str) -> Any:
    try:
        import yaml  # never a top-level import: PyYAML is an optional extra
    except ImportError:
        yaml = None  # type: ignore[assignment]
    if yaml is not None:
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise RuleError(f"invalid YAML: {exc}") from exc
    try:
        return parse_simple_yaml(text)
    except RuleError:
        raise
    except Exception as exc:
        raise RuleError(
            "invalid YAML (install PyYAML for the full syntax: "
            f"pip install 'crashlab[yaml]'): {exc}"
        ) from exc


def _loads(path: Path, text: str) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuleError(f"{path}: invalid JSON: {exc}") from exc
    if suffix == ".toml":
        try:
            return _toml_loads(text)
        except RuleError as exc:
            raise RuleError(f"{path}: {exc}") from exc
    if suffix in {".yaml", ".yml"}:
        try:
            return _yaml_loads(text)
        except RuleError as exc:
            raise RuleError(f"{path}: {exc}") from exc
    errors: list[str] = []
    for loader, label in (
        (lambda t: json.loads(t), "JSON"),
        (_toml_loads, "TOML"),
        (_yaml_loads, "YAML"),
    ):
        try:
            return loader(text)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise RuleError(f"{path}: could not parse as JSON, TOML or YAML ({'; '.join(errors)})")


def _is_off(source: str | Path | None) -> bool:
    return isinstance(source, str) and source.strip().lower() in _OFF


def _is_auto(source: str | Path | None) -> bool:
    if source is None:
        return True
    return isinstance(source, str) and source.strip().lower() in _AUTO


def discover_rules_path(start: str | os.PathLike[str] | None = None) -> Path | None:
    """The first rules file at or above ``start`` (cwd): ``.crashlab/rules.*``, else
    a ``pyproject.toml`` that has a ``[tool.crashlab.rules]`` section."""
    here = Path(start if start is not None else os.getcwd()).resolve()
    for directory in (here, *here.parents):
        crashlab_dir = directory / ".crashlab"
        for name in _RULE_FILES:
            candidate = crashlab_dir / name
            if candidate.is_file():
                return candidate
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and _pyproject_rules_data(pyproject) is not None:
            return pyproject
    return None


def _pyproject_rules_data(path: Path) -> Any | None:
    try:
        data = _toml_loads(_read_text(path))
    except RuleError:
        return None
    if not isinstance(data, dict):
        return None
    section = ((data.get("tool") or {}).get("crashlab") or {}).get("rules")
    if section is None or section == {} or section == []:
        return None
    return section


def discover_rules(
    start: str | os.PathLike[str] | None = None,
) -> tuple[Path | None, list[Rule]]:
    """Auto-discover a rules file and parse it. ``(None, [])`` when none is present."""
    path = discover_rules_path(start)
    if path is None:
        return None, []
    return path, load_rules(path)


def load_rules(
    source: str | Path | Iterable[Any] | bool | None = "auto",
    *,
    start: str | os.PathLike[str] | None = None,
) -> list[Rule]:
    """Load and validate rules.

    ``None`` / ``"auto"`` discovers ``.crashlab/rules.*`` or ``[tool.crashlab.rules]``.
    ``"off"`` / ``"none"`` / ``False`` returns an empty list. A path is read as
    JSON, TOML or YAML. A list or mapping is validated in memory.
    """
    if source is False or _is_off(
        source if isinstance(source, (str, Path)) or source is None else None
    ):
        return []
    if isinstance(source, (list, tuple, dict)):
        return parse_rules(source, source="<memory>")
    if _is_auto(source if isinstance(source, (str, Path)) or source is None else "x"):
        _path, rules = discover_rules(start)
        return rules
    path = Path(str(source)).expanduser()
    if not path.is_file():
        raise RuleError(f"rules file not found: {path}")
    raw = _loads(path, _read_text(path))
    if path.name == "pyproject.toml":
        section = _pyproject_rules_data(path)
        if section is None:
            return []
        raw = section
    return parse_rules(raw, source=str(path))


def validate_rules(
    source: str | Path | Iterable[Any] | None,
    *,
    start: str | os.PathLike[str] | None = None,
) -> list[Rule]:
    """Parse and schema-validate a rules file (or in-memory payload). Same as
    :func:`load_rules`, but ``auto`` with no file raises :class:`RuleError`."""
    if _is_auto(source if isinstance(source, (str, Path)) or source is None else "x"):
        path, rules = discover_rules(start)
        if path is None:
            raise RuleError(
                "no rules file found (looked for .crashlab/rules.json, "
                ".crashlab/rules.toml, .crashlab/rules.yaml, .crashlab/rules.yml, "
                "and [tool.crashlab.rules] in pyproject.toml)"
            )
        return rules
    rules = load_rules(source, start=start)
    if isinstance(source, (str, Path)) and not _is_off(source) and not rules:
        raise RuleError(f"no rules in {source}")
    return rules


def parse_rules(data: Any, *, source: str = "<memory>") -> list[Rule]:
    """Validate a parsed document (list, ``{rules: [...]}``, or one rule mapping)."""
    items = _rule_items(data, source)
    rules: list[Rule] = []
    seen: set[str] = set()
    for i, item in enumerate(items, 1):
        loc = f"{source} rule {i}"
        if not isinstance(item, dict):
            raise RuleError(f"{loc}: expected a mapping, got {type(item).__name__}")
        rule = _parse_one(item, index=i, source=source)
        if rule.id in seen:
            raise RuleError(f"{loc}: duplicate rule id {rule.id!r}")
        seen.add(rule.id)
        rules.append(rule)
    if not rules:
        raise RuleError(f"{source}: no rules found")
    return rules


def _rule_items(data: Any, source: str) -> list[Any]:
    if data is None:
        raise RuleError(f"{source}: no rules found")
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise RuleError(f"{source}: expected a list or a mapping, got {type(data).__name__}")
    if "rules" in data:
        rules = data["rules"]
        if isinstance(rules, dict) and "type" in rules:
            return [rules]
        if not isinstance(rules, list):
            raise RuleError(f"{source}: 'rules' must be a list")
        return rules
    if "type" in data:
        return [data]
    raise RuleError(f"{source}: expected a 'rules' list or a rule with a 'type'")


def _as_type(value: Any, loc: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuleError(f"{loc}: 'type' must be a non-empty string")
    raw = value.strip()
    name = _TYPE_ALIASES.get(raw, raw.replace("-", "_"))
    name = _TYPE_ALIASES.get(name, name)
    if name not in _TYPE_SET:
        known = ", ".join(_RULE_TYPES)
        raise RuleError(f"{loc}: unknown rule type {value!r} (expected one of {known})")
    return name


def _optional_str(item: dict[str, Any], key: str, loc: str) -> str | None:
    if key not in item or item[key] is None:
        return None
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise RuleError(f"{loc}: {key!r} must be a non-empty string")
    return value.strip()


def _parse_one(item: dict[str, Any], *, index: int, source: str) -> Rule:
    loc = f"{source} rule {index}"
    rtype = _as_type(item.get("type"), loc)
    loc = f"{source} rule {index} ({rtype})"
    severity = _optional_str(item, "severity", loc)
    if severity is not None and severity not in _SEVERITIES:
        raise RuleError(
            f"{loc}: severity must be one of {', '.join(_SEVERITIES)}, not {severity!r}"
        )
    agent = _optional_str(item, "agent", loc)
    name = _optional_str(item, "name", loc)
    message = _optional_str(item, "message", loc)
    rid = _optional_str(item, "id", loc)
    params = {k: v for k, v in item.items() if k not in _META}
    _validate_params(rtype, params, loc, item)
    if rid is None:
        rid = _default_id(rtype, params, index)
    return Rule(
        type=rtype,
        id=rid,
        params=params,
        agent=agent,
        name=name,
        severity=severity,
        message=message,
        source=f"{source}:{index}",
    )


def _default_id(rtype: str, params: dict[str, Any], index: int) -> str:
    tools = _tool_names(params, required=False)
    if tools:
        return f"{rtype}:{','.join(tools)}"
    if rtype in {"max_cost_usd", "max_tokens", "max_duration_ms", "max_steps"}:
        cap = _limit_value(params, rtype, required=False)
        if cap is not None:
            return f"{rtype}:{cap}"
    return f"{rtype}#{index}"


def _tool_names(params: dict[str, Any], *, required: bool, loc: str = "rule") -> list[str]:
    raw = params.get("tools", params.get("tool", params.get("order")))
    if raw is None:
        if required:
            raise RuleError(f"{loc}: missing 'tool' / 'tools'")
        return []
    if isinstance(raw, str):
        names = [raw]
    elif isinstance(raw, dict):
        names = [str(k) for k in raw]
    elif isinstance(raw, (list, tuple)):
        if any(not isinstance(x, str) for x in raw):
            raise RuleError(f"{loc}: 'tools' must be a name or a list of names")
        names = [str(x) for x in raw]
    else:
        raise RuleError(f"{loc}: 'tool' / 'tools' must be a name or a list of names")
    names = [n.strip() for n in names if str(n).strip()]
    if required and not names:
        raise RuleError(f"{loc}: 'tool' / 'tools' must name at least one tool")
    return names


def _limit_value(params: dict[str, Any], key: str, *, required: bool, loc: str = "rule") -> Any:
    for candidate in ("max", "limit", "value", key):
        if candidate in params and params[candidate] is not None:
            return params[candidate]
    if required:
        raise RuleError(f"{loc}: missing 'max' (or {key})")
    return None


def _as_number(value: Any, loc: str, *, what: str, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuleError(f"{loc}: {what} must be a number, got {value!r}")
    if integer:
        if isinstance(value, float) and not value.is_integer():
            raise RuleError(f"{loc}: {what} must be an integer, got {value!r}")
        return int(value)
    return float(value) if not isinstance(value, int) else value


def _as_positive(value: Any, loc: str, *, what: str, integer: bool = False) -> float | int:
    number = _as_number(value, loc, what=what, integer=integer)
    if number < 0:
        raise RuleError(f"{loc}: {what} must be >= 0, got {number}")
    return number


def _compile_regex(pattern: str, loc: str, flags: int = 0) -> re.Pattern[str]:
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise RuleError(f"{loc}: invalid regex {pattern!r}: {exc}") from exc


def _regex_flags(params: dict[str, Any], loc: str, *, default: int = 0) -> int:
    flags = default
    if params.get("ignore_case") or params.get("case_insensitive") or params.get("casefold"):
        flags |= re.IGNORECASE
    raw = params.get("flags")
    if raw is None:
        return flags
    if isinstance(raw, int):
        return flags | raw
    names = raw if isinstance(raw, (list, tuple)) else str(raw).split("|")
    table = {
        "I": re.IGNORECASE,
        "IGNORECASE": re.IGNORECASE,
        "M": re.MULTILINE,
        "MULTILINE": re.MULTILINE,
        "S": re.DOTALL,
        "DOTALL": re.DOTALL,
        "X": re.VERBOSE,
        "VERBOSE": re.VERBOSE,
    }
    for name in names:
        key = str(name).strip().upper()
        if not key:
            continue
        if key not in table:
            raise RuleError(f"{loc}: unknown regex flag {name!r}")
        flags |= table[key]
    return flags


def _pattern_of(params: dict[str, Any], loc: str) -> str:
    for key in ("pattern", "regex", "match"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    raise RuleError(f"{loc}: missing 'pattern' (or 'regex')")


def _validate_params(rtype: str, params: dict[str, Any], loc: str, raw: dict[str, Any]) -> None:
    if rtype in {"must_call", "must_not_call", "tool_order"}:
        names = _tool_names(params, required=True, loc=loc)
        if rtype == "tool_order" and len(names) < 2:
            raise RuleError(f"{loc}: tool_order needs at least two tool names")
        if rtype == "must_call" and "min_calls" in params:
            _as_positive(params["min_calls"], loc, what="min_calls", integer=True)
    elif rtype == "max_tool_calls":
        cap = _limit_value(params, "max_tool_calls", required=False, loc=loc)
        tools = params.get("tools", params.get("tool"))
        if cap is not None:
            _as_positive(cap, loc, what="max", integer=True)
        if isinstance(tools, dict):
            for key, val in tools.items():
                _as_positive(val, loc, what=f"max for tool {key}", integer=True)
        elif tools is not None:
            _tool_names(params, required=True, loc=loc)
        if cap is None and not isinstance(tools, dict):
            raise RuleError(f"{loc}: missing 'max' (total or per-tool)")
    elif rtype in {"max_cost_usd", "max_tokens", "max_duration_ms", "max_steps"}:
        value = _limit_value(params, rtype, required=True, loc=loc)
        integer = rtype in {"max_tokens", "max_duration_ms", "max_steps"}
        _as_positive(value, loc, what="max", integer=integer)
    elif rtype in {"reply_matches", "reply_not_matches"}:
        _compile_regex(_pattern_of(params, loc), loc, _regex_flags(params, loc))
    elif rtype in {"reply_contains", "reply_not_contains"}:
        text = params.get("text", params.get("contains", params.get("value")))
        if not isinstance(text, str) or text == "":
            raise RuleError(f"{loc}: missing 'text' (or 'contains')")
    elif rtype in {"tool_arg_matches", "tool_arg_forbidden"}:
        _tool_names(params, required=True, loc=loc)
        if not _arg_constraints(params, loc, compile_regex=True):
            raise RuleError(
                f"{loc}: give an argument constraint (arg + pattern / op / min / max / "
                f"equals, a 'constraint' like 'amount <= 500', or an 'args' mapping)"
            )
    elif rtype == "json_valid_reply":
        keys = params.get("required_keys", params.get("required"))
        if keys is not None and not (
            isinstance(keys, (list, tuple)) and all(isinstance(k, str) for k in keys)
        ):
            raise RuleError(f"{loc}: 'required_keys' must be a list of strings")
        props = params.get("properties")
        if props is not None and not isinstance(props, dict):
            raise RuleError(f"{loc}: 'properties' must be a mapping of name -> type")
        if isinstance(props, dict):
            for key, spec in props.items():
                _property_type(spec, f"{loc} properties.{key}")
    elif rtype == "no_findings_above":
        threshold = _findings_threshold(params, raw, loc)
        if threshold not in SEVERITY_RANK:
            raise RuleError(f"{loc}: threshold must be warn, error or critical, not {threshold!r}")


def _findings_threshold(params: dict[str, Any], raw: dict[str, Any], loc: str) -> str:
    for key in ("threshold", "above", "min_severity", "max", "value"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = raw.get("severity")
    if isinstance(value, str) and value.strip() and "threshold" not in params:
        return value.strip()
    raise RuleError(f"{loc}: missing 'threshold' (warn, error or critical)")


def _property_type(spec: Any, loc: str) -> str:
    if isinstance(spec, str):
        name = spec.strip().lower()
        if name not in _JSON_TYPES:
            raise RuleError(
                f"{loc}: unknown type {spec!r} (string, number, integer, boolean, "
                "object, array, null)"
            )
        return name
    if isinstance(spec, dict):
        typ = spec.get("type")
        if typ is None:
            raise RuleError(f"{loc}: missing 'type'")
        return _property_type(typ, loc)
    raise RuleError(f"{loc}: type must be a string or {{type: ...}}")


@dataclass(frozen=True)
class _ArgConstraint:
    key: str
    kind: str  # regex, op, equals, contains, exists, in
    value: Any = None
    op: str | None = None
    pattern: re.Pattern[str] | None = None


_ARG_SKIP = frozenset(
    {
        "tools",
        "tool",
        "order",
        "arg",
        "key",
        "args",
        "constraint",
        "expr",
        "where",
        "agent",
        "name",
        "min_calls",
    }
)


def _arg_constraints(
    params: dict[str, Any], loc: str, *, compile_regex: bool = False
) -> list[_ArgConstraint]:
    out: list[_ArgConstraint] = []
    expr = params.get("constraint") or params.get("expr") or params.get("where")
    if expr:
        out.append(_parse_expr(str(expr), loc))
    args = params.get("args")
    if isinstance(args, dict):
        for key, spec in args.items():
            out.extend(_constraints_for(str(key), spec, loc, compile_regex=compile_regex))
    elif isinstance(args, list):
        for i, spec in enumerate(args):
            if isinstance(spec, str):
                out.append(_parse_expr(spec, f"{loc} args[{i}]"))
            elif isinstance(spec, dict):
                key = spec.get("arg") or spec.get("key") or spec.get("name")
                if not key:
                    raise RuleError(f"{loc} args[{i}]: missing 'arg'")
                out.extend(
                    _constraints_for(
                        str(key), spec, f"{loc} args[{i}]", compile_regex=compile_regex
                    )
                )
    key = params.get("arg") or params.get("key")
    if key:
        spec = {k: v for k, v in params.items() if k not in _ARG_SKIP}
        if not spec:
            spec = {"exists": True}
        out.extend(_constraints_for(str(key), spec, loc, compile_regex=compile_regex))
    return out


def _parse_expr(expr: str, loc: str) -> _ArgConstraint:
    m = _OP_RE.match(expr)
    if not m:
        raise RuleError(f"{loc}: cannot parse constraint {expr!r} (expected e.g. 'amount <= 500')")
    key, op, raw = m.group(1), m.group(2), m.group(3)
    return _ArgConstraint(key=key, kind="op", op=op, value=_literal(raw))


def _literal(raw: str) -> Any:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if any(ch in text for ch in ".eE") and _is_float(text):
            return float(text)
        return int(text, 10)
    except ValueError:
        return text


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _constraints_for(key: str, spec: Any, loc: str, *, compile_regex: bool) -> list[_ArgConstraint]:
    if isinstance(spec, str):
        m = _OP_RE.match(spec)
        if m:
            return [_ArgConstraint(key=m.group(1), kind="op", op=m.group(2),
                                   value=_literal(m.group(3)))]  # fmt: skip
        # "<= 500" relative to this key
        rel = re.match(r"^\s*(<=|>=|==|!=|<|>)\s*(.+?)\s*$", spec)
        if rel:
            return [_ArgConstraint(key=key, kind="op", op=rel.group(1),
                                   value=_literal(rel.group(2)))]  # fmt: skip
        if compile_regex:
            return [_ArgConstraint(key=key, kind="regex",
                                   pattern=_compile_regex(spec, loc), value=spec)]  # fmt: skip
        return [_ArgConstraint(key=key, kind="regex", value=spec)]
    if isinstance(spec, (bool, int, float)):
        return [_ArgConstraint(key=key, kind="equals", value=spec)]
    if not isinstance(spec, dict):
        raise RuleError(f"{loc}: constraint for {key} must be a mapping, string or number")
    out: list[_ArgConstraint] = []
    if spec.get("constraint") or spec.get("expr"):
        out.append(_parse_expr(str(spec.get("constraint") or spec.get("expr")), loc))
    if any(k in spec for k in ("pattern", "regex", "match")):
        pattern = spec.get("pattern") or spec.get("regex") or spec.get("match")
        flags = _regex_flags(spec, loc)
        compiled = _compile_regex(str(pattern), loc, flags) if compile_regex else None
        out.append(_ArgConstraint(key=key, kind="regex", value=str(pattern), pattern=compiled))
    if "equals" in spec or "eq" in spec:
        out.append(_ArgConstraint(key=key, kind="equals",
                                  value=spec.get("equals", spec.get("eq"))))  # fmt: skip
    if "contains" in spec or ("text" in spec and "pattern" not in spec):
        out.append(_ArgConstraint(key=key, kind="contains",
                                  value=spec.get("contains", spec.get("text"))))  # fmt: skip
    if "in" in spec or "one_of" in spec:
        values = spec.get("in", spec.get("one_of"))
        if not isinstance(values, (list, tuple)):
            raise RuleError(f"{loc}: 'in' must be a list")
        out.append(_ArgConstraint(key=key, kind="in", value=list(values)))
    if "exists" in spec:
        out.append(_ArgConstraint(key=key, kind="exists", value=bool(spec["exists"])))
    if spec.get("op") is not None:
        op = str(spec["op"])
        if op not in _OPS:
            raise RuleError(f"{loc}: unknown op {op!r}")
        if "value" not in spec and "rhs" not in spec:
            raise RuleError(f"{loc}: op {op!r} needs a 'value'")
        out.append(_ArgConstraint(key=key, kind="op", op=op,
                                  value=spec.get("value", spec.get("rhs"))))  # fmt: skip
    for name, op_name in (
        ("min", ">="),
        ("gte", ">="),
        ("gt", ">"),
        ("max", "<="),
        ("lte", "<="),
        ("lt", "<"),
    ):
        if name in spec and spec[name] is not None:
            out.append(_ArgConstraint(key=key, kind="op", op=op_name, value=spec[name]))
    uniq: list[_ArgConstraint] = []
    seen: set[tuple[Any, ...]] = set()
    for c in out:
        sig = (c.key, c.kind, c.op, repr(c.value))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(c)
    if not uniq:
        uniq.append(_ArgConstraint(key=key, kind="exists", value=True))
    return uniq


def _applies(rule: Rule, run: Run) -> bool:
    return rule.name is None or run.name == rule.name


def _tools(run: Run, *, agent: str | None = None, name: str | None = None) -> list[Step]:
    return run.tool_calls(name, agent=agent)


def _counts(run: Run, *, agent: str | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in _tools(run, agent=agent):
        counts[step.name] = counts.get(step.name, 0) + 1
    return counts


def _first_idx(run: Run, tool: str, *, agent: str | None) -> int | None:
    for step in _tools(run, agent=agent, name=tool):
        return step.idx
    return None


def _final_reply(run: Run) -> str:
    """The concluding assistant text: ``run.output`` when it is a string, else the last
    chat step that replied with text and no tool call. Non-string outputs are JSON."""
    if isinstance(run.output, str) and run.output.strip():
        return run.output
    if run.output is not None and not isinstance(run.output, (bytes, bytearray)):
        try:
            return json.dumps(run.output, ensure_ascii=False, default=str)
        except TypeError:
            return str(run.output)
    for step in reversed([s for s in run.steps if s.operation == "chat"]):
        if step.error is not None:
            continue
        if (step.response or {}).get("tool_calls"):
            continue
        text = _chat_text(step)
        if text.strip():
            return text
    return ""


def _chat_text(step: Step) -> str:
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
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                elif isinstance(item, str):
                    parts.append(item)
    if not parts and isinstance(response.get("text"), str):
        parts.append(response["text"])
    return "\n".join(parts)


def _lookup(args: Any, key: str) -> tuple[bool, Any]:
    if not isinstance(args, dict):
        return False, None
    if key in args:
        return True, args[key]
    cur: Any = args
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def _num_eq(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return False


def _cmp(op: str, left: Any, right: Any) -> bool:
    try:
        if op == "<=":
            return left <= right
        if op == "<":
            return left < right
        if op == ">=":
            return left >= right
        if op == ">":
            return left > right
        if op == "==":
            return left == right or _num_eq(left, right)
        if op == "!=":
            return not (left == right or _num_eq(left, right))
    except TypeError:
        return False
    return False


def _maybe_numeric(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        try:
            return int(value, 10) if re.fullmatch(r"-?\d+", value) else float(value)
        except ValueError:
            return value
    return value


def _constraint_ok(constraint: _ArgConstraint, args: Any) -> tuple[bool, str]:
    present, value = _lookup(args, constraint.key)
    if constraint.kind == "exists":
        ok = present if constraint.value else not present
        if ok:
            return True, ""
        want = "present" if constraint.value else "absent"
        return False, f"{constraint.key} should be {want}"
    if not present:
        return False, f"missing argument {constraint.key}"
    if constraint.kind == "regex":
        pattern = constraint.pattern or re.compile(str(constraint.value))
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        if pattern.search(text):
            return True, ""
        return False, f"{constraint.key}={value!r} does not match /{pattern.pattern}/"
    if constraint.kind == "equals":
        if value == constraint.value or _num_eq(value, constraint.value):
            return True, ""
        return False, f"{constraint.key}={value!r} != {constraint.value!r}"
    if constraint.kind == "contains":
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        needle = str(constraint.value)
        if needle.lower() in text.lower():
            return True, ""
        return False, f"{constraint.key}={value!r} does not contain {needle!r}"
    if constraint.kind == "in":
        if value in constraint.value or any(_num_eq(value, x) for x in constraint.value):
            return True, ""
        return False, f"{constraint.key}={value!r} not in {constraint.value!r}"
    if constraint.kind == "op" and constraint.op:
        left, rhs = _maybe_numeric(value), _maybe_numeric(constraint.value)
        if _cmp(constraint.op, left, rhs):
            return True, ""
        return False, (
            f"{constraint.key}={value!r} does not satisfy "
            f"{constraint.key} {constraint.op} {constraint.value!r}"
        )
    return False, f"unknown constraint on {constraint.key}"


def _fail(rule: Rule, detail: str) -> CheckResult:
    message = rule.message or detail
    if rule.message and detail and detail not in rule.message:
        message = f"{rule.message} ({detail})"
    return CheckResult(name=rule.id, status="fail", message=message)


def _pass(rule: Rule) -> CheckResult:
    return CheckResult(name=rule.id, status="pass")


def _error(rule: Rule, detail: str) -> CheckResult:
    return CheckResult(name=rule.id, status="error", message=detail)


def evaluate_rule(rule: Rule, run: Run) -> CheckResult:
    """Grade one rule against ``run``. Skipped name filters pass."""
    if not _applies(rule, run):
        return _pass(rule)
    try:
        failed = _evaluate(rule, run)
    except Exception as exc:
        return _error(rule, f"{type(exc).__name__}: {exc}")
    if failed is None:
        return _pass(rule)
    return _fail(rule, failed)


def evaluate_rules(
    run: Run,
    rules: Sequence[Rule] | None = None,
    *,
    start: str | os.PathLike[str] | None = None,
) -> list[CheckResult]:
    """Evaluate ``rules`` (or auto-discovered rules) and return :class:`CheckResult`s."""
    loaded: Sequence[Rule] = load_rules("auto", start=start) if rules is None else rules
    return [evaluate_rule(rule, run) for rule in loaded]


def rules_to_checks(rules: Sequence[Rule]) -> list[CheckSpec]:
    """Wrap each rule as a :class:`~crashlab.verify.oracles.CheckSpec` / CheckFn.

    The specs compose with :func:`crashlab.verify.run_checks`,
    :func:`crashlab.suite.eval_runs` and ``crashlab check``. ``CheckSpec.agent``
    is the optional run-name filter (``rule.name``).
    """
    specs: list[CheckSpec] = []
    for rule in rules:

        def _fn(run: Run, *, _rule: Rule = rule) -> Any:
            result = evaluate_rule(_rule, run)
            if result.status == "pass":
                return True
            if result.status == "error":
                raise RuntimeError(result.message or "rule error")
            return result.message or False

        _fn.__name__ = re.sub(r"[^A-Za-z0-9_]+", "_", rule.id)
        _fn.__doc__ = rule.message or f"policy rule {rule.type} ({rule.id})"
        specs.append(CheckSpec(name=rule.id, agent=rule.name, fn=_fn))
    return specs


def register_rules(rules: Sequence[Rule]) -> list[CheckSpec]:
    """Register rule wrappers in the ``@crashlab.check`` registry (for ``verify``)."""
    from .verify.oracles import check

    specs = rules_to_checks(rules)
    for spec in specs:
        check(spec.agent, name=spec.name)(spec.fn)
    return specs


def _evaluate(rule: Rule, run: Run) -> str | None:
    typ, params, agent = rule.type, rule.params, rule.agent
    if typ == "must_call":
        min_calls = int(params.get("min_calls", 1) or 1)
        missing: list[str] = []
        for tool in _tool_names(params, required=True):
            n = sum(1 for _ in _tools(run, agent=agent, name=tool))
            if n < min_calls:
                missing.append(f"{tool} called {n} time(s) (min {min_calls})")
        return ("must_call: " + "; ".join(missing)) if missing else None
    if typ == "must_not_call":
        hits: list[str] = []
        for tool in _tool_names(params, required=True):
            steps = _tools(run, agent=agent, name=tool)
            if steps:
                idxs = ", ".join(str(s.idx) for s in steps[:6])
                hits.append(f"{tool} at step {idxs}")
        return ("must_not_call: forbidden tool " + "; ".join(hits)) if hits else None
    if typ == "tool_order":
        names = _tool_names(params, required=True)
        prev_name, prev_idx = names[0], _first_idx(run, names[0], agent=agent)
        if prev_idx is None:
            return f"tool_order: {names[0]!r} was not called"
        for name in names[1:]:
            idx = _first_idx(run, name, agent=agent)
            if idx is None:
                return f"tool_order: {name!r} was not called"
            if idx < prev_idx:
                return (
                    f"tool_order: {prev_name!r} must appear before {name!r} "
                    f"(first {prev_name} at step {prev_idx}, first {name} at step {idx})"
                )
            prev_name, prev_idx = name, idx
        return None
    if typ == "max_tool_calls":
        return _eval_max_tool_calls(rule, run)
    if typ == "max_cost_usd":
        cap = float(_limit_value(params, typ, required=True))
        cost = _cost_usd(run, agent=agent)
        if cost is None:
            return "max_cost_usd: cost is unknown (no priced usage on this run)"
        if cost > cap:
            return f"max_cost_usd: ${cost:.4f} > max ${cap:.4f}"
        return None
    if typ == "max_tokens":
        cap = int(_limit_value(params, typ, required=True))
        tokens = _token_count(run, agent=agent)
        if tokens > cap:
            return f"max_tokens: {tokens} > max {cap}"
        return None
    if typ == "max_duration_ms":
        cap = int(_limit_value(params, typ, required=True))
        duration = _duration_ms(run)
        if duration is None:
            return "max_duration_ms: duration is unknown (the run has no wall time)"
        if duration > cap:
            return f"max_duration_ms: {duration:g}ms > max {cap}ms"
        return None
    if typ == "max_steps":
        cap = int(_limit_value(params, typ, required=True))
        n = len(run.steps) if run.steps else int(run.usage.steps or 0)
        if n > cap:
            return f"max_steps: {n} steps > max {cap}"
        return None
    if typ in {"reply_matches", "reply_not_matches"}:
        pattern = _compile_regex(
            _pattern_of(params, rule.id), rule.id, _regex_flags(params, rule.id)
        )
        hit = pattern.search(_final_reply(run)) is not None
        if typ == "reply_matches" and not hit:
            return f"reply_matches: final reply did not match /{pattern.pattern}/"
        if typ == "reply_not_matches" and hit:
            return f"reply_not_matches: final reply matched /{pattern.pattern}/"
        return None
    if typ in {"reply_contains", "reply_not_contains"}:
        needle = str(params.get("text", params.get("contains", params.get("value"))))
        reply = _final_reply(run)
        ignore = params.get("ignore_case", params.get("case_insensitive"))
        if ignore is None:
            ignore = True
        hay, pin = (reply.lower(), needle.lower()) if ignore else (reply, needle)
        hit = pin in hay
        if typ == "reply_contains" and not hit:
            return f"reply_contains: final reply does not contain {needle!r}"
        if typ == "reply_not_contains" and hit:
            return f"reply_not_contains: final reply contains {needle!r}"
        return None
    if typ in {"tool_arg_matches", "tool_arg_forbidden"}:
        return _eval_tool_args(rule, run)
    if typ == "json_valid_reply":
        return _eval_json_reply(rule, run)
    if typ == "no_findings_above":
        threshold = _findings_threshold(params, {"severity": rule.severity, **params}, rule.id)
        if threshold not in SEVERITY_RANK:
            return f"no_findings_above: unknown threshold {threshold!r}"
        floor = SEVERITY_RANK[threshold]
        hits = [f for f in run.findings if f.rank >= floor]
        if not hits:
            return None
        shown = ", ".join(
            f"{f.kind}:{f.severity}" + (f" at step {f.step}" if f.step is not None else "")
            for f in hits[:6]
        )
        return f"no_findings_above: {len(hits)} finding(s) at or above {threshold} ({shown})"
    return f"unknown rule type {typ}"


def _eval_max_tool_calls(rule: Rule, run: Run) -> str | None:
    params, agent = rule.params, rule.agent
    counts = _counts(run, agent=agent)
    total = sum(counts.values())
    problems: list[str] = []
    cap = _limit_value(params, "max_tool_calls", required=False)
    tools = params.get("tools", params.get("tool"))
    names = _tool_names(params, required=False) if not isinstance(tools, dict) else []
    # A scalar max with one named tool is a per-tool cap; otherwise it is a total.
    if cap is not None:
        limit = int(cap)
        if isinstance(tools, str) or (len(names) == 1 and not isinstance(tools, (list, tuple))):
            name = names[0] if names else str(tools)
            n = counts.get(name, 0)
            if n > limit:
                problems.append(f"{name} called {n} time(s) > max {limit}")
        elif len(names) == 1 and "tool" in params and "tools" not in params:
            name = names[0]
            n = counts.get(name, 0)
            if n > limit:
                problems.append(f"{name} called {n} time(s) > max {limit}")
        else:
            if total > limit:
                problems.append(f"{total} tool calls > max {limit}")
    if isinstance(tools, dict):
        for name, limit in tools.items():
            n = counts.get(str(name), 0)
            if n > int(limit):
                problems.append(f"{name} called {n} time(s) > max {int(limit)}")
    if problems:
        return "max_tool_calls: " + "; ".join(problems)
    return None


def _eval_tool_args(rule: Rule, run: Run) -> str | None:
    names = _tool_names(rule.params, required=True)
    constraints = _arg_constraints(rule.params, rule.id, compile_regex=True)
    steps: list[Step] = []
    for name in names:
        steps.extend(_tools(run, agent=rule.agent, name=name))
    if rule.type == "tool_arg_matches" and not steps:
        return f"tool_arg_matches: tool {', '.join(names)} was not called"
    problems: list[str] = []
    for step in steps:
        args = step.args if isinstance(step.args, dict) else {}
        for constraint in constraints:
            ok, detail = _constraint_ok(constraint, args)
            if rule.type == "tool_arg_matches" and not ok:
                problems.append(f"{step.name} at step {step.idx}: {detail}")
            elif rule.type == "tool_arg_forbidden" and ok:
                shown = args.get(constraint.key, args)
                problems.append(
                    f"{step.name} at step {step.idx}: {constraint.key}={shown!r} is forbidden"
                )
    if problems:
        return rule.type + ": " + "; ".join(problems[:8])
    return None


def _eval_json_reply(rule: Rule, run: Run) -> str | None:
    payload, err = _parse_reply_json(run)
    if err:
        return f"json_valid_reply: {err}"
    required = list(rule.params.get("required_keys") or rule.params.get("required") or [])
    if required:
        if not isinstance(payload, dict):
            return "json_valid_reply: reply JSON is not an object, cannot check required_keys"
        missing = [k for k in required if k not in payload]
        if missing:
            label = "s " if len(missing) > 1 else " "
            return (
                "json_valid_reply: missing required key"
                + label
                + ", ".join(repr(k) for k in missing)
            )
    props = rule.params.get("properties") or {}
    if props:
        if not isinstance(payload, dict):
            return "json_valid_reply: reply JSON is not an object, cannot check properties"
        bad: list[str] = []
        for key, spec in props.items():
            if key not in payload:
                continue
            typ = _property_type(spec, key)
            if not _JSON_TYPES[typ](payload[key]):
                got = type(payload[key]).__name__
                bad.append(f"{key!r} should be {typ}, got {got}")
        if bad:
            return "json_valid_reply: " + "; ".join(bad)
    return None


def _parse_reply_json(run: Run) -> tuple[Any, str | None]:
    if isinstance(run.output, (dict, list)):
        return run.output, None
    if (
        run.output is not None
        and not isinstance(run.output, str)
        and (isinstance(run.output, (int, float, bool)))
    ):
        return run.output, None
    text = _final_reply(run).strip()
    if not text:
        return None, "final reply is empty"
    candidates = [text]
    m = _JSON_FENCE.search(text)
    if m:
        candidates.insert(0, m.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            snippet = text[start : end + 1]
            if snippet not in candidates:
                candidates.append(snippet)
    last = "final reply is not valid JSON"
    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as exc:
            last = f"final reply is not valid JSON: {exc}"
    return None, last


def _cost_usd(run: Run, *, agent: str | None) -> float | None:
    from .budgets import summarize_usage

    summary = summarize_usage(run, agent=agent)
    if summary.cost_usd is not None:
        return summary.cost_usd
    return run.usage.cost_usd if agent is None else summary.cost_usd


def _token_count(run: Run, *, agent: str | None) -> int:
    from .budgets import summarize_usage

    summary = summarize_usage(run, agent=agent)
    if summary.total_tokens:
        return summary.total_tokens
    if agent is None:
        return int(run.usage.total_tokens or 0)
    return int(summary.total_tokens or 0)


def _duration_ms(run: Run) -> float | None:
    if run.usage.wall_time_ms is not None:
        return float(run.usage.wall_time_ms)
    if run.ended_at is not None:
        try:
            return (run.ended_at - run.started_at).total_seconds() * 1000.0
        except TypeError:
            return None
    return None


def parse_simple_yaml(text: str) -> Any:
    """A small YAML subset for rule files: mappings, lists, scalars, inline
    ``[]`` / ``{}``, comments, and ``---``. Raises :class:`RuleError` otherwise."""
    raw_lines = text.splitlines()
    cleaned: list[tuple[int, int, str]] = []
    for no, line in enumerate(raw_lines, 1):
        if line.startswith("\ufeff"):
            line = line[1:]
        if not cleaned and line.strip() in {"---", "..."}:
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line.split("#")[0]:
            raise RuleError(f"line {no}: tabs are not allowed in the built-in YAML parser")
        indent = len(line) - len(line.lstrip(" "))
        cleaned.append((no, indent, line))
    if not cleaned:
        return None
    value, next_i = _yaml_block(cleaned, 0, cleaned[0][1])
    if next_i != len(cleaned):
        raise RuleError(f"line {cleaned[next_i][0]}: unexpected content after the document")
    return value


def _yaml_block(lines: list[tuple[int, int, str]], i: int, indent: int) -> tuple[Any, int]:
    if i >= len(lines):
        return None, i
    _no, ind, raw = lines[i]
    if ind < indent:
        return None, i
    body = raw[ind:]
    if body.startswith("- ") or body == "-":
        return _yaml_list(lines, i, indent)
    return _yaml_map(lines, i, indent)


def _yaml_list(lines: list[tuple[int, int, str]], i: int, indent: int) -> tuple[list[Any], int]:
    out: list[Any] = []
    while i < len(lines):
        no, ind, raw = lines[i]
        if ind < indent:
            break
        if ind > indent:
            raise RuleError(f"line {no}: unexpected indent")
        body = raw[ind:]
        if not (body.startswith("- ") or body == "-"):
            break
        rest = body[1:].lstrip()
        child_indent = ind + 2
        if not rest:
            if i + 1 < len(lines) and lines[i + 1][1] >= child_indent:
                value, i = _yaml_block(lines, i + 1, child_indent)
            else:
                value, i = None, i + 1
            out.append(value)
            continue
        if rest.startswith("- ") or (
            ":" in _strip_comment(rest) and not rest.startswith("{") and _looks_like_key(rest)
        ):
            key, _, remainder = _split_key(rest)
            entry: dict[str, Any] = {}
            if remainder == "":
                if i + 1 < len(lines) and lines[i + 1][1] > ind:
                    value, i = _yaml_block(lines, i + 1, lines[i + 1][1])
                else:
                    value, i = None, i + 1
                entry[key] = value
            else:
                entry[key] = _yaml_scalar(remainder, no)
                i += 1
            while i < len(lines) and lines[i][1] > ind and not lines[i][2].lstrip().startswith("-"):
                more, i = _yaml_map(lines, i, lines[i][1])
                if not isinstance(more, dict):
                    raise RuleError(f"line {no}: expected a mapping in a list item")
                entry.update(more)
            out.append(entry)
            continue
        out.append(_yaml_scalar(rest, no))
        i += 1
    return out, i


def _yaml_map(lines: list[tuple[int, int, str]], i: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while i < len(lines):
        no, ind, raw = lines[i]
        if ind < indent:
            break
        if ind > indent:
            raise RuleError(f"line {no}: unexpected indent")
        body = raw[ind:]
        if body.startswith("- ") or body == "-":
            break
        key, sep, remainder = _split_key(body)
        if sep is None:
            raise RuleError(f"line {no}: expected 'key: value'")
        if remainder == "":
            if i + 1 < len(lines) and lines[i + 1][1] > ind:
                value, i = _yaml_block(lines, i + 1, lines[i + 1][1])
            else:
                value, i = None, i + 1
            out[key] = value
        else:
            out[key] = _yaml_scalar(remainder, no)
            i += 1
    return out, i


def _looks_like_key(text: str) -> bool:
    left = _strip_comment(text).split(":", 1)[0]
    if not left or left[0] in "[{":
        return False
    if (left.startswith('"') and left.endswith('"')) or (
        left.startswith("'") and left.endswith("'")
    ):
        return True
    return all(c not in left for c in "[]{},") and not left.startswith("- ")


def _split_key(text: str) -> tuple[str, str | None, str]:
    in_s = in_d = False
    for i, ch in enumerate(text):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            break
        elif ch == ":" and not in_s and not in_d:
            rest = text[i + 1 :]
            if rest == "" or rest[0].isspace():
                return _unquote(text[:i].strip()), ":", rest.strip()
    return text, None, ""


def _strip_comment(text: str) -> str:
    in_s = in_d = False
    for i, ch in enumerate(text):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            return text[:i].rstrip()
    return text


def _yaml_scalar(text: str, no: int) -> Any:
    text = _strip_comment(text).strip()
    if text == "" or text in {"~", "null", "Null", "NULL"}:
        return None
    if text in {"true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"}:
        return True
    if text in {"false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"}:
        return False
    if text and text[0] in "[{":
        try:
            return _yaml_inline(text, no)
        except RuleError:
            raise
        except Exception as exc:
            raise RuleError(f"line {no}: invalid inline YAML: {exc}") from exc
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return _unquote(text)
    if re.fullmatch(r"-?0[xX][0-9a-fA-F]+", text):
        return int(text, 16)
    if re.fullmatch(r"-?\d+", text):
        return int(text, 10)
    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", text):
        return float(text)
    return _unquote(text)


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1]
        if text[0] == '"':
            return bytes(inner, "utf-8").decode("unicode_escape") if "\\" in inner else inner
        return inner.replace("''", "'")
    return text


def _yaml_inline(text: str, no: int) -> Any:
    try:
        value, end = _inline_value(text, 0)
    except ValueError as exc:
        raise RuleError(f"line {no}: {exc}") from exc
    rest = text[end:].strip()
    if rest:
        raise RuleError(f"line {no}: trailing {rest!r} after inline value")
    return value


def _inline_value(text: str, i: int) -> tuple[Any, int]:
    i = _skip_ws(text, i)
    if i >= len(text):
        raise ValueError("unexpected end of inline YAML")
    if text[i] == "[":
        return _inline_list(text, i)
    if text[i] == "{":
        return _inline_map(text, i)
    if text[i] in "\"'":
        return _inline_quoted(text, i)
    start = i
    while i < len(text) and text[i] not in ",]}:" and not text[i].isspace():
        i += 1
    j = i
    while j < len(text) and text[j] not in ",]}:{[":
        j += 1
    if j > i:
        mid = text[i:j]
        if mid.strip() and not mid.strip().startswith("#"):
            i = j
    token = text[start:i].strip()
    return _yaml_scalar(token, 0), i


def _skip_ws(text: str, i: int) -> int:
    while i < len(text):
        if text[i] in " \t":
            i += 1
            continue
        if text[i] == "#":
            return len(text)
        break
    return i


def _inline_list(text: str, i: int) -> tuple[list[Any], int]:
    i += 1
    out: list[Any] = []
    i = _skip_ws(text, i)
    if i < len(text) and text[i] == "]":
        return out, i + 1
    while i < len(text):
        value, i = _inline_value(text, i)
        out.append(value)
        i = _skip_ws(text, i)
        if i < len(text) and text[i] == ",":
            i = _skip_ws(text, i + 1)
            if i < len(text) and text[i] == "]":
                return out, i + 1
            continue
        if i < len(text) and text[i] == "]":
            return out, i + 1
        raise ValueError("expected ',' or ']' in inline list")
    raise ValueError("unterminated inline list")


def _inline_map(text: str, i: int) -> tuple[dict[str, Any], int]:
    i += 1
    out: dict[str, Any] = {}
    i = _skip_ws(text, i)
    if i < len(text) and text[i] == "}":
        return out, i + 1
    while i < len(text):
        key_val, i = _inline_value(text, i)
        i = _skip_ws(text, i)
        if i >= len(text) or text[i] != ":":
            raise ValueError("expected ':' in inline mapping")
        value, i = _inline_value(text, i + 1)
        out[str(key_val)] = value
        i = _skip_ws(text, i)
        if i < len(text) and text[i] == ",":
            i = _skip_ws(text, i + 1)
            if i < len(text) and text[i] == "}":
                return out, i + 1
            continue
        if i < len(text) and text[i] == "}":
            return out, i + 1
        raise ValueError("expected ',' or '}' in inline mapping")
    raise ValueError("unterminated inline mapping")


def _inline_quoted(text: str, i: int) -> tuple[str, int]:
    quote = text[i]
    i += 1
    chars: list[str] = []
    while i < len(text):
        ch = text[i]
        if quote == "'" and ch == "'" and i + 1 < len(text) and text[i + 1] == "'":
            chars.append("'")
            i += 2
            continue
        if ch == quote:
            return _unquote(quote + "".join(chars) + quote), i + 1
        if ch == "\\" and quote == '"':
            if i + 1 >= len(text):
                raise ValueError("unterminated escape")
            chars.append("\\" + text[i + 1])
            i += 2
            continue
        chars.append(ch)
        i += 1
    raise ValueError("unterminated quoted string")


def _parse_simple_toml(text: str) -> dict[str, Any]:
    """Minimal TOML: ``key = value``, ``[[rules]]`` arrays of tables, inline arrays
    and inline tables. Enough for a standalone rules file."""
    root: dict[str, Any] = {}
    current: dict[str, Any] = root
    for no, raw in enumerate(text.splitlines(), 1):
        line = _toml_strip(raw)
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            path = [p.strip() for p in line[2:-2].split(".") if p.strip()]
            parent = _toml_ensure(root, path[:-1])
            key = path[-1]
            bucket = parent.setdefault(key, [])
            if not isinstance(bucket, list):
                raise RuleError(f"line {no}: {'.'.join(path)} is not an array")
            current = {}
            bucket.append(current)
            continue
        if line.startswith("[") and line.endswith("]"):
            path = [p.strip() for p in line[1:-1].split(".") if p.strip()]
            current = _toml_ensure(root, path)
            continue
        if "=" not in line:
            raise RuleError(f"line {no}: expected key = value")
        key, _, rest = line.partition("=")
        current[key.strip()] = _toml_value(rest.strip(), no)
    return root


def _toml_strip(raw: str) -> str:
    in_s = in_d = False
    for i, ch in enumerate(raw):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            return raw[:i].strip()
    return raw.strip()


def _toml_ensure(root: dict[str, Any], path: list[str]) -> dict[str, Any]:
    cur: dict[str, Any] = root
    for part in path:
        nxt = cur.get(part)
        if nxt is None:
            nxt = {}
            cur[part] = nxt
        elif isinstance(nxt, list) and nxt and isinstance(nxt[-1], dict):
            nxt = nxt[-1]
        elif not isinstance(nxt, dict):
            raise RuleError(f"TOML path {'.'.join(path)} conflicts with an existing value")
        cur = nxt
    return cur


def _toml_value(text: str, no: int) -> Any:
    if text.startswith("[") or text.startswith("{"):
        return _yaml_inline(text, no)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return _unquote(text)
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text, 10)
    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", text):
        return float(text)
    raise RuleError(f"line {no}: cannot parse TOML value {text!r}")
