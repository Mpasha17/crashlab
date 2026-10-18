"""A small JSON-schema validator for tool-call arguments (no dependency).

Covers what tool schemas use in practice: type (incl. lists of types and nullable),
properties, required, additionalProperties, enum, const, items, min/max length / items,
minimum / maximum, anyOf / oneOf / allOf (loosely) and local ``$ref`` to ``$defs``.
Unknown keywords are ignored, so it never rejects arguments it does not understand.
"""

from __future__ import annotations

from typing import Any

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: (
        (isinstance(v, int) and not isinstance(v, bool))
        or (isinstance(v, float) and v.is_integer())
    ),
}


def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def validate(value: Any, schema: Any, path: str = "", root: Any = None) -> list[str]:
    """Return human-readable problems (empty list: valid)."""
    if not isinstance(schema, dict):
        return []
    root = schema if root is None else root
    where = path or "arguments"
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        target: Any = root
        for part in ref[2:].split("/"):
            target = target.get(part) if isinstance(target, dict) else None
        return validate(value, target, path, root)
    errors: list[str] = []
    types = schema.get("type")
    if types is not None:
        tlist = types if isinstance(types, list) else [types]
        if schema.get("nullable"):
            tlist = [*tlist, "null"]
        known = [t for t in tlist if t in _TYPES]
        if known and not any(_TYPES[t](value) for t in known):
            want = " or ".join(known)
            return [f"{where}: expected {want}, got {_type_name(value)} {_short(value)}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{where}: must be {schema['const']!r}, got {_short(value)}")
    if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
        errors.append(f"{where}: {_short(value)} is not one of {schema['enum']}")
    for key in ("allOf",):
        for sub in schema.get(key) or []:
            errors += validate(value, sub, path, root)
    for key in ("anyOf", "oneOf"):
        subs = schema.get(key)
        if isinstance(subs, list) and subs and all(validate(value, s, path, root) for s in subs):
            errors.append(f"{where}: {_short(value)} matches none of the allowed shapes")
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"{where}: missing required {name!r}")
        extra = [k for k in value if k not in props]
        if extra and schema.get("additionalProperties") is False:
            errors.append(f"{where}: unexpected {', '.join(repr(k) for k in extra)}")
        for name, sub in props.items():
            if name in value:
                errors += validate(value[name], sub, f"{path}.{name}" if path else name, root)
    if isinstance(value, list):
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                errors += validate(item, schema["items"], f"{where}[{i}]", root)
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            errors.append(f"{where}: needs at least {schema['minItems']} items")
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            errors.append(f"{where}: allows at most {schema['maxItems']} items")
    if isinstance(value, str):
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            errors.append(f"{where}: shorter than {schema['minLength']} characters")
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            errors.append(f"{where}: longer than {schema['maxLength']} characters")
    if _TYPES["number"](value):
        for key, ok in (
            ("minimum", lambda v, b: v >= b),
            ("maximum", lambda v, b: v <= b),
            ("exclusiveMinimum", lambda v, b: v > b),
            ("exclusiveMaximum", lambda v, b: v < b),
        ):
            bound = schema.get(key)
            if _TYPES["number"](bound) and not ok(value, bound):
                errors.append(f"{where}: {value} violates {key} {bound}")
    return errors


def _short(value: Any, width: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 1] + "…"
