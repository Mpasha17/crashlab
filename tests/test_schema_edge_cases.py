"""Tool-argument schema validator edge cases seen in real tool schemas."""

from __future__ import annotations

from crashlab.checks.schema import validate


def test_ref_cycle_without_progress_is_not_a_recursion_error():
    schema = {
        "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"allOf": [{"$ref": "#/$defs/A"}]}},
        "$ref": "#/$defs/A",
    }
    assert validate({"a": 1}, schema) == []


def test_recursive_model_still_validated_deeply():
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"name": {"type": "string"},
                               "kids": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
                "required": ["name"],
            }
        },
        "$ref": "#/$defs/Node",
    }  # fmt: skip
    deep = {"name": "0"}
    cur = deep
    for i in range(1, 30):
        cur["kids"] = [{"name": str(i)}]
        cur = cur["kids"][0]
    assert validate(deep, schema) == []
    cur["kids"] = [{"nom": "x"}]
    assert any("missing required 'name'" in p for p in validate(deep, schema))


def test_uppercase_types_google_style():
    schema = {"type": "OBJECT", "properties": {"q": {"type": "STRING"}}, "required": ["q"]}
    assert validate({"q": "x"}, schema) == []
    assert validate({"q": 5}, schema) == ["q: expected string, got number 5"]


def test_json_pointer_escapes_in_ref():
    schema = {"$defs": {"a/b": {"type": "string"}, "c~d": {"type": "integer"}}}
    assert validate(1, {**schema, "$ref": "#/$defs/a~1b"}) == [
        "arguments: expected string, got number 1"
    ]
    assert validate("x", {**schema, "$ref": "#/$defs/c~0d"})
