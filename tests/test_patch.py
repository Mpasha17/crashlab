"""Typed patches: every kind, on every request family, composition and serialization.
Pure functions on request dicts; no network."""

from __future__ import annotations

import json

import pytest

import crashlab
from crashlab import patch as P

POLICY = "You are a bot.\n\nPolicy:\n1. Look up first.\n2. Never refund above total.\n\nBe brief."
TOOLS_CHAT = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up.",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
        },
    }
]
TOOLS_ANTHROPIC = [
    {
        "name": "get_order",
        "description": "Look up.",
        "input_schema": {"type": "object", "properties": {}},
    }
]
TOOLS_RESPONSES = [
    {
        "type": "function",
        "name": "get_order",
        "description": "Look up.",
        "parameters": {"type": "object", "properties": {}},
    }
]


def chat_kwargs(system: str = POLICY) -> dict:
    return {
        "model": "m",
        "temperature": 0,
        "tools": TOOLS_CHAT,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "refund C03"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"total": 1}'},
        ],
    }


def anthropic_kwargs() -> dict:
    return {
        "model": "claude",
        "max_tokens": 10,
        "system": POLICY,
        "tools": TOOLS_ANTHROPIC,
        "messages": [{"role": "user", "content": "refund C03"}],
    }


def responses_kwargs() -> dict:
    return {
        "model": "gpt",
        "instructions": POLICY,
        "tools": TOOLS_RESPONSES,
        "input": [
            {"role": "user", "content": "refund C03"},
            {"type": "function_call", "call_id": "c1", "name": "get_order", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "{}"},
        ],
    }


def system_of(api: str, kwargs: dict) -> str | None:
    return P.normalize_request(api, kwargs)["system"]


FAMILIES = [
    ("openai.chat.completions", chat_kwargs),
    ("litellm.completion", chat_kwargs),
    ("anthropic.messages", anthropic_kwargs),
    ("openai.responses", responses_kwargs),
]


def apply(patch, api, kwargs, idx=0, fork_step=0, state=None):
    out, applied = P.as_patchset(patch).apply_chat(
        api, idx, kwargs, fork_step=fork_step, state=state
    )
    return out, applied


# --- system prompt -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        (P.SystemPromptPatch(op="replace", text="new"), "new"),
        (P.SystemPromptPatch(op="append", text="Extra."), POLICY + "\nExtra."),
        (P.SystemPromptPatch(op="prepend", text="Top."), "Top.\n" + POLICY),
        (
            P.SystemPromptPatch(op="insert_line", after="1. Look up", text="1b. Window: 30 days."),
            POLICY.replace("first.\n", "first.\n1b. Window: 30 days.\n"),
        ),
        (
            P.SystemPromptPatch(op="insert_line", before="2. Never", text="X"),
            POLICY.replace("2. Never", "X\n2. Never"),
        ),
        (P.SystemPromptPatch(op="insert_line", line=1, text="Header"), "Header\n" + POLICY),
        (
            P.SystemPromptPatch(op="replace_line", match="Be brief", text="Be kind."),
            POLICY.replace("Be brief.", "Be kind."),
        ),
        (
            P.SystemPromptPatch(op="delete_line", line=5),
            POLICY.replace("2. Never refund above total.\n", ""),
        ),
        (
            P.SystemPromptPatch(op="substitute", old="above total", text="above the order total"),
            POLICY.replace("above total", "above the order total"),
        ),
    ],
)
@pytest.mark.parametrize(("api", "make"), FAMILIES)
def test_system_prompt_ops_on_every_family(patch, expected, api, make):
    kwargs = make()
    before = json.dumps(kwargs, sort_keys=True)
    out, applied = apply(patch, api, kwargs)
    assert applied == [0]
    assert system_of(api, out) == expected
    assert json.dumps(kwargs, sort_keys=True) == before  # the caller's request is untouched


def test_system_prompt_created_when_missing_and_anthropic_blocks_kept():
    kw = chat_kwargs()
    kw["messages"] = kw["messages"][1:]
    out, _ = apply(P.SystemPromptPatch(op="append", text="Be nice."), "openai.chat.completions", kw)
    assert out["messages"][0] == {"role": "system", "content": "Be nice."}
    ak = anthropic_kwargs()
    ak["system"] = [{"type": "text", "text": POLICY, "cache_control": {"type": "ephemeral"}}]
    out, _ = apply(P.SystemPromptPatch(op="append", text="X"), "anthropic.messages", ak)
    assert out["system"] == [
        {"type": "text", "text": POLICY + "\nX", "cache_control": {"type": "ephemeral"}}
    ]


def test_developer_role_and_several_system_messages_are_merged():
    kw = chat_kwargs()
    kw["messages"].insert(1, {"role": "developer", "content": "Second."})
    kw["messages"][0]["role"] = "developer"
    out, _ = apply(P.SystemPromptPatch(op="append", text="Third."), "openai.chat.completions", kw)
    assert out["messages"][0] == {"role": "developer", "content": POLICY + "\n\nSecond.\nThird."}
    assert out["messages"][1]["role"] == "user"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"op": "insert_line", "after": "nope", "text": "x"}, "no line containing"),
        ({"op": "insert_line", "after": ".", "text": "x"}, "matches 4 system prompt lines"),
        ({"op": "replace_line", "line": 99, "text": "x"}, "has 7 lines"),
        ({"op": "substitute", "old": "absent", "text": "x"}, "does not contain"),
    ],
)
def test_system_prompt_patch_errors(kwargs, match):
    with pytest.raises(P.PatchError, match=match):
        apply(P.SystemPromptPatch(**kwargs), "openai.chat.completions", chat_kwargs())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"op": "insert_line", "text": "x"},
        {"op": "insert_line", "text": "x", "line": 1, "after": "a"},
        {"op": "delete_line"},
        {"op": "substitute", "text": "x"},
        {"op": "replace_line", "line": 0, "text": "x"},
    ],
)
def test_system_prompt_patch_validation(kwargs):
    with pytest.raises(ValueError):
        P.SystemPromptPatch(**kwargs)


def test_request_wide_patches_respect_fork_step_and_from_step():
    p = P.SystemPromptPatch(op="append", text="X", from_step=4)
    assert apply(p, "openai.chat.completions", chat_kwargs(), idx=2)[1] == []
    assert apply(p, "openai.chat.completions", chat_kwargs(), idx=4)[1] == [0]
    q = P.SystemPromptPatch(op="append", text="X")
    assert apply(q, "openai.chat.completions", chat_kwargs(), idx=1, fork_step=3)[1] == []


# --- messages ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("api", "make", "index"),
    [
        ("openai.chat.completions", chat_kwargs, 0),
        ("anthropic.messages", anthropic_kwargs, 0),
        ("openai.responses", responses_kwargs, 0),
    ],
)
def test_message_patch_edits_normalized_index(api, make, index):
    out, applied = apply(P.MessagePatch(step=0, index=index, content="refund A17"), api, make())
    assert applied == [0]
    msgs = P.normalize_request(api, out)["messages"]
    assert msgs[index]["parts"][0]["content"] == "refund A17"


def test_message_index_maps_through_merged_responses_items():
    # assistant text + function_call items normalize to ONE assistant message, so the
    # normalized index of the tool output (2) is native item 3
    kw = {
        "model": "gpt",
        "input": [
            {"role": "user", "content": "refund C03"},
            {"role": "assistant", "content": "Looking it up."},
            {"type": "function_call", "call_id": "c1", "name": "get_order", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "{}"},
        ],
    }
    assert len(P.normalize_request("openai.responses", kw)["messages"]) == 3
    ed = P.RequestEditor("openai.responses", kw)
    assert [ed.native_index(i) for i in range(3)] == [0, 1, 3]
    assert ed.native_index(3, insert=True) == 4
    out, _ = apply(P.MessagePatch(step=0, index=1, content="Checking."), "openai.responses", kw)
    assert out["input"][1]["content"] == "Checking." and out["input"][2]["type"] == "function_call"


def test_message_patch_is_sticky_while_the_original_message_is_carried():
    patch = P.MessagePatch(step=0, index=0, content="refund A17", role="user")
    state: dict = {}
    first, _ = apply(patch, "openai.chat.completions", chat_kwargs(), idx=0, state=state)
    later, applied = apply(patch, "openai.chat.completions", chat_kwargs(), idx=2, state=state)
    assert applied == [0] and later["messages"][1]["content"] == "refund A17"
    other = chat_kwargs()
    other["messages"][1]["content"] = "something else"  # a different conversation: untouched
    out, applied = apply(patch, "openai.chat.completions", other, idx=4, state=state)
    assert applied == [] and out["messages"][1]["content"] == "something else"
    assert first["messages"][1]["content"] == "refund A17"


def test_message_patch_errors():
    with pytest.raises(P.PatchError, match="out of range"):
        apply(
            P.MessagePatch(step=0, index=9, content="x"), "openai.chat.completions", chat_kwargs()
        )
    with pytest.raises(ValueError):
        P.MessagePatch(step=0, index=0)


def test_inject_message_at_end_and_before_index_and_sticky():
    state: dict = {}
    patch = P.InjectMessagePatch(step=1, content="Remember the 30-day window.")
    out, applied = apply(patch, "openai.chat.completions", chat_kwargs(), idx=1, state=state)
    assert applied == [0] and out["messages"][-1] == {
        "role": "user",
        "content": "Remember the 30-day window.",
    }
    longer = chat_kwargs()
    longer["messages"].append({"role": "assistant", "content": "ok"})
    out, _ = apply(patch, "openai.chat.completions", longer, idx=3, state=state)
    assert out["messages"][4]["content"] == "Remember the 30-day window."
    assert out["messages"][5]["content"] == "ok"
    first, _ = apply(
        P.InjectMessagePatch(step=0, index=0, role="user", content="hi"),
        "anthropic.messages",
        anthropic_kwargs(),
    )
    assert first["messages"][0]["content"] == "hi"


# --- tools / params ----------------------------------------------------------------


@pytest.mark.parametrize(("api", "make"), FAMILIES)
def test_tool_description_and_schema(api, make):
    out, _ = apply(
        P.ToolDescriptionPatch(
            tool="get_order",
            description="Returns JSON, or an HTML error page when the backend is down.",
        ),
        api,
        make(),
    )
    tool = P.normalize_request(api, out)["tools"][0]
    assert tool["description"].startswith("Returns JSON, or an HTML error page")
    schema = {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    }
    out, _ = apply(P.ToolSchemaPatch(tool="get_order", parameters=schema), api, make())
    norm = P.normalize_request(api, out)["tools"][0]
    assert (norm.get("parameters") or norm.get("input_schema")) == schema


def test_tool_patches_append_merge_and_unknown_tool():
    out, _ = apply(
        P.ToolDescriptionPatch(tool="get_order", description="Never guess.", op="append"),
        "openai.chat.completions",
        chat_kwargs(),
    )
    assert out["tools"][0]["function"]["description"] == "Look up. Never guess."
    merged, _ = apply(
        P.ToolSchemaPatch(tool="get_order", merge=True, parameters={"required": ["id"]}),
        "openai.chat.completions",
        chat_kwargs(),
    )
    params = merged["tools"][0]["function"]["parameters"]
    assert params["required"] == ["id"] and "id" in params["properties"]
    with pytest.raises(P.PatchError, match="not in the request"):
        apply(
            P.ToolDescriptionPatch(tool="nope", description="x"),
            "openai.chat.completions",
            chat_kwargs(),
        )


def test_model_params_patch():
    out, _ = apply(
        P.ModelParamPatch(model="bigger", params={"temperature": 0.7, "seed": 1}, unset=["tools"]),
        "openai.chat.completions",
        chat_kwargs(),
    )
    assert (out["model"], out["temperature"], out["seed"]) == ("bigger", 0.7, 1)
    assert "tools" not in out
    with pytest.raises(ValueError):
        P.ModelParamPatch()
    with pytest.raises(ValueError):
        P.ModelParamPatch(params={"model": "x"})


def test_tool_result_patch_does_not_touch_requests():
    out, applied = apply(
        P.ToolResultPatch(step=1, result={"total": 49}), "openai.chat.completions", chat_kwargs()
    )
    assert applied == [] and out == chat_kwargs()
    assert not P.as_patchset(P.ToolResultPatch(step=1)).edits_requests


# --- composition / serialization ---------------------------------------------------


def all_kinds() -> P.PatchSet:
    return (
        P.SystemPromptPatch(op="insert_line", after="1. Look up", text="1b. 30-day window.")
        + P.MessagePatch(step=2, index=0, content="hi")
        + P.InjectMessagePatch(step=2, content="be careful")
        + P.ToolResultPatch(step=1, result={"total": 49.0})
        + P.ToolResultPatch(step=3, error="timeout", tool="get_order")
        + P.ToolDescriptionPatch(tool="get_order", description="d")
        + P.ToolSchemaPatch(tool="get_order", parameters={"type": "object"})
        + P.ModelParamPatch(model="m2", params={"temperature": 0.2})
    )


def test_composition_is_ordered_and_flat():
    ps = all_kinds()
    assert isinstance(ps, P.PatchSet) and len(ps) == 8
    assert [p.kind for p in ps][:3] == ["system_prompt", "message", "inject_message"]
    both = P.PatchSet.of(ps, [P.ModelParamPatch(params={"seed": 3})], None)
    assert len(both) == 9 and len(ps + ps) == 16
    a = P.SystemPromptPatch(op="append", text="A")
    b = P.SystemPromptPatch(op="append", text="B")
    out, applied = apply(a + b, "openai.chat.completions", chat_kwargs())
    assert applied == [0, 1] and system_of("openai.chat.completions", out).endswith("\nA\nB")
    assert [x.text for x in [b] + P.PatchSet.of(a)] == ["B", "A"]  # __radd__


def test_json_round_trip_and_as_patchset_inputs(tmp_path):
    ps = all_kinds()
    again = P.as_patchset(ps.to_json())
    assert again == ps
    assert P.as_patchset(json.loads(ps.to_json())["patches"]) == ps
    assert P.as_patchset({"kind": "model_params", "model": "x"}).patches[0].model == "x"
    path = tmp_path / "fix.json"
    path.write_text(ps.model_dump_json())
    assert P.load(path) == ps
    with pytest.raises(P.PatchError):
        bad = tmp_path / "bad.json"
        bad.write_text('{"kind": "nope"}')
        P.load(bad)
    with pytest.raises(TypeError):
        P.as_patchset(42)


def test_yaml_patch_file(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "fix.yaml"
    path.write_text(
        "patches:\n"
        "  - kind: system_prompt\n"
        "    op: insert_line\n"
        "    after: '1. Look up'\n"
        "    text: '1b. Only refund orders delivered at most 30 days ago.'\n"
        "  - kind: model_params\n"
        "    params: {temperature: 0}\n"
    )
    ps = P.load(path)
    assert [p.kind for p in ps] == ["system_prompt", "model_params"]


def test_describe_is_human_readable():
    lines = all_kinds().describe_lines()
    assert lines[0] == (
        'system prompt: insert after the line with "1. Look up": "1b. 30-day window."'
    )
    assert lines[3] == 'tool result at step 1 -> {"total": 49.0}'
    assert lines[4] == 'tool result at step 3 (get_order): raise "timeout"'
    assert lines[7] == "model params: model -> m2, temperature=0.2"
    assert P.PatchSet().describe() == "no patch"
    assert "delete the line" in P.SystemPromptPatch(op="delete_line", match="x").describe()


def test_short_aliases_and_top_level_exports():
    assert crashlab.patch.SystemPrompt is crashlab.SystemPromptPatch
    assert crashlab.patch.ToolResult is crashlab.ToolResultPatch
    assert crashlab.patch.Model is crashlab.ModelParamPatch
