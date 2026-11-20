"""crashlab.bisect / crashlab bisect: compare runs and isolate the causal change.

No module-level openai / anthropic / litellm / langchain / mcp / yaml imports.
"""

from __future__ import annotations

import re
from typing import Any

from typer.testing import CliRunner

import crashlab
from crashlab.bisect import bisect_runs
from crashlab.cli import app
from tests.fakes import ChatCompletion, Choice, CompletionUsage, Message


def _reply(text: str, model: str = "m") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-1",
        model=model,
        choices=[Choice(message=Message(content=text), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
    )


class PromptClient:
    """OpenAI-shaped client whose reply depends on the request (system / model / tools)."""

    base_url = "http://fake.local/v1"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        messages = kwargs.get("messages") or []
        system = ""
        if messages and messages[0].get("role") == "system":
            system = messages[0].get("content") or ""
        model = kwargs.get("model") or "m"
        tools = kwargs.get("tools") or []
        desc = ""
        if tools:
            fn = tools[0].get("function") or tools[0]
            desc = fn.get("description") or ""
        temp = kwargs.get("temperature")
        if "refund promptly" in system or model == "good-model" or desc == "look up the price":
            text = "ok"
        elif temp == 0 and "refund promptly" not in system and model != "good-model" and not desc:
            # temperature alone is not enough unless we treat 0 as the good param
            text = "nope"
        else:
            text = "nope"
        # good temperature is 0 only when that is the isolated factor — handled below
        if temp == 0 and system == "be brief" and model == "m" and not tools:
            text = "ok"
        return _reply(text, model=model)


@crashlab.check("shop")
def answered_ok(run: crashlab.Run) -> None:
    """The agent answered ok."""
    assert run.output == "ok", f"answered {run.output!r}"


# Module-level knobs + agent so forks can re-import the entry point.
CLIENT = PromptClient()
KNOBS: dict[str, Any] = {
    "system": "be brief",
    "model": "m",
    "tools": None,
    "temperature": None,
}


def _tools(description: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_price",
                "description": description,
                "parameters": {"type": "object", "properties": {"item": {"type": "string"}}},
            },
        }
    ]


@crashlab.agent(name="shop")
def run_shop(msg: str) -> str:
    oc = crashlab.wrap(CLIENT)
    system, model = KNOBS["system"], KNOBS["model"]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": msg}]
    kw: dict[str, Any] = {"model": model, "messages": messages}
    if KNOBS["tools"] is not None:
        kw["tools"] = KNOBS["tools"]
    if KNOBS["temperature"] is not None:
        kw["temperature"] = KNOBS["temperature"]
    out = oc.chat.completions.create(**kw)
    return out.choices[0].message.content


def _run(
    *,
    system: str = "be brief",
    model: str = "m",
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
) -> crashlab.Run:
    KNOBS.update(system=system, model=model, tools=tools, temperature=temperature)
    run_shop("go")
    return crashlab.load("last")


def test_bisect_ranks_system_prompt_and_model_differences():
    good = _run(system="refund promptly")
    bad = _run(system="be brief", model="other-model")
    report = bisect_runs(good, bad)
    kinds = [f.kind for f in report.factors]
    assert "system_prompt" in kinds
    assert "model" in kinds
    assert "trajectory" in kinds
    assert report.first_divergence == 0
    # the diverging step's system field makes the prompt the closest factor (or tied at 0)
    prompt = next(f for f in report.factors if f.kind == "system_prompt")
    assert prompt.proximity == 0
    assert prompt.good == "refund promptly" and prompt.bad == "be brief"
    assert "first divergence at step 0" in report.summary()
    assert report.causal is None and report.isolated is False


def test_isolate_finds_the_system_prompt_as_the_minimal_causal_change():
    good = _run(system="refund promptly")
    bad = _run(system="be brief")
    assert good.outcome == "pass" and bad.outcome == "fail"
    report = bisect_runs(good, bad, isolate=True)
    assert report.isolated
    assert report.causal is not None
    assert report.causal.factor_kind == "system_prompt"
    assert report.causal.restored and report.causal.outcome == "pass"
    assert "replace" in report.causal.patch_description
    assert any(t.factor_kind == "system_prompt" and t.restored for t in report.trials)
    assert "minimal causal change" in report.summary()


def test_isolate_finds_a_model_change():
    good = _run(system="be brief", model="good-model")
    bad = _run(system="be brief", model="other-model")
    report = bisect_runs(good, bad, isolate=True)
    assert report.causal is not None
    assert report.causal.factor_kind == "model"
    assert report.causal.restored


def test_isolate_finds_a_tool_description_change():
    good = _run(tools=_tools("look up the price"))
    bad = _run(tools=_tools("look up an item"))
    report = bisect_runs(good, bad, isolate=True)
    desc = [f for f in report.factors if f.kind == "tool_description"]
    assert desc and desc[0].details["tool"] == "get_price"
    assert report.causal is not None
    assert report.causal.factor_kind == "tool_description"
    assert report.causal.restored


def test_identical_runs_have_no_factors_besides_maybe_code():
    # two live recordings of the same passing prompt: steps should match; code_version
    # is the same git commit in this worktree
    a = _run(system="refund promptly")
    b = _run(system="refund promptly")
    report = bisect_runs(a, b)
    kinds = {f.kind for f in report.factors}
    assert "trajectory" not in kinds
    assert "system_prompt" not in kinds
    assert report.first_divergence is None
    assert "all " in report.summary() and "steps are the same" in report.summary()


def test_cli_bisect_json_and_help(tmp_path, monkeypatch):
    monkeypatch.setenv("CRASHLAB_DIR", str(tmp_path))
    good = _run(system="refund promptly")
    bad = _run(system="be brief")
    runner = CliRunner()
    help_out = runner.invoke(app, ["bisect", "--help"])
    assert help_out.exit_code == 0
    clean = re.sub(r"\x1b\[[0-9;]*m", "", help_out.output)
    assert "--isolate" in clean and "--agent" in clean
    res = runner.invoke(app, ["bisect", good.run_id, bad.run_id, "--json"])
    assert res.exit_code == 0, res.output
    import json

    payload = json.loads(res.output)
    assert payload["good"] == good.run_id and payload["bad"] == bad.run_id
    assert payload["first_divergence"] == 0
    assert any(f["kind"] == "system_prompt" for f in payload["factors"])
    iso = runner.invoke(app, ["bisect", good.run_id, bad.run_id, "--isolate"])
    assert iso.exit_code == 0, iso.output
    assert "minimal causal change" in iso.output
    missing = runner.invoke(app, ["bisect", "no-such-run", bad.run_id])
    assert missing.exit_code == 1
