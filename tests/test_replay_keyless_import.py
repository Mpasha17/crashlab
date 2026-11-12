"""Replay, `crashlab check` and exported tests must work in CI with no API key, also when
the agent module builds its client at import time.

New-user finding: the README quickstart does ``client = crashlab.wrap(OpenAI())`` at
module level. The exported regression test then failed in a keyless CI with
``importing entry point module ... raised OpenAIError: Missing credentials``: placeholder
SDK keys were only set while the agent ran, not while its module was imported. A module
that reads ``os.environ["GROQ_API_KEY"]`` at import got a bare ``KeyError``."""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.replay.divergence import ReplayError

KEYED = """
import os
import crashlab

if not os.environ.get("OPENAI_API_KEY"):  # what OpenAI() does at construction
    raise RuntimeError("Missing credentials: set OPENAI_API_KEY")

@crashlab.tool()
def add(a: int, b: int) -> int:
    return a + b

@crashlab.agent(name="keyless-bot")
def run_agent(x: int) -> int:
    return add(x, 1)

@crashlab.check("keyless-bot")
def is_small(run):
    assert run.output < 10, f"got {run.output}"
"""


def _module(tmp_path, monkeypatch, name: str, source: str):
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    return __import__(name)


def test_replay_and_check_import_a_module_that_needs_a_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-looking-but-fake")
    mod = _module(tmp_path, monkeypatch, "keylessbot", KEYED)
    mod.run_agent(20)
    run = crashlab.load("last")
    assert run.outcome == "fail"

    monkeypatch.delenv("OPENAI_API_KEY")
    sys.modules.pop("keylessbot", None)
    replayed = crashlab.replay(run.run_id)
    assert replayed.output == 21 and replayed.outcome == "fail"

    sys.modules.pop("keylessbot", None)
    res = CliRunner().invoke(app, ["check", run.run_id])
    assert "could not import" not in res.output, res.output
    assert "is_small" in res.output


def test_a_missing_env_var_at_import_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    mod = _module(
        tmp_path,
        monkeypatch,
        "groqbot",
        KEYED.replace(
            "if not os.environ", 'KEY = os.environ.get("GROQ_API_KEY")\nif not os.environ'
        ),
    )
    mod.run_agent(1)
    run = crashlab.load("last")
    (tmp_path / "groqbot.py").write_text(
        KEYED.replace("if not os.environ", 'KEY = os.environ["GROQ_API_KEY"]\nif not os.environ')
    )
    sys.modules.pop("groqbot", None)
    with pytest.raises(ReplayError) as err:
        crashlab.replay(run.run_id)
    msg = str(err.value)
    assert "GROQ_API_KEY" in msg and "any value" in msg and "'unused'" in msg, msg
    assert "REDACTED" not in msg, msg  # the redactor ate "NAME=unused"
