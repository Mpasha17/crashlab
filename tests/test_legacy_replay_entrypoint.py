"""Runs recorded by crashlab<=0.0.1 during a replay of a script carry the entry point
``crashlab_replay_<stem>:fn``, a module that only existed in that process. Replay must find
the script by name instead of depending on another test having imported it first."""

from __future__ import annotations

import sys

import pytest

from crashlab.replay.engine import ReplayError, resolve_entrypoint


def test_legacy_replay_module_resolves_to_the_script(tmp_path, monkeypatch):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "legacybot.py").write_text("def run_agent(x):\n    return x * 2\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "legacybot.py").write_text("raise SystemExit('wrong file')\n")
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("crashlab_replay_legacybot", None)
    fn = resolve_entrypoint("crashlab_replay_legacybot:run_agent")
    assert fn(21) == 42


def test_legacy_replay_module_ambiguous_or_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("crashlab_replay_nobot", None)
    with pytest.raises(ReplayError, match="not found"):
        resolve_entrypoint("crashlab_replay_nobot:run_agent")
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "nobot.py").write_text("def run_agent():\n    pass\n")
    with pytest.raises(ReplayError, match="ambiguous"):
        resolve_entrypoint("crashlab_replay_nobot:run_agent")


def test_exported_fix_cassette_replays_without_its_sibling_running_first(crashlab_replay):
    """The refund_bot fix cassette (crashlab 0.0.1) records ``crashlab_replay_agent``; it
    used to pass only when the sibling reproduce test had imported that module in the same
    process, so it failed under ``pytest -n`` / random order."""
    from pathlib import Path

    for name in [m for m in sys.modules if m.startswith("crashlab_replay_")]:
        sys.modules.pop(name)
    cassettes = Path(__file__).parent.parent / "examples/refund_bot/tests/cassettes"
    run = crashlab_replay(cassettes / "refund_bot_01m5gtk1fr_fixed.jsonl")
    assert run.outcome == "pass"
