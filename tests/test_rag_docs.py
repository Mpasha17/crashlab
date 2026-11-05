"""docs/rag-testing.md stays honest: helper names, parameters and defaults it names exist,
listed keys/values match the code, and its commands and snippets run in clean processes."""

from __future__ import annotations

import inspect
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from examples.rag_qa import agent as rag_qa

from crashlab import rag

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "rag-testing.md").read_text("utf-8")
DEFAULTS = {"min_overlap": 0.5, "min_words": 3, "duplicates": "versions", "exclude_stale": False,
            "id_key": "id", "text_key": "text", "top_k": None, "tool": "search_docs",
            "superseded_key": "superseded_by", "require": True}  # fmt: skip


def _blocks(lang):
    return re.findall(rf"```{lang}\n(.*?)```", DOC, re.S)


def test_documented_helpers_and_parameters_exist():
    calls = re.findall(r"rag\.(\w+)\(([^)`]*)\)", DOC)
    assert calls
    for name, args in calls:
        assert name in rag.__all__, name
        params = inspect.signature(getattr(rag, name)).parameters
        for kw in re.findall(r"(\w+)=", args):
            assert kw in params, (name, kw)
    for name in rag.__all__:
        if callable(getattr(rag, name)) and not name.isupper():
            assert f"`{name}" in DOC or f"rag.{name}" in DOC, f"{name} undocumented"


def test_documented_defaults_match_signatures():
    for name in rag.__all__:
        fn = getattr(rag, name)
        if not callable(fn) or name.isupper():
            continue
        for p in inspect.signature(fn).parameters.values():
            if p.name in DEFAULTS:
                assert p.default == DEFAULTS[p.name], (name, p.name, p.default)
    assert "min_overlap=0.5, min_words=3" in DOC and '`"versions"` (default)' in DOC


def test_documented_lists_match_code():
    for key in rag._WRAPPERS:
        assert f"`{key}`" in DOC, key
    for key in rag._TEXT_FALLBACKS[1:] + rag._ID_FALLBACKS[1:]:
        assert f"`{key}`" in DOC, key
    for key in rag.VERSION_KEYS:
        assert f"`{key}`" in DOC, key
    for mode in rag.DUPLICATES:
        assert f'`"{mode}"`' in DOC, mode
    for fault in rag_qa.FAULTS:
        assert f"`{fault}`" in DOC, fault


def _env(store):
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env.update(CRASHLAB_DIR=str(store), PYTHONPATH=str(ROOT))
    return env


RECORD = """
import crashlab
from examples.rag_qa import agent
from tests.fakes import FakeOpenAI, text_response, tool_call_response
script = [tool_call_response(1, "search_docs", {"query": "return policy days"}),
          text_response(2, "You have 60 days to return an item [refund-policy-v2].")]
agent.run_agent(next(iter(agent.GOLD)), client=crashlab.wrap(FakeOpenAI(script)), model="m")
"""

PRELUDE = """
import crashlab
from crashlab import rag
def my_retriever(query):
    return [{"id": "a", "text": "Fee is 5 EUR."}]
run = crashlab.load("last")
answer = run.output
"""


def test_documented_commands_and_snippets_run_in_clean_processes(tmp_path):
    pytest.importorskip("openai")
    exe = shutil.which("crashlab", path=str(Path(sys.executable).parent))
    assert exe, "crashlab console script not installed"
    store, cwd = tmp_path / "store", tmp_path / "work"
    cwd.mkdir()
    env = _env(store)

    def sh(argv):
        return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,
                              timeout=120)  # fmt: skip

    rec = sh([sys.executable, "-c", RECORD])
    assert rec.returncode == 0, rec.stderr
    lines = [ln.split("#")[0].strip() for b in _blocks("bash") for ln in b.splitlines()]
    cmds = [ln for ln in lines if ln]
    assert cmds and all(c.startswith("crashlab ") for c in cmds), cmds
    for cmd in cmds:
        res = sh([exe, *cmd.split()[1:]])
        assert res.returncode in (0, 1), (cmd, res.stdout, res.stderr)
        assert "Traceback" not in res.stdout + res.stderr, (cmd, res.stderr)
        if "--json" in cmd:
            assert '"answer_grounded"' in res.stdout and '"fail"' in res.stdout, res.stdout
    exported = cwd / "tests" / "crashlab"
    assert list(exported.glob("test_rag_qa_*.py")), "export-test last wrote nothing"
    res = sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o",
              "addopts=", str(exported)])  # fmt: skip
    assert res.returncode == 0, res.stdout + res.stderr
    for block in _blocks("python"):
        code = PRELUDE + textwrap.dedent(block)
        res = sh([sys.executable, "-c", code])
        assert res.returncode == 0, (block, res.stderr)
