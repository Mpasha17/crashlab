"""An exported test whose agent module cannot be imported fails loudly, it does not skip."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

AGENT = textwrap.dedent(
    """
    import crashlab

    @crashlab.tool()
    def double(x: int) -> int:
        return 2 * x

    @crashlab.agent(name="w3-mod-agent")
    def run(x: int) -> int:
        return double(x)

    @crashlab.check("w3-mod-agent")
    def is_even(r):
        assert r.output % 2 == 0
    """
)


def _env(store):
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env["CRASHLAB_DIR"] = str(store)
    env.pop("PYTHONPATH", None)
    return env


def test_exported_test_fails_when_agent_module_is_not_importable(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "w3_mod_agent.py").write_text(AGENT, "utf-8")
    store = tmp_path / "store"
    env = _env(store)
    subprocess.run([sys.executable, "-c", "import w3_mod_agent; w3_mod_agent.run(2)"],
                   cwd=proj, env=env, check=True)  # fmt: skip
    subprocess.run([sys.executable, "-m", "crashlab.cli", "export-test", "last", "-o",
                    str(proj / "tests")], cwd=proj, env=env, check=True)  # fmt: skip
    env.pop("CRASHLAB_DIR")
    base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
            "-p", "no:randomly", "-rs"]  # fmt: skip
    ok = subprocess.run([*base, "tests"], cwd=proj, env=env, capture_output=True, text=True)
    assert "1 passed" in ok.stdout, ok.stdout + ok.stderr
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    bad = subprocess.run([*base, str(proj / "tests"), "--rootdir", str(elsewhere)],
                         cwd=elsewhere, env=env, capture_output=True, text=True)  # fmt: skip
    out = bad.stdout + bad.stderr
    assert bad.returncode == 1 and "1 failed" in out, out
    assert "cannot import the agent's module 'w3_mod_agent'" in out
