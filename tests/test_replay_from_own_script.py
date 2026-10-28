"""Replaying from inside the script that recorded (no ``__main__`` guard, the natural
quickstart shape) must not execute the script's top level again: it used to record and
replay recursively until RecursionError."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_replay_inside_unguarded_script_runs_top_level_once(tmp_path):
    (tmp_path / "bot.py").write_text(
        textwrap.dedent("""
        import crashlab
        print("top-level", __name__, flush=True)

        @crashlab.tool
        def add(a: int, b: int) -> int:
            return a + b

        @crashlab.agent(name="noguard")
        def run(x: int) -> int:
            return add(x, 1)

        print("live", run(41), flush=True)
        print("replay", crashlab.replay(crashlab.load("last")).output, flush=True)
    """)
    )
    env = {**os.environ, "CRASHLAB_DIR": str(tmp_path / "store")}
    proc = subprocess.run(
        [sys.executable, "bot.py"], cwd=tmp_path, env=env, capture_output=True, text=True,
        timeout=60,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = proc.stdout.splitlines()
    assert lines == ["top-level __main__", "live 42", "replay 42"], proc.stdout[-2000:]
