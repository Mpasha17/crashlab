"""A run recorded from a script (``python agent.py``) replays, and so does its replay."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import crashlab

SCRIPT = textwrap.dedent(
    """
    import crashlab

    @crashlab.tool()
    def double(x: int) -> int:
        return 2 * x

    @crashlab.agent(name="script-agent")
    def main(x: int) -> int:
        return double(x) + double(x + 1)

    if __name__ == "__main__":
        main(3)
    """
)


def test_replay_of_a_replay_of_a_script(tmp_path, monkeypatch, crashlab_dir):
    (tmp_path / "agent_script.py").write_text(SCRIPT, "utf-8")
    subprocess.run(
        [sys.executable, "agent_script.py"],
        cwd=tmp_path,
        check=True,
        env={**os.environ, "CRASHLAB_DIR": str(crashlab_dir)},
    )
    monkeypatch.chdir(tmp_path)
    src = crashlab.load("last")
    assert src.entrypoint == "agent_script.py:main"
    first = crashlab.replay(src)
    assert first.status == "ok" and first.output == 14
    # the replay imported the file under a synthetic module name; it must still record
    # the file path so `crashlab replay last` works on the replay
    assert first.entrypoint == "agent_script.py:main"
    second = crashlab.replay("last")
    assert second.replay_of == first.run_id and second.output == 14
