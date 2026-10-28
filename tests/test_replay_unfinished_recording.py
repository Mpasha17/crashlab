"""A run whose process was killed mid-recording stays 'running' in the store. Its
replay used to report "the final output differs" (recorded null vs whatever the replay
returned) or a bare "recording has only N steps"; `crashlab ls` showed 0 steps although
the cassette held them. The replay now says the recording never finished and passes when
everything recorded matched; ls counts the stored steps."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab.cli import app
from crashlab.replay.divergence import DivergenceError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL")

SCRIPT = textwrap.dedent(
    """
    import sys, time
    import crashlab

    @crashlab.tool
    def step(i):
        return i

    @crashlab.agent(name="killed")
    def agent():
        for i in range(3):
            step(i)
        print("READY", flush=True)
        time.sleep(120)
        return "done"

    agent()
    """
)


@crashlab.tool
def step(i):
    return i


def _record_killed(tmp_path, crashlab_dir) -> crashlab.Run:
    script = tmp_path / "killme.py"
    script.write_text(SCRIPT)
    env = {**os.environ, "CRASHLAB_DIR": os.environ["CRASHLAB_DIR"]}
    p = subprocess.Popen([sys.executable, "-u", str(script)], cwd=tmp_path, env=env,
                         stdout=subprocess.PIPE, text=True)  # fmt: skip
    try:
        assert p.stdout.readline().strip() == "READY"
    finally:
        p.send_signal(signal.SIGKILL)
        p.wait(timeout=20)
    run = crashlab.load("last")
    assert run.status == "running" and [s.idx for s in run.steps] == [0, 1, 2]
    return run


def test_ls_counts_steps_of_an_unfinished_run(tmp_path, crashlab_dir):
    run = _record_killed(tmp_path, crashlab_dir)
    row = crashlab.Store().list_runs()[0]
    assert row["run_id"] == run.run_id and row["steps"] == 3


def test_replay_of_a_killed_run_reproduces_what_was_recorded(tmp_path, crashlab_dir):
    run = _record_killed(tmp_path, crashlab_dir)

    @crashlab.agent(name="killed")
    def agent():
        for i in range(3):
            step(i)
        return "done"  # the recording has no output to compare this with

    assert crashlab.replay(run, fn=agent).status == "ok"

    @crashlab.agent(name="killed")
    def goes_on():
        for i in range(4):
            step(i)

    with pytest.raises(DivergenceError, match="recording never finished"):
        crashlab.replay(run, fn=goes_on)


def test_cli_replay_says_the_recording_never_finished(tmp_path, crashlab_dir, monkeypatch):
    run = _record_killed(tmp_path, crashlab_dir)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "killme.py").write_text(SCRIPT.replace("time.sleep(120)", "pass").replace(
        "agent()\n", 'if __name__ == "__main__":\n    agent()\n'))  # fmt: skip
    monkeypatch.syspath_prepend(str(tmp_path))
    res = CliRunner().invoke(app, ["replay", run.run_id], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    assert "never finished" in res.output and "identical requests and output" not in res.output
