"""Failure handling around cassette writes and compact expansion: a failed write
(disk full) must not damage the next line or leave dangling compact references; long
and out-of-order chains expand; warnings and flags never carry secrets from the file."""

from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from crashlab.store import compact
from crashlab.store.cassette import CassetteWriter, read_cassette

SECRET = "sk-proj-" + "A1b2C3d4" * 6


def line(i: int, n_msgs: int) -> dict:
    msgs = [{"role": "user", "content": f"q{j} " * 5} for j in range(n_msgs)]
    return {"seq": i, "op": "chat", "name": "m", "hash": f"h{i}", "run_id": "r",
            "span_id": f"s{i}", "request": {"messages": msgs}, "response": {"t": i}}  # fmt: skip


def read(p):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        steps = read_cassette(p)
    return steps, [str(w.message) for w in rec]


class _HalfThenFull:
    """Writes half of the data, then fails like a full disk."""

    def __init__(self, fh):
        self.fh = fh

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.fh.close()

    def write(self, data):
        self.fh.write(data[: len(data) // 2])
        self.fh.flush()
        raise OSError(errno.ENOSPC, "No space left on device")

    def flush(self):
        pass

    def __getattr__(self, name):  # seek / truncate go to the real file
        return getattr(self.fh, name)


@pytest.mark.parametrize("compact_on", [False, True])
def test_disk_full_mid_write_damages_only_that_step(monkeypatch, tmp_path, compact_on):
    if compact_on:
        monkeypatch.setenv(compact.ENV, "1")
    else:
        monkeypatch.delenv(compact.ENV, raising=False)
    w = CassetteWriter(tmp_path / "c.jsonl")
    real_open = Path.open
    fail = {"on": False}

    def fake_open(self, mode="r", *a, **k):
        fh = real_open(self, mode, *a, **k)
        return _HalfThenFull(fh) if fail["on"] and mode.startswith("a") else fh

    monkeypatch.setattr(Path, "open", fake_open)
    for i in range(6):
        fail["on"] = i == 3
        if i == 3:
            with pytest.raises(OSError):
                w.append(line(i, i + 1))
        else:
            w.append(line(i, i + 1))
    monkeypatch.setattr(Path, "open", real_open)
    steps, msgs = read(tmp_path / "c.jsonl")
    assert [s.idx for s in steps] == [0, 1, 2, 4, 5]
    assert not any(compact.UNRESOLVED_ATTR in s.attributes for s in steps)
    assert [len(s.request["messages"]) for s in steps] == [1, 2, 3, 5, 6]
    assert not msgs  # the failed write was rolled back, nothing left to warn about


def test_long_chain_and_out_of_order_lines_expand(monkeypatch, tmp_path):
    monkeypatch.setenv(compact.ENV, "1")
    p = tmp_path / "c.jsonl"
    w = CassetteWriter(p)
    for i in range(1200):
        w.append(line(i, i + 1))
    steps, msgs = read(p)
    assert not msgs and len(steps) == 1200
    assert all(len(s.request["messages"]) == s.idx + 1 for s in steps)
    rev = tmp_path / "rev.jsonl"
    rev.write_text("\n".join(p.read_text().splitlines()[::-1]) + "\n")
    steps2, msgs2 = read(rev)
    assert not msgs2
    dump = lambda ss: [s.model_dump(exclude={"started_at"}) for s in ss]  # noqa: E731
    assert dump(steps2) == dump(steps)


def test_warnings_and_flags_do_not_leak_secrets_from_the_file(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({**line(0, 1), f"key_{SECRET}": SECRET, "error": None}) + "\n"
        + json.dumps({**line(1, 1), "compact": {"v": 1, f"request.{SECRET}": {"base": 0}}})
        + "\n"
        + json.dumps({**line(2, 1), "compact": {"v": SECRET}}) + "\n"
    )  # fmt: skip
    steps, msgs = read(p)
    assert len(steps) == 3 and msgs
    blob = " ".join(msgs) + json.dumps([s.model_dump(mode="json") for s in steps])
    assert SECRET not in blob and SECRET[:20] not in blob


def test_unknown_future_field_values_are_not_kept(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({**line(0, 1), "future": {"token": SECRET}}) + "\n")
    steps, msgs = read(p)
    assert "future" in " ".join(msgs)
    assert SECRET not in json.dumps(steps[0].model_dump(mode="json"))


WRITER = textwrap.dedent(
    """
    import sys
    from crashlab.store.cassette import CassetteWriter
    w = CassetteWriter(sys.argv[1])
    msgs = []
    for i in range(100000):
        msgs.append({"role": "user", "content": "q%d " % i * 20})
        w.append({"seq": i, "op": "chat", "name": "m", "hash": "h%d" % i, "run_id": "r",
                  "span_id": "s%d" % i, "request": {"messages": msgs}, "response": {}})
        if i == 3:
            print("go", flush=True)
    """
)


@pytest.mark.parametrize("delay", [0.05, 0.15, 0.3])
def test_killed_writer_process_leaves_a_readable_prefix(tmp_path, delay):
    p = tmp_path / "c.jsonl"
    env = {**os.environ, compact.ENV: "1"}
    proc = subprocess.Popen([sys.executable, "-c", WRITER, str(p)], env=env,
                            stdout=subprocess.PIPE, text=True)  # fmt: skip
    assert proc.stdout.readline().strip() == "go"
    try:
        proc.wait(timeout=delay)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    steps, msgs = read(p)
    assert len(steps) >= 4
    assert [s.idx for s in steps] == list(range(len(steps)))  # a prefix, no gaps
    assert all(len(s.request["messages"]) == s.idx + 1 for s in steps)
    assert not any(compact.UNRESOLVED_ATTR in s.attributes for s in steps)
    assert all("could not be expanded" not in m for m in msgs)


def test_default_writer_bytes_are_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv(compact.ENV, raising=False)
    w = CassetteWriter(tmp_path / "c.jsonl")
    ls = [line(i, i + 1) for i in range(5)]
    for x in ls:
        w.append(x)
    expected = "".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in ls)
    assert (tmp_path / "c.jsonl").read_text() == expected


def _sdk_agent():
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    from openai import OpenAI

    import crashlab
    from tests.fake_llm_server import transport

    @crashlab.tool()
    def get_price(item: str) -> str:
        return f"{item}=2.5"

    @crashlab.tool(side_effects=True)
    def place_order(item: str, quantity: int) -> str:
        return f"ordered {quantity} {item}"

    fns = {"get_price": get_price, "place_order": place_order}
    tools = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
             for n in fns]  # fmt: skip

    @crashlab.agent(name="sdk-diff")
    def agent(turns: int) -> str:
        c = crashlab.wrap(OpenAI(api_key="x", base_url="http://llm.invalid/v1",
                                 http_client=httpx.Client(transport=transport())))  # fmt: skip
        msgs = [{"role": "system", "content": "shop " * 40}]
        text = ""
        for t in range(turns):
            msgs.append({"role": "user", "content": f"turn {t} " * 10})
            for _ in range(2):
                stream = t % 2 == 1
                r = c.chat.completions.create(model="m", messages=list(msgs), tools=tools,
                                              stream=stream)  # fmt: skip
                if stream:
                    content, calls = "", {}
                    for ch in r:
                        if not ch.choices:
                            continue
                        d = ch.choices[0].delta
                        content += d.content or ""
                        for tc in d.tool_calls or []:
                            e = calls.setdefault(tc.index, {"id": tc.id, "name": "", "args": ""})
                            e["name"] += (tc.function.name or "") if tc.function else ""
                            e["args"] += (tc.function.arguments or "") if tc.function else ""
                    calls = [calls[i] for i in sorted(calls)]
                else:
                    m = r.choices[0].message
                    content = m.content or ""
                    calls = [{"id": x.id, "name": x.function.name, "args": x.function.arguments}
                             for x in m.tool_calls or []]  # fmt: skip
                if not calls:
                    msgs.append({"role": "assistant", "content": content})
                    text = content
                    break
                tcs = [{"id": x["id"], "type": "function",
                        "function": {"name": x["name"], "arguments": x["args"]}}
                       for x in calls]  # fmt: skip
                msgs.append({"role": "assistant", "content": None, "tool_calls": tcs})
                with ThreadPoolExecutor(4) as ex:  # parallel tool calls in one run
                    outs = list(ex.map(lambda x: fns[x["name"]](**json.loads(x["args"])), calls))
                msgs += [{"role": "tool", "tool_call_id": x["id"], "content": o}
                         for x, o in zip(calls, outs, strict=True)]  # fmt: skip
        return text

    return agent


def test_sdk_streaming_parallel_recording_matches_full(monkeypatch, tmp_path):
    import crashlab
    from crashlab.store import Store, export_run

    pytest.importorskip("httpx")
    pytest.importorskip("openai")
    agent = _sdk_agent()
    got = {}
    for on in (False, True):
        d = tmp_path / f"s{int(on)}"
        monkeypatch.setenv("CRASHLAB_DIR", str(d))
        if on:
            monkeypatch.setenv(compact.ENV, "1")
        else:
            monkeypatch.delenv(compact.ENV, raising=False)
        out = agent(8)
        run = crashlab.load("last")
        st = Store(d)
        raw = st.cassette_path(run.run_id).read_text()
        exp = export_run(st, run.run_id, tmp_path / f"e{int(on)}.jsonl").read_text()
        assert ('"compact"' in raw) == on and '"compact"' not in exp
        assert crashlab.replay(run.run_id, store_dir=d, fn=agent).output == out
        chats = [(s.request, s.raw_request, s.response, s.request_hash)
                 for s in run.steps if s.operation == "chat"]  # fmt: skip
        tools = sorted(json.dumps([s.request, s.response, s.request_hash], sort_keys=True)
                       for s in run.steps if s.operation != "chat")  # fmt: skip
        got[on] = (out, chats, tools, len(raw))
    assert got[True][:3] == got[False][:3]
    assert got[True][3] < 0.7 * got[False][3]
