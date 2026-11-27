"""Prompt version tracking and A/B comparison (crashlab.prompts / crashlab prompts)."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

import crashlab
from crashlab import prompts
from crashlab.cli import app
from crashlab.prompts import (
    PromptError,
    compare_prompts,
    fisher_exact,
    list_prompts,
    mann_whitney,
    normalize_prompt,
    prompt_id,
    step_prompt,
    two_proportion_z,
)
from crashlab.store import Store
from tests.debug_fixtures import chat, make_run
from tests.fakes import FakeOpenAI, text_response

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
ENV = {"COLUMNS": "200", "NO_COLOR": "1"}

V1 = "You are a refund assistant. Never refund twice."
V2 = "You are a refund assistant. Never refund twice.\nAlways confirm the order id first."


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _cli(*args: str):
    return runner.invoke(app, list(args), env=ENV)


_n = [0]


def _add(
    store: Store, system: str | None, *, outcome="pass", status="ok", cost=0.0001, name="bot",
    label=None,
):  # fmt: skip
    _n[0] += 1
    n = _n[0]
    rid = f"r{n:03d}"
    step = chat(rid, 0, "hi", "ok", cost=cost)
    req = dict(step.request)
    if system is None:
        req.pop("system", None)
    else:
        req["system"] = system
    attrs = {**step.attributes, **({"crashlab.prompt.label": label} if label else {})}
    step = step.model_copy(update={"request": req, "attributes": attrs})
    return make_run(store, n, name, status=status, outcome=outcome, steps=[step])


def _fill(store: Store, system: str, passes: int, fails: int, **kw) -> None:
    for _ in range(passes):
        _add(store, system, **kw)
    for _ in range(fails):
        _add(store, system, outcome="fail", **kw)


# --- fingerprint -------------------------------------------------------------------------


def test_prompt_id_shape_and_stability():
    pid = prompt_id(V1)
    assert re.fullmatch(r"p-[0-9a-f]{6}", pid)
    assert prompt_id(V1) == pid == prompt_id(V1)
    assert prompt_id(V2) != pid
    assert prompt_id(None) is None and prompt_id("  \n ") is None


def test_whitespace_placeholders_and_volatile_values_do_not_change_the_id():
    base = (
        "Help {user_name} with {{ topic }}. Today is 2026-11-27. "
        "Session 123e4567-e89b-12d3-a456-426614174000."
    )
    same = (
        "Help   ${who}   with {topic}.\n\n Today is 2027-01-02T10:00:00Z. "
        "Session 00000000-0000-0000-0000-000000000000."
    )
    assert prompt_id(base) == prompt_id(same)
    assert prompt_id(base) != prompt_id(base.replace("Help", "Assist"))
    assert normalize_prompt("a {x}  b") == "a {} b"
    assert normalize_prompt("price is {{ p }} or %(q)s") == "price is {} or {}"


# --- record time -------------------------------------------------------------------------


def _call(client, system=V1):
    client.chat.completions.create(
        model="m",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": "yo"}],
    )


def test_chat_steps_are_stamped_and_labelled(crashlab_dir):
    client = crashlab.wrap(FakeOpenAI([text_response(i, "hi") for i in range(1, 6)]))
    with crashlab.record("a"):
        _call(client)
    with crashlab.record("a"):
        with crashlab.prompt_version("support-v3", templates={"planner": "Plan {steps}"}):
            _call(client)
            with crashlab.prompt_version("inner"):
                _call(client)
    with crashlab.record("a", prompt_version="run-label"):
        _call(client)
    with crashlab.record("a"):  # no system prompt: nothing to fingerprint
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    store = Store()
    runs = [store.load_run(r["run_id"]) for r in reversed(store.list_runs(limit=None))]
    a = runs[0].steps[0].attributes
    assert a["crashlab.prompt.id"] == prompt_id(V1) and "crashlab.prompt.label" not in a
    b0, b1 = (s.attributes for s in runs[1].steps)
    assert b0["crashlab.prompt.label"] == "support-v3"
    assert b0["crashlab.prompt.templates"] == {"planner": prompt_id("Plan {steps}")}
    assert b1["crashlab.prompt.label"] == "inner"
    assert b1["crashlab.prompt.templates"] == b0["crashlab.prompt.templates"]
    assert runs[2].steps[0].attributes["crashlab.prompt.label"] == "run-label"
    assert runs[2].env["prompt_version"] == "run-label"
    assert "crashlab.prompt.id" not in runs[3].steps[0].attributes
    assert step_prompt(runs[3].steps[0]) is None


def test_label_validation():
    with pytest.raises(ValueError):
        with crashlab.prompt_version(""):
            pass
    with pytest.raises(ValueError):
        with crashlab.prompt_version():
            pass
    with pytest.raises(ValueError):
        crashlab.record("x", prompt_version=" ")


def test_legacy_steps_are_fingerprinted_lazily(crashlab_dir):
    store = Store()
    run = _add(store, V1)
    step = store.load_run(run.run_id).steps[0]
    assert "crashlab.prompt.id" not in step.attributes
    info = step_prompt(step)
    assert info == {"id": prompt_id(V1), "label": None, "text": V1}


# --- statistics --------------------------------------------------------------------------


def test_fisher_exact_known_values():
    assert fisher_exact(8, 10, 2, 10) == pytest.approx(0.02301, abs=1e-5)
    assert fisher_exact(3, 4, 1, 4) == pytest.approx(0.4857, abs=1e-4)
    assert fisher_exact(5, 10, 5, 10) == pytest.approx(1.0)
    assert fisher_exact(0, 0, 1, 2) == 1.0


def test_two_proportion_z_known_values():
    z, p = two_proportion_z(45, 100, 60, 100)
    assert z == pytest.approx(2.1245, abs=1e-3)
    assert p == pytest.approx(0.0336, abs=1e-3)
    assert two_proportion_z(10, 10, 10, 10) == (0.0, 1.0)
    assert two_proportion_z(0, 0, 1, 2) == (0.0, 1.0)


def test_mann_whitney_exact_and_normal():
    u, p = mann_whitney([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert u == 0 and p == pytest.approx(0.007937, abs=1e-6)  # R: wilcox.test exact
    u, p = mann_whitney([6, 7, 8, 9, 10], [1, 2, 3, 4, 5])
    assert u == 25 and p == pytest.approx(0.007937, abs=1e-6)
    _, p = mann_whitney([1, 3, 5, 7], [2, 4, 6, 8])
    assert p == pytest.approx(0.6857, abs=1e-4)  # R: W = 6, p = 0.6857
    _, p = mann_whitney([1] * 5, [2] * 5)  # ties: normal approximation
    assert p == pytest.approx(0.00398, abs=2e-4)
    _, p = mann_whitney(list(range(1, 21)), list(range(21, 41)))
    assert p < 1e-6
    assert mann_whitney([1, 2], [])[1] == 1.0
    assert mann_whitney([3, 3, 3], [3, 3, 3])[1] == 1.0


# --- list / show / diff ------------------------------------------------------------------


def test_list_groups_runs_by_prompt_version(crashlab_dir):
    store = Store()
    _fill(store, V1, 3, 1)
    _fill(store, V2, 4, 0, cost=0.01)
    _add(store, None)  # a run with no system prompt
    versions, without = list_prompts(store)
    assert without == 1
    by_id = {v.id: v for v in versions}
    s1, s2 = by_id[prompt_id(V1)].summary(), by_id[prompt_id(V2)].summary()
    assert (s1["runs"], s1["pass_rate"]) == (4, 0.75)
    assert (s2["runs"], s2["pass_rate"]) == (4, 1.0)
    assert s2["avg_cost_usd"] == pytest.approx(0.01)
    assert s1["first_seen"] <= s1["last_seen"]

    res = _cli("prompts", "list", "--json")
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["runs_without_prompt"] == 1
    assert {p["id"] for p in data["prompts"]} == {prompt_id(V1), prompt_id(V2)}
    assert _cli("prompts", "list", "--since", "1h", "--json").exit_code == 0
    empty = json.loads(_cli("prompts", "list", "--agent", "nobody", "--json").stdout)
    assert empty["prompts"] == []
    text = _plain(_cli("prompts", "list").output)
    assert prompt_id(V1) in text and "75.0%" in text and "without a system prompt" in text


def test_list_empty_store(crashlab_dir):
    res = _cli("prompts", "list")
    assert res.exit_code == 0 and "no prompt versions" in _plain(res.output)


def test_show_and_diff(crashlab_dir):
    store = Store()
    _fill(store, V1, 2, 0)
    _fill(store, V2, 1, 1)
    p1, p2 = prompt_id(V1), prompt_id(V2)
    shown = _cli("prompts", "show", p1[2:])  # id without the "p-" prefix
    assert shown.exit_code == 0 and V1 in shown.output
    as_json = json.loads(_cli("prompts", "show", p2, "--json").stdout)
    assert as_json["text"] == V2 and as_json["runs"] == 2
    res = _cli("prompts", "diff", p1, p2)
    assert res.exit_code == 0, res.output
    out = _plain(res.output)
    assert "+Always confirm the order id first." in out
    assert "-50.0pp" in out
    data = json.loads(_cli("prompts", "diff", p1, p2, "--json").stdout)
    assert data["delta"]["pass_rate"] == pytest.approx(-0.5)
    assert "+Always confirm" in data["diff"]
    missing = _cli("prompts", "show", "p-ffffff")
    assert missing.exit_code == 2 and "no prompt version" in missing.output


def test_labels_select_versions(crashlab_dir):
    store = Store()
    _add(store, V1, label="v-old")
    versions, _ = list_prompts(store)
    assert versions[0].labels == ["v-old"]
    assert json.loads(_cli("prompts", "show", "v-old", "--json").stdout)["id"] == prompt_id(V1)


# --- A/B ---------------------------------------------------------------------------------


def test_ab_pass_rate_better_worse_inconclusive(crashlab_dir):
    store = Store()
    _fill(store, V1, 2, 8)  # 20%
    _fill(store, V2, 9, 1)  # 90%
    p1, p2 = prompt_id(V1), prompt_id(V2)
    res = compare_prompts(store, p1, p2)
    assert (res.n_a, res.n_b, res.verdict, res.test) == (10, 10, "better", "fisher_exact")
    assert res.p_value == pytest.approx(fisher_exact(2, 10, 9, 10))
    assert res.delta == pytest.approx(0.7)
    worse = compare_prompts(store, p2, p1)
    assert worse.verdict == "worse"

    out = _cli("prompts", "ab", p1, p2, "--json")
    assert out.exit_code == 0 and json.loads(out.stdout)["verdict"] == "better"
    bad = _cli("prompts", "ab", p2, p1, "--fail-on-regression")
    assert bad.exit_code == 1 and "WORSE" in _plain(bad.output)
    assert _cli("prompts", "ab", p2, p1).exit_code == 0

    V3 = V1 + " Be brief."
    _fill(store, V3, 3, 1)  # 75% on 4 runs against V1's 20%: too few to call
    inc = compare_prompts(store, p1, prompt_id(V3))
    assert inc.verdict == "inconclusive" and inc.low_sample and inc.note


def test_ab_large_samples_use_z_test(crashlab_dir):
    store = Store()
    _fill(store, V1, 20, 20)
    _fill(store, V2, 35, 5)
    res = compare_prompts(store, prompt_id(V1), prompt_id(V2))
    assert res.test == "two_proportion_z" and res.verdict == "better"
    assert res.p_value == pytest.approx(two_proportion_z(20, 40, 35, 40)[1])


def test_ab_cost_and_latency_use_mann_whitney(crashlab_dir):
    store = Store()
    for i in range(8):
        _add(store, V1, cost=0.01 + i * 1e-4)
        _add(store, V2, cost=0.002 + i * 1e-4)
    res = compare_prompts(store, prompt_id(V1), prompt_id(V2), metric="cost")
    assert res.test == "mann_whitney_u" and res.verdict == "better"  # cheaper is better
    assert res.value_b < res.value_a
    pricey = compare_prompts(store, prompt_id(V2), prompt_id(V1), metric="cost")
    assert pricey.verdict == "worse"
    lat = compare_prompts(store, prompt_id(V1), prompt_id(V2), metric="latency")
    assert lat.verdict == "inconclusive"  # every fixture run took the same 2s
    out = _cli("prompts", "ab", prompt_id(V1), prompt_id(V2), "-m", "cost")
    assert out.exit_code == 0 and "BETTER" in _plain(out.output)


def test_ab_judge_metric_offline(crashlab_dir):
    store = Store()
    _fill(store, V1, 6, 0)
    _fill(store, V2, 6, 0)

    def judge_fn(messages):
        text = json.dumps(messages)
        return json.dumps({"score": 0.2 if "Always confirm" in text else 0.9, "reasoning": "x"})

    res = compare_prompts(
        store, prompt_id(V1), prompt_id(V2), metric="judge:task_completion", judge_fn=judge_fn
    )
    assert res.metric == "judge:task_completion" and res.n_a == res.n_b == 6
    assert res.value_a == pytest.approx(0.9) and res.verdict in ("worse", "inconclusive")
    with pytest.raises(PromptError, match="unknown"):
        compare_prompts(store, prompt_id(V1), prompt_id(V2), metric="judge:nope", judge_fn=judge_fn)
    cli = _cli("prompts", "ab", prompt_id(V1), prompt_id(V2), "--metric", "judge:task_completion")
    assert cli.exit_code == 2  # no judge model configured: a clear error, not a traceback


def test_ab_errors_and_overlap(crashlab_dir):
    store = Store()
    _fill(store, V1, 3, 0)
    _fill(store, V2, 3, 0)
    p1, p2 = prompt_id(V1), prompt_id(V2)
    with pytest.raises(PromptError, match="no prompt version"):
        compare_prompts(store, p1, "p-000000")
    with pytest.raises(PromptError, match="same prompt version"):
        compare_prompts(store, p1, p1)
    with pytest.raises(PromptError, match="unknown metric"):
        compare_prompts(store, p1, p2, metric="vibes")
    with pytest.raises(PromptError, match="alpha"):
        compare_prompts(store, p1, p2, alpha=2)
    for args in (("p-000000",), ("--metric", "vibes")):
        res = _cli("prompts", "ab", p1, p2 if args[0].startswith("--") else args[0], *args[1:])
        assert res.exit_code == 2

    # a multi-step run that used both versions is left out of both groups
    steps = [
        chat("x", 0, "q", "a").model_copy(update={"request": {"system": V1, "messages": []}}),
        chat("x", 1, "q", "a").model_copy(update={"request": {"system": V2, "messages": []}}),
    ]
    make_run(store, 99, steps=steps)
    res = compare_prompts(store, p1, p2)
    assert res.overlap_runs == 1 and (res.n_a, res.n_b) == (3, 3)


def test_ab_agent_filter(crashlab_dir):
    store = Store()
    _fill(store, V1, 5, 0, name="alpha")
    _fill(store, V2, 5, 0, name="alpha")
    _fill(store, V1, 0, 5, name="beta")
    p1, p2 = prompt_id(V1), prompt_id(V2)
    assert compare_prompts(store, p1, p2).n_a == 10
    assert compare_prompts(store, p1, p2, agent="alpha").n_a == 5
    assert compare_prompts(store, p1, p2, agent="alpha").verdict == "inconclusive"


# --- CLI help ----------------------------------------------------------------------------


def test_help_lists_prompts_under_evaluate():
    top = _plain(_cli("--help").output)
    assert "prompts" in top
    res = _cli("prompts", "--help")
    out = _plain(res.output)
    assert res.exit_code == 0
    assert "A/B compare" in out and "Typer" not in out
    for sub in ("list", "show", "diff", "ab"):
        assert sub in out
    ab = _plain(_cli("prompts", "ab", "--help").output)
    for flag in ("--metric", "--json", "--fail-on-regression", "--alpha"):
        assert flag in ab


def test_public_api():
    assert crashlab.prompt_id is prompts.prompt_id
    assert crashlab.prompt_version is prompts.prompt_version
