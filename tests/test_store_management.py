"""Store tagging, deletion, prune, and the tag / rm / prune / ls --tag CLI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from crashlab.cli import app
from crashlab.models import Diagnosis, Finding
from crashlab.store import PruneReport, RunNotFoundError, Store, norm_tag
from tests.debug_fixtures import make_run, run_id

runner = CliRunner()
ENV = {"COLUMNS": "220", "TERM": "xterm-256color", "NO_COLOR": "1"}


def test_norm_tag_rejects_empty_and_odd_names():
    assert norm_tag("  prod  ") == "prod"
    assert norm_tag("verify:abc") == "verify:abc"
    with pytest.raises(ValueError):
        norm_tag("")
    with pytest.raises(ValueError):
        norm_tag("  ")
    with pytest.raises(ValueError):
        norm_tag("+prod")
    with pytest.raises(ValueError):
        norm_tag("has space")


def test_tag_run_add_remove_dedupes_and_preserves_order(crashlab_dir):
    store = Store()
    run = make_run(store, 1)
    assert store.tag_run(run.run_id) == []
    assert store.tag_run(run.run_id, add=["prod", "ci", "prod"]) == ["prod", "ci"]
    assert store.tag_run(run.run_id, add=["nightly"], remove=["ci"]) == ["prod", "nightly"]
    loaded = store.load_run(run.run_id)
    assert loaded.tags == ["prod", "nightly"]
    data = json.loads((Path(store.root) / "runs" / run.run_id / "run.json").read_text())
    assert data["tags"] == ["prod", "nightly"]
    row = store.list_runs(limit=1)[0]
    assert json.loads(row["tags_json"]) == ["prod", "nightly"]
    # prefix works
    assert store.tag_run(run.run_id[:12], add=["keep"]) == ["prod", "nightly", "keep"]


def test_list_runs_filters_by_tag(crashlab_dir):
    store = Store()
    a = make_run(store, 1, name="a")
    b = make_run(store, 2, name="b")
    make_run(store, 3, name="c")
    store.tag_run(a.run_id, add=["prod"])
    store.tag_run(b.run_id, add=["prod", "nightly"])
    rows = store.list_runs(limit=None, tag="prod")
    assert {r["name"] for r in rows} == {"a", "b"}
    assert [r["name"] for r in store.list_runs(limit=None, tag="nightly")] == ["b"]
    assert store.list_runs(limit=None, tag="missing") == []


def _index_counts(store: Store, run_id: str) -> tuple[int, int, int, int]:
    idx = store.index
    runs = idx.get_run_json(run_id) is not None
    return (
        int(runs),
        len(idx.steps(run_id)),
        len(idx.flags(run_id)),
        len(idx.diagnoses(run_id)),
    )


def test_delete_run_removes_directory_and_sqlite_rows(crashlab_dir):
    store = Store()
    run = make_run(
        store,
        1,
        findings=[Finding(kind="budget", severity="warn", message="over", step=0)],
    )
    run.diagnosis = Diagnosis(run_id=run.run_id, label="NO_FAILURE", root_cause="none")
    store.save_diagnosis(run, run.diagnosis, cache_key="k")
    rid = run.run_id
    path = store.run_dir(rid)
    assert path.is_dir() and _index_counts(store, rid)[0] == 1
    assert _index_counts(store, rid)[1] >= 1
    assert _index_counts(store, rid)[2] >= 1
    assert _index_counts(store, rid)[3] >= 1
    store.delete_run(rid)
    assert not path.exists()
    assert _index_counts(store, rid) == (0, 0, 0, 0)
    with pytest.raises(RunNotFoundError):
        store.load_run(rid)
    with pytest.raises(RunNotFoundError):
        store.delete_run(rid)


def test_delete_run_refuses_path_traversal(crashlab_dir):
    store = Store()
    make_run(store, 1)
    for bad in ("../x", "..", "/tmp/crashlab-nope", "runs/../../etc"):
        with pytest.raises(ValueError, match="not a valid run id"):
            store.delete_run(bad)
    # the store itself is untouched
    assert store.list_runs() and (store.root / "index.sqlite").exists()


def _aged(store: Store, n: int, *, hours: float, **kw):
    return make_run(store, n, hours_ago=hours, **kw)


def test_prune_dry_run_default_and_older_than(crashlab_dir):
    store = Store()
    old = _aged(store, 1, hours=48, name="old")
    new = _aged(store, 2, hours=1, name="new")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    report = store.prune(older_than=cutoff)
    assert isinstance(report, PruneReport)
    assert report.deleted == []
    assert report.would_delete == [old.run_id]
    assert report.kept == 1
    assert store.load_run(old.run_id).run_id == old.run_id  # still there
    did = store.prune(older_than=cutoff, dry_run=False)
    assert did.deleted == [old.run_id] and did.would_delete == []
    assert did.freed_bytes > 0
    with pytest.raises(RunNotFoundError):
        store.load_run(old.run_id)
    assert store.load_run(new.run_id).name == "new"


def test_prune_keeps_failed_and_tagged_unless_asked(crashlab_dir):
    store = Store()
    old_ok = _aged(store, 1, hours=48, name="ok")
    old_fail = _aged(store, 2, hours=48, name="fail", status="error", outcome="fail")
    old_tag = _aged(store, 3, hours=48, name="tagged")
    store.tag_run(old_tag.run_id, add=["keep"])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    report = store.prune(older_than=cutoff, dry_run=False)
    assert report.deleted == [old_ok.run_id]
    assert {store.load_run(old_fail.run_id).name, store.load_run(old_tag.run_id).name} == {
        "fail",
        "tagged",
    }
    assert report.kept_failed == 1 and report.kept_tagged == 1
    gone = store.prune(older_than=cutoff, keep_failed=False, keep_tagged=False, dry_run=False)
    assert set(gone.deleted) == {old_fail.run_id, old_tag.run_id}


def test_prune_max_runs_and_max_bytes(crashlab_dir):
    store = Store()
    runs = [make_run(store, i, name=f"r{i}") for i in range(1, 6)]
    report = store.prune(max_runs=2, dry_run=False)
    kept = [r["run_id"] for r in store.list_runs(limit=None)]
    # newest first: r5, r4 kept
    assert kept == [runs[4].run_id, runs[3].run_id]
    assert set(report.deleted) == {runs[0].run_id, runs[1].run_id, runs[2].run_id}

    # fill the store again and cap by size: a tiny budget drops every unprotected run
    more = make_run(store, 10, name="extra")
    tiny = store.prune(max_bytes=1, dry_run=False)
    assert more.run_id in tiny.deleted
    # tagged newest is kept even when over the byte cap
    keep = make_run(store, 11, name="keep-me")
    store.tag_run(keep.run_id, add=["prod"])
    report = store.prune(max_bytes=1, dry_run=False)
    assert keep.run_id not in report.deleted
    assert store.load_run(keep.run_id).tags == ["prod"]


def test_cli_tag_rm_prune_and_ls_tag(crashlab_dir):
    store = Store()
    a = make_run(store, 1, name="alpha")
    b = make_run(store, 2, name="beta")
    c = make_run(store, 3, name="gamma", status="error", outcome="fail")

    res = runner.invoke(app, ["tag", a.run_id], env=ENV)
    assert res.exit_code == 0 and "no tags" in res.output
    res = runner.invoke(app, ["tag", a.run_id, "+prod", "+ci"], env=ENV)
    assert res.exit_code == 0 and "prod, ci" in res.output
    res = runner.invoke(app, ["tag", a.run_id, "-ci", "--json"], env=ENV)
    assert res.exit_code == 0
    body = json.loads(res.output)
    assert body["run_id"] == a.run_id and body["tags"] == ["prod"]

    res = runner.invoke(app, ["ls", "--tag", "prod"], env=ENV)
    assert res.exit_code == 0 and a.run_id in res.output and b.run_id not in res.output
    res = runner.invoke(app, ["ls", "--tag", "nope"], env=ENV)
    assert res.exit_code == 0 and "No runs tagged 'nope'" in res.output

    res = runner.invoke(app, ["rm", b.run_id], env=ENV)
    assert res.exit_code == 1 and "Pass --yes" in res.output
    assert store.load_run(b.run_id).name == "beta"
    res = runner.invoke(app, ["rm", b.run_id, "-y"], env=ENV)
    assert res.exit_code == 0 and "deleted" in res.output
    with pytest.raises(RunNotFoundError):
        store.load_run(b.run_id)

    res = runner.invoke(app, ["prune"], env=ENV)
    assert res.exit_code == 2 and "needs --older-than" in res.output
    res = runner.invoke(app, ["prune", "--max-runs", "0", "--include-failed", "--json"], env=ENV)
    assert res.exit_code == 0
    preview = json.loads(res.output)
    # --include-failed allows deleting gamma; alpha is still tagged, so it is kept
    assert preview["would_delete"] == [c.run_id]
    assert preview["deleted"] == []
    assert store.load_run(a.run_id).tags == ["prod"]  # dry-run
    res = runner.invoke(app, ["prune", "--max-runs", "0", "--yes", "--json"], env=ENV)
    assert res.exit_code == 0
    done = json.loads(res.output)
    # tagged + failed kept by default
    assert done["deleted"] == []
    assert store.load_run(a.run_id) and store.load_run(c.run_id)
    res = runner.invoke(
        app,
        ["prune", "--max-runs", "0", "--include-tagged", "--include-failed", "--yes", "--json"],
        env=ENV,
    )
    assert res.exit_code == 0
    gone = json.loads(res.output)
    assert set(gone["deleted"]) == {a.run_id, c.run_id}
    assert store.list_runs(limit=None) == []


def test_cli_tag_rejects_bare_names_and_rm_missing(crashlab_dir):
    store = Store()
    run = make_run(store, 1)
    res = runner.invoke(app, ["tag", run.run_id, "prod"], env=ENV)
    assert res.exit_code != 0 and "+name" in res.output
    res = runner.invoke(app, ["rm", "-y", "NOPE"], env=ENV)
    assert res.exit_code == 1 and "no run" in res.output.lower()


def test_run_ids_in_fixtures_are_safe(crashlab_dir):
    # sanity: debug fixture ids pass the store's path-component rule
    store = Store()
    rid = run_id(1)
    assert rid.startswith("01FIXTURE")
    make_run(store, 1)
    store.delete_run(rid)
    assert store.list_runs() == []
