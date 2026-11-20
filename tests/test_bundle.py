"""Shareable debug bundles and deep secret / PII audits."""

from __future__ import annotations

import json
import re
import zipfile
from datetime import timedelta

from typer.testing import CliRunner

from crashlab.bundle import (
    BundleError,
    audit_secrets,
    export_bundle,
    import_bundle,
    redact_run,
)
from crashlab.cli import app
from crashlab.models import Finding, Run, Usage
from crashlab.store import Store
from tests.debug_fixtures import T0, chat, save, tool

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def _tok(*parts: str) -> str:
    """Build a synthetic token at runtime so the source never holds a whole value."""
    return "".join(parts)


def _openai() -> str:
    return _tok("sk-", "test", "A" * 24)


def _anthropic() -> str:
    return _tok("sk-ant-", "api03-", "B" * 24)


def _ghp() -> str:
    return _tok("ghp_", "C" * 24)


def _gh_pat() -> str:
    return _tok("github_pat_", "D" * 24)


def _aws() -> str:
    return _tok("AKIA", "E" * 16)


def _slack() -> str:
    return _tok("xoxb-", "F" * 16)


def _google() -> str:
    return _tok("AIza", "G" * 35)


def _bearer() -> str:
    return _tok("Bearer ", "H" * 20)


def _jwt() -> str:
    return _tok("eyJ", "hbGciOiJIUzI1NiJ9", ".", "eyJ", "zdWIiOiJ4In0", ".", "I" * 20)


def _pem() -> str:
    return "-----BEGIN PRIVATE KEY-----\n" + _tok("MIIB", "J" * 40) + "\n-----END PRIVATE KEY-----"


def _email() -> str:
    return _tok("ada.lovelace", "@", "example.com")


def _phone() -> str:
    return _tok("+1-", "415-", "555-", "2671")


def _disable_store_redactor(monkeypatch) -> None:
    """Let tests plant synthetic secrets in the store (the live redactor would strip them)."""
    monkeypatch.setattr("crashlab.store.redact.Redactor.redact", lambda self, obj: obj)
    monkeypatch.setattr("crashlab.store.redact.Redactor.redact_str", lambda self, value: value)


def _leaky_run(*, child: bool = False, parent_id: str | None = None) -> tuple[Run, list]:
    """An in-memory run with one planted secret in each scanned region."""
    rid = "01BUNDLECHILD00000000000001" if child else "01BUNDLEROOT000000000000001"
    openai, anthropic, ghp, pat = _openai(), _anthropic(), _ghp(), _gh_pat()
    aws, slack, google = _aws(), _slack(), _google()
    bearer, jwt, pem = _bearer(), _jwt(), _pem()
    email, phone = _email(), _phone()
    user = f"refund with key {openai}"
    first = chat(rid, 0, user, tool_call=("get_order", {"order_id": "A17", "token": ghp}))
    mid = tool(
        rid,
        1,
        "get_order",
        {"order_id": "A17"},
        {"status": "delivered", "auth": anthropic, "note": email},
    )
    second = chat(rid, 2, None, f"done, pem follows:\n{pem}")
    second.request["system"] = f"Authorization: {bearer}"
    run = Run(
        run_id=rid,
        name="refund-bot" if not child else "lookup-agent",
        status="ok",
        outcome="fail",
        outcome_reason=f"policy token {pat}",
        started_at=T0,
        ended_at=T0 + timedelta(seconds=2),
        input={"user_msg": user, "aws": aws},
        output={"text": "refunded", "slack": slack, "jwt": jwt},
        env={"google": google, "phone": phone},
        findings=[
            Finding(kind="budget", severity="warn", step=1, message=f"cost spike near {jwt}")
        ],
        usage=Usage(wall_time_ms=2000, input_tokens=20, output_tokens=8, cost_usd=0.001),
    )
    if child and parent_id:
        run.env = {**run.env, "link": {"parent_run": parent_id, "trace": parent_id}}
    steps = [first, mid, second]
    return run, steps


def _assert_no_leak(haystack: str, *secrets: str) -> None:
    for secret in secrets:
        assert secret not in haystack


def test_audit_secrets_finds_each_kind_without_leaking_values():
    run, steps = _leaky_run()
    run.steps = steps
    secrets = (
        _openai(),
        _anthropic(),
        _ghp(),
        _gh_pat(),
        _aws(),
        _slack(),
        _google(),
        _bearer(),
        _jwt(),
        _pem(),
        _email(),
        _phone(),
    )
    report = audit_secrets(run)
    kinds = {f.kind for f in report.findings}
    assert "openai_api_key" in kinds
    assert "anthropic_api_key" in kinds
    assert "github_token" in kinds
    assert "aws_access_key" in kinds
    assert "slack_token" in kinds
    assert "google_api_key" in kinds
    assert "bearer_token" in kinds
    assert "jwt" in kinds
    assert "private_key" in kinds
    assert "email" not in kinds
    assert "phone" not in kinds
    assert report.ok is False
    dumped = json.dumps(report.to_dict())
    _assert_no_leak(dumped, *secrets)
    _assert_no_leak(repr(report), *secrets)
    for item in report.findings:
        assert item.redacted_preview == f"[REDACTED:{item.kind}]"
        assert item.kind in item.redacted_preview
        _assert_no_leak(item.redacted_preview, *secrets)

    pii = audit_secrets(run, include_pii=True)
    pii_kinds = {f.kind for f in pii.findings}
    assert "email" in pii_kinds
    assert "phone" in pii_kinds
    _assert_no_leak(json.dumps(pii.to_dict()), *secrets)

    openai_hits = [f for f in report.findings if f.kind == "openai_api_key"]
    assert any(f.field_path.startswith("input.") for f in openai_hits)
    assert any(f.step_idx == 0 and "request" in f.field_path for f in openai_hits)


def test_redact_run_replaces_secrets_and_counts_them():
    run, steps = _leaky_run()
    run.steps = steps
    before = audit_secrets(run, include_pii=True)
    redacted, count = redact_run(run, include_pii=True)
    assert count == before.count
    assert count > 0
    assert audit_secrets(redacted, include_pii=True).ok
    blob = json.dumps(redacted.model_dump(mode="json"))
    for kind in {
        "openai_api_key",
        "anthropic_api_key",
        "github_token",
        "aws_access_key",
        "slack_token",
        "google_api_key",
        "bearer_token",
        "jwt",
        "private_key",
        "email",
        "phone",
    }:
        assert f"[REDACTED:{kind}]" in blob
    _assert_no_leak(
        blob,
        _openai(),
        _anthropic(),
        _ghp(),
        _gh_pat(),
        _aws(),
        _slack(),
        _google(),
        _bearer(),
        _jwt(),
        _pem(),
        _email(),
        _phone(),
    )
    # original is unchanged
    assert _openai() in json.dumps(run.model_dump(mode="json"))


def test_redact_run_in_place_requires_store_and_rewrites_cassette(crashlab_dir, monkeypatch):
    _disable_store_redactor(monkeypatch)
    store = Store(crashlab_dir)
    run, steps = _leaky_run()
    save(store, run, steps)
    loaded = store.load_run(run.run_id)
    assert _openai() in json.dumps(loaded.model_dump(mode="json"))

    try:
        redact_run(loaded, in_place=True)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "store" in str(exc)

    redacted, count = redact_run(loaded, store=store, in_place=True)
    assert count > 0
    again = store.load_run(run.run_id)
    blob = json.dumps(again.model_dump(mode="json"))
    assert _openai() not in blob
    assert "[REDACTED:openai_api_key]" in blob
    assert redacted.run_id == run.run_id


def test_already_redacted_placeholders_are_not_flagged_again():
    rid = "01BUNDLECLEAN00000000000001"
    run = Run(run_id=rid, name="clean", status="ok", outcome="pass", started_at=T0)
    run.input = {"user_msg": "key [REDACTED:openai_api_key] already gone"}
    report = audit_secrets(run)
    assert report.ok
    redacted, count = redact_run(run)
    assert count == 0
    assert redacted.input["user_msg"] == run.input["user_msg"]


def test_bundle_round_trip_includes_children_and_html(crashlab_dir, monkeypatch, tmp_path):
    _disable_store_redactor(monkeypatch)
    store = Store(crashlab_dir)
    parent, parent_steps = _leaky_run()
    save(store, parent, parent_steps)
    child, child_steps = _leaky_run(child=True, parent_id=parent.run_id)
    save(store, child, child_steps)

    dest = tmp_path / "debug.crashlab.zip"
    manifest = export_bundle(store, parent.run_id, dest, redact=True, include_html=True)
    assert dest.is_file()
    assert manifest.root_run_id == parent.run_id
    assert parent.run_id in manifest.run_ids
    assert child.run_id in manifest.run_ids
    assert manifest.redacted is True
    assert manifest.include_html is True
    assert manifest.audit["count"] > 0

    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "diagnosis.json" in names
        assert "context.json" in names
        assert "report.html" in names
        assert f"runs/{parent.run_id}.jsonl" in names
        assert f"runs/{child.run_id}.jsonl" in names
        raw = zf.read(f"runs/{parent.run_id}.jsonl").decode("utf-8")
        html = zf.read("report.html").decode("utf-8")
        ctx = json.loads(zf.read("context.json").decode("utf-8"))
        diag = json.loads(zf.read("diagnosis.json").decode("utf-8"))
        meta = json.loads(zf.read("manifest.json").decode("utf-8"))
    secrets = (
        _openai(),
        _anthropic(),
        _ghp(),
        _gh_pat(),
        _aws(),
        _slack(),
        _google(),
        _jwt(),
        _pem(),
    )
    archive = dest.read_bytes().decode("latin-1")
    _assert_no_leak(raw, *secrets)
    _assert_no_leak(html, *secrets)
    _assert_no_leak(json.dumps(ctx), *secrets)
    _assert_no_leak(json.dumps(diag), *secrets)
    _assert_no_leak(json.dumps(meta), *secrets)
    _assert_no_leak(archive, *secrets)
    assert "[REDACTED:openai_api_key]" in raw
    assert "crashlab report" in html.lower() or parent.name in html
    assert ctx["root_run_id"] == parent.run_id
    assert "context" in ctx and "tool_audit" in ctx
    assert diag["root_run_id"] == parent.run_id

    fresh = Store(tmp_path / "fresh")
    imported = import_bundle(fresh, dest)
    assert parent.run_id in imported
    assert child.run_id in imported
    loaded = fresh.load_run(parent.run_id)
    child_loaded = fresh.load_run(child.run_id)
    assert child_loaded.parent_run == parent.run_id
    blob = json.dumps(loaded.model_dump(mode="json"))
    _assert_no_leak(blob, *secrets)
    assert "[REDACTED:openai_api_key]" in blob
    assert len(loaded.steps) == 3


def test_bundle_cli_help_is_ansi_stripped_and_fail_on_match(crashlab_dir, monkeypatch, tmp_path):
    for command in ("bundle", "unbundle", "audit-secrets", "redact-run"):
        res = runner.invoke(app, [command, "--help"])
        assert res.exit_code == 0, res.output
        clean = _plain(res.output)
        assert "\x1b[" not in clean
        if command == "bundle":
            assert "--no-redact" in clean
            assert "--no-html" in clean
            assert "-o" in clean or "--output" in clean
        elif command == "unbundle":
            assert "BUNDLE" in clean.upper() or "bundle" in clean.lower()
        elif command == "audit-secrets":
            assert "--include-pii" in clean
            assert "--json" in clean
            assert "--fail-on-match" in clean
            assert "--all" in clean
        else:
            assert "--include-pii" in clean
            assert "--in-place" in clean

    monkeypatch.setenv("COLUMNS", "160")
    _disable_store_redactor(monkeypatch)
    store = Store(crashlab_dir)
    run, steps = _leaky_run()
    save(store, run, steps)

    failed = runner.invoke(app, ["audit-secrets", run.run_id, "--fail-on-match"])
    assert failed.exit_code == 1, failed.output
    clean = _plain(failed.output)
    _assert_no_leak(clean, _openai(), _anthropic(), _jwt(), _pem())
    assert "openai_api_key" in clean
    assert "REDACTED" in clean

    as_json = runner.invoke(app, ["audit-secrets", run.run_id, "--json"])
    assert as_json.exit_code == 0, as_json.output
    payload = json.loads(as_json.output)
    assert payload["count"] > 0
    _assert_no_leak(as_json.output, _openai(), _jwt(), _pem())

    dest = tmp_path / "out.crashlab.zip"
    bundled = runner.invoke(app, ["bundle", run.run_id, "-o", str(dest)])
    assert bundled.exit_code == 0, bundled.output
    assert dest.is_file()
    with zipfile.ZipFile(dest) as zf:
        assert "report.html" in zf.namelist()

    no_html = runner.invoke(
        app, ["bundle", run.run_id, "-o", str(tmp_path / "plain.crashlab.zip"), "--no-html"]
    )
    assert no_html.exit_code == 0, no_html.output
    with zipfile.ZipFile(tmp_path / "plain.crashlab.zip") as zf:
        assert "report.html" not in zf.namelist()
        assert "manifest.json" in zf.namelist()

    fresh = tmp_path / "other-store"
    unbundled = runner.invoke(
        app,
        ["--dir", str(fresh), "unbundle", str(dest)],
    )
    assert unbundled.exit_code == 0, unbundled.output
    assert run.run_id in unbundled.output
    imported = Store(fresh).load_run(run.run_id)
    assert "[REDACTED:openai_api_key]" in json.dumps(imported.model_dump(mode="json"))

    via_import = runner.invoke(
        app,
        ["--dir", str(tmp_path / "via-import"), "import", str(dest)],
    )
    assert via_import.exit_code == 0, via_import.output
    assert run.run_id in via_import.output

    # `--dir` writes CRASHLAB_DIR on the process; point later commands back
    # at the original leaky store.
    monkeypatch.setenv("CRASHLAB_DIR", str(crashlab_dir))
    dry = runner.invoke(app, ["redact-run", run.run_id])
    assert dry.exit_code == 0, dry.output
    assert "would redact" in _plain(dry.output)
    still = Store(crashlab_dir).load_run(run.run_id)
    assert _openai() in json.dumps(still.model_dump(mode="json"))

    written = runner.invoke(app, ["redact-run", run.run_id, "--in-place"])
    assert written.exit_code == 0, written.output
    assert "redacted in place" in _plain(written.output)
    cleaned = Store(crashlab_dir).load_run(run.run_id)
    blob = json.dumps(cleaned.model_dump(mode="json"))
    assert _openai() not in blob
    assert "[REDACTED:openai_api_key]" in blob

    clean_run, clean_steps = _leaky_run()
    # a second, already-clean run for --fail-on-match exit 0
    clean_run.run_id = "01BUNDLESAFE00000000000001"
    for step in clean_steps:
        step.run_id = clean_run.run_id
    clean_run.input = {"user_msg": "Refund my order A17"}
    clean_run.output = {"text": "ok"}
    clean_run.env = {}
    clean_run.findings = []
    clean_run.outcome_reason = None
    clean_steps[0] = chat(clean_run.run_id, 0, "Refund my order A17", "ok")
    clean_steps[1] = tool(clean_run.run_id, 1, "get_order", {"order_id": "A17"}, {"ok": True})
    clean_steps[2] = chat(clean_run.run_id, 2, None, "done")
    save(store, clean_run, clean_steps)
    ok = runner.invoke(app, ["audit-secrets", clean_run.run_id, "--fail-on-match"])
    assert ok.exit_code == 0, ok.output
    assert "no secrets found" in _plain(ok.output)


def test_import_bundle_rejects_a_non_bundle(tmp_path, crashlab_dir):
    store = Store(crashlab_dir)
    empty = tmp_path / "empty.crashlab.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("readme.txt", "not a bundle")
    try:
        import_bundle(store, empty)
        raise AssertionError("expected BundleError")
    except BundleError as exc:
        assert "not a crashlab bundle" in str(exc)
    missing = runner.invoke(app, ["unbundle", str(tmp_path / "nope.zip")])
    assert missing.exit_code == 2
