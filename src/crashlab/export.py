"""``crashlab export-test``: turn a recorded run into an offline pytest regression test.

The test replays the run's cassette (no keys, no network) and asserts what the
``@crashlab.check`` oracles said. With a fix, it also replays a *patched* run that passed
(a ``crashlab verify`` / ``--try`` child) with the fix re-applied, and asserts the fix is
pinned and the outcome stays ``pass``.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

from .models import Run
from .store import Store, export_run

DEFAULT_DIR = "tests/crashlab"


class ExportError(ValueError):
    pass


def _slug(run: Run) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", run.name.lower()).strip("_") or "run"
    return f"{name}_{run.run_id[:10].lower()}"


def _passing_fix_run(store: Store, run: Run, fix_id: str) -> Run | None:
    """The newest passing patched child of ``run`` for ``fix_id`` (verify, then --try)."""
    from .verify import verifications

    ids: list[str] = []
    for v in reversed(verifications(store, run.run_id)):
        if v.fix_id == fix_id:
            ids += [t.run_id for t in v.trials if t.role == "patched" and t.status == "pass"]
    for f in run.diagnosis.fixes if run.diagnosis else []:
        if f.fix_id == fix_id and f.tried and f.tried.get("outcome") == "pass":
            ids.append(f.tried["run_id"])
    for rid in ids:
        if rid:
            child = store.load_run(rid)
            if child.outcome == "pass" and child.patch is not None:
                return child
    return None


def _checks(run: Run) -> dict[str, str]:
    return {c.name: c.status for c in run.checks}


def _q(text: str) -> str:
    """A Python string literal the way ruff format writes it (double quotes unless the
    text contains double quotes and no single ones)."""
    if '"' in text and "'" not in text and "\\" not in text:
        return "'" + text + "'"
    return json.dumps(text, ensure_ascii=False)


def _render(
    run: Run,
    cassette: str,
    fixed: Run | None,
    fixed_cassette: str | None,
    verdict: str | None,
) -> str:
    """The test module, already ruff-format / ruff-check clean."""
    slug = _slug(run)
    d = run.diagnosis
    about = [f"Exported by `crashlab export-test` from run {run.run_id} ({run.name})."]
    if d is not None and d.is_failure:
        about.append(f"Diagnosis: step {d.failing_step} {d.label.value}: {d.root_cause}")
    if fixed is not None:
        about.append(f"Fix: {fixed.patch_description}")
        if verdict:
            about.append(f"Verification verdict: {verdict}")
    about.append("Offline: replays recorded cassettes; no API keys and no network needed.")
    lines = [textwrap.fill(a, 88, subsequent_indent="    ") for a in about]
    doc = "\n".join(lines).replace('"""', "'''").replace("\\", "/")
    checks = "".join(f"    {_q(k)}: {_q(v)},\n" for k, v in _checks(run).items())
    out = [
        f'"""{doc}\n"""',
        "",
        "from pathlib import Path",
        "",
        "import pytest",
        "",
        "pytestmark = pytest.mark.crashlab",
        'CASSETTES = Path(__file__).parent / "cassettes"',
        ("EXPECTED_CHECKS = {\n" + checks + "}") if checks else "EXPECTED_CHECKS = {}",
    ]
    if fixed is not None:
        fix_line = f"FIX = {_q(fixed.patch_description or '')}"
        if len(fix_line) <= 100:
            out.append(fix_line)
        else:
            out.append(f"FIX = (\n    {fix_line[6:]}  # noqa: E501\n)")
    out += [
        "",
        "",
        f"def test_{slug}_reproduces(crashlab_replay):",
        f'    run = crashlab_replay(CASSETTES / "{cassette}")',
        f"    assert run.outcome == {_q(run.outcome)}",
        "    assert {c.name: c.status for c in run.checks} == EXPECTED_CHECKS",
    ]
    if fixed is not None:
        out += [
            "",
            "",
            f"def test_{slug}_fix_keeps_passing(crashlab_replay):",
            f'    run = crashlab_replay(CASSETTES / "{fixed_cassette}")',
            "    # the fix is pinned: the recorded patch is re-applied to every replayed request",
            "    assert run.patch_description == FIX",
            '    assert run.outcome == "pass"',
            '    assert all(c.status == "pass" for c in run.checks), run.checks',
        ]
    return "\n".join(out) + "\n"


def _local_roots(store: Store) -> list[tuple[str, str]]:
    """Absolute prefixes that must not land in a committed cassette (longest first)."""
    import crashlab as _pkg

    roots = [
        (str(Path(store.root).resolve()), "$CRASHLAB_DIR"),
        (str(Path.home().resolve()), "~"),
        (str(Path.cwd().resolve()), "$PWD"),
        # editable install: .../src/crashlab/__init__.py → repo root; wheel: site-packages
        (str(Path(_pkg.__file__).resolve().parents[2]), "$CRASHLAB_REPO"),
    ]
    # longest first so /a/b wins over /a
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for local, token in sorted(roots, key=lambda x: -len(x[0])):
        if local in seen or len(local) < 2:
            continue
        seen.add(local)
        out.append((local, token))
    return out


def _export(store: Store, run_id: str, path: Path) -> Path:
    """``export_run`` with this machine's store / home / cwd / install paths replaced
    (run headers carry fix-file paths; error stack traces carry local files), so the
    cassette can be committed."""
    out = export_run(store, run_id, path)
    text = out.read_text("utf-8")
    for local, token in _local_roots(store):
        text = text.replace(json.dumps(local)[1:-1], token)
        text = text.replace(local, token)  # raw paths inside stack traces / messages
    out.write_text(text, "utf-8")
    return out


def export_test(
    run: Run | str = "last",
    *,
    fix: str | None = None,
    out_dir: str | Path = DEFAULT_DIR,
    store_dir: Any = None,
) -> list[Path]:
    """Write ``test_<agent>_<id>.py`` and its cassettes under ``out_dir``; returns the
    written paths. ``fix``: a fix id on the run's diagnosis (``"fix-1"``) or ``"auto"``
    (the fix of the newest VERIFIED/LIKELY verification, if any)."""
    from .verify import verifications

    store = Store(store_dir)
    run = store.load_run(run.run_id if isinstance(run, Run) else run)
    if run.forked_from or run.replay_of:
        raise ExportError(f"{run.run_id} is a fork/replay; export the original run")
    vs = verifications(store, run.run_id)
    if fix == "auto":
        good = [v for v in vs if v.verdict in ("VERIFIED", "LIKELY")]
        fix = good[-1].fix_id if good else None
    fixed = None
    verdict = None
    if fix:
        fixed = _passing_fix_run(store, run, fix)
        if fixed is None:
            raise ExportError(
                f"no passing patched run for {fix} on {run.run_id}: run "
                f"`crashlab verify {run.run_id[:10]} --fix {fix}` (or `fix --try`) first"
            )
        verdict = next((v.verdict for v in reversed(vs) if v.fix_id == fix), None)
    out = Path(out_dir)
    slug = _slug(run)
    cass = out / "cassettes" / f"{slug}.jsonl"
    written = [_export(store, run.run_id, cass)]
    fixed_name = None
    if fixed is not None:
        fixed_name = f"{slug}_fixed.jsonl"
        written.append(_export(store, fixed.run_id, out / "cassettes" / fixed_name))
    test = out / f"test_{slug}.py"
    test.write_text(_render(run, cass.name, fixed, fixed_name, verdict), "utf-8")
    written.append(test)
    return written
