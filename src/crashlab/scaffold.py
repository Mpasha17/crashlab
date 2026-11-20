"""Project scaffolding: ``crashlab init`` / :func:`init_project`."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["GITIGNORE_ENTRY", "InitResult", "init_project"]

GITIGNORE_ENTRY = ".crashlab/"
_SUPPORTED_CI = ("github", "none")

_WORKFLOW = """\
# Offline crashlab eval: grade exported cassettes (or the store) with JUnit + Markdown.
# Nothing here calls a model unless you add `--replay` (still offline) or a live sweep.
name: crashlab eval

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install crashlab
        run: pip install crashlab
      - name: Grade recorded runs
        run: |
          crashlab eval --checks crashlab_checks.py \\
            --junit eval-results.xml --markdown eval-summary.md
        # GITHUB_STEP_SUMMARY is set by Actions; crashlab eval appends the Markdown there.
"""

_CHECKS = '''\
"""Starter @crashlab.check oracles for this project.

Import this module (or pass it to `crashlab eval --checks crashlab_checks.py`) so the
oracles are registered before grading. A check passes when it returns True or None, and
fails when it returns False, a non-empty string, or raises AssertionError.
"""

from __future__ import annotations

import crashlab
from crashlab.models import Run


@crashlab.check
def run_did_not_error(run: Run) -> str | None:
    """The recorded run ended without a process-level error."""
    if run.status == "error":
        err = run.error.message if run.error is not None else "unknown error"
        return f"run ended with status=error: {err}"
    return None


@crashlab.check
def no_critical_findings(run: Run) -> str | None:
    """No runtime check fired at critical severity."""
    hits = run.findings_of(min_severity="critical")
    if not hits:
        return None
    kinds = ", ".join(sorted({f.kind for f in hits}))
    return f"{len(hits)} critical finding(s): {kinds}"
'''

_GITIGNORE = """\
# crashlab local run store
.crashlab/
"""


@dataclass
class InitResult:
    """What :func:`init_project` wrote, skipped, or noted."""

    target_dir: str
    created: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def init_project(
    target_dir: str | os.PathLike[str],
    *,
    ci: str = "github",
    checks: bool = True,
    force: bool = False,
) -> InitResult:
    """Write a ready-to-use crashlab eval workflow and starter check suite.

    Existing files are left alone unless ``force=True``. ``ci="github"`` writes
    ``.github/workflows/crashlab-eval.yml``; ``ci="none"`` skips CI files.
    ``.gitignore`` is created or updated so ``.crashlab/`` is ignored.
    """
    ci_name = (ci or "none").strip().lower()
    if ci_name not in _SUPPORTED_CI:
        supported = ", ".join(_SUPPORTED_CI)
        raise ValueError(f"unsupported ci={ci!r}; expected one of {supported}")
    root = Path(target_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = InitResult(target_dir=str(root))

    if ci_name == "github":
        _write(
            root / ".github" / "workflows" / "crashlab-eval.yml",
            _WORKFLOW,
            force=force,
            result=result,
        )
    else:
        result.notes.append("skipped CI workflow (ci=none)")

    if checks:
        _write(root / "crashlab_checks.py", _CHECKS, force=force, result=result)
    else:
        result.notes.append("skipped crashlab_checks.py (--no-checks)")

    _ensure_gitignore(root, result=result)
    return result


def _write(path: Path, content: str, *, force: bool, result: InitResult) -> None:
    rel = _rel(path, result.target_dir)
    if path.exists() and not force:
        result.skipped.append(rel)
        result.notes.append(f"left existing {rel} in place (pass --force to overwrite)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, "utf-8")
    if existed:
        result.updated.append(rel)
    else:
        result.created.append(rel)


def _ensure_gitignore(root: Path, *, result: InitResult) -> None:
    path = root / ".gitignore"
    rel = ".gitignore"
    if not path.exists():
        path.write_text(_GITIGNORE, "utf-8")
        result.created.append(rel)
        result.notes.append(f"wrote {rel} with {GITIGNORE_ENTRY}")
        return
    text = path.read_text("utf-8")
    if _ignores_crashlab(text):
        result.skipped.append(rel)
        result.notes.append(f"{rel} already ignores {GITIGNORE_ENTRY}")
        return
    addition = GITIGNORE_ENTRY if text.endswith("\n") or not text else "\n" + GITIGNORE_ENTRY
    if not addition.endswith("\n"):
        addition += "\n"
    if text and not text.endswith("\n") and not addition.startswith("\n"):
        addition = "\n" + addition
    # a file that does not already end in a newline gets a separator first
    if text and not text.endswith("\n"):
        path.write_text(text + "\n" + GITIGNORE_ENTRY + "\n", "utf-8")
    else:
        path.write_text(text + GITIGNORE_ENTRY + "\n", "utf-8")
    result.updated.append(rel)
    result.notes.append(f"added {GITIGNORE_ENTRY} to {rel}")


def _ignores_crashlab(text: str) -> bool:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.rstrip("/") == ".crashlab" or line in {
            ".crashlab/",
            "**/.crashlab/",
            "**/.crashlab",
        }:
            return True
    return False


def _rel(path: Path, root: str) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)
