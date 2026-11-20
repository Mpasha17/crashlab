"""Shareable debug bundles and deep secret / PII audits of recorded runs.

``crashlab bundle`` packs a run (and its linked children) into a portable
``.crashlab.zip`` archive. Secrets are redacted by default. ``audit_secrets``
and ``redact_run`` scan every step, input, output, metadata field and finding;
reports never include the raw secret values.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ._version import __version__
from .models import Run
from .store import Store, import_run

# Replacement written into redacted runs and used as the public preview.
_PLACEHOLDER = "[REDACTED:{kind}]"
_ALREADY = re.compile(r"\[REDACTED:[a-z0-9_]+\]")

# More-specific patterns must be listed first so overlap resolution keeps them.
# Each pattern is compiled with the flags given here; matches are classified by
# ``kind`` and never returned in reports — only ``_preview`` / the placeholder.
_SECRET_SPECS: tuple[tuple[str, str, str, int], ...] = (
    # kind, severity, pattern, flags
    ("anthropic_api_key", "error", r"sk-ant-(?:api\d+-)?[A-Za-z0-9_\-]{16,}", 0),
    ("openai_api_key", "error", r"sk-(?:proj-)?[A-Za-z0-9_\-]{16,}", 0),
    ("github_token", "error", r"github_pat_[A-Za-z0-9_]{20,}", 0),
    ("github_token", "error", r"gh[pousr]_[A-Za-z0-9]{20,}", 0),
    ("aws_access_key", "error", r"AKIA[0-9A-Z]{16}\b", 0),
    ("slack_token", "error", r"xox[abprs]-[A-Za-z0-9\-]{10,}", 0),
    ("google_api_key", "error", r"AIza[0-9A-Za-z_\-]{35}", 0),
    ("jwt", "error", r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}", 0),
    ("bearer_token", "error", r"(?i)bearer\s+[A-Za-z0-9._\-~+/]{8,}=*", 0),
    (
        "private_key",
        "critical",
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z]+ )?PRIVATE KEY-----",
        re.DOTALL,
    ),
)

_PII_SPECS: tuple[tuple[str, str, str, int], ...] = (
    ("email", "warn", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0),
    # E.164 or a North-American number with optional separators / area-code parens.
    (
        "phone",
        "warn",
        r"(?<!\w)(?:\+\d{1,3}[-.\s]?)?(?:\(?[2-9]\d{2}\)?[-.\s]?)?[2-9]\d{2}[-.\s]?\d{4}\b",
        0,
    ),
)

_COMPILED_SECRETS = tuple(
    (kind, sev, re.compile(pat, flags)) for kind, sev, pat, flags in _SECRET_SPECS
)
_COMPILED_PII = tuple((kind, sev, re.compile(pat, flags)) for kind, sev, pat, flags in _PII_SPECS)

BUNDLE_FORMAT = "crashlab.bundle"
BUNDLE_VERSION = 1


class BundleError(ValueError):
    """The path is not a crashlab debug bundle, or the archive is unreadable."""


@dataclass
class SecretFinding:
    """One leaked secret or PII hit. ``redacted_preview`` never holds the raw value."""

    step_idx: int | None
    field_path: str
    kind: str
    severity: str
    redacted_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SecretAuditReport:
    """Deep secret / PII scan of one recorded run."""

    run_id: str
    name: str = ""
    include_pii: bool = False
    findings: list[SecretFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def count(self) -> int:
        return len(self.findings)

    def counts_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.findings:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "include_pii": self.include_pii,
            "ok": self.ok,
            "count": self.count,
            "by_kind": self.counts_by_kind(),
            "findings": [item.to_dict() for item in self.findings],
        }


@dataclass
class BundleManifest:
    """Descriptor written to ``manifest.json`` inside a ``.crashlab.zip``."""

    crashlab_version: str
    root_run_id: str
    run_ids: list[str]
    redacted: bool
    include_html: bool = True
    include_children: bool = True
    format: str = BUNDLE_FORMAT
    version: int = BUNDLE_VERSION
    audit: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "format": self.format,
            "version": self.version,
            "crashlab_version": self.crashlab_version,
            "root_run_id": self.root_run_id,
            "run_ids": list(self.run_ids),
            "redacted": self.redacted,
            "include_html": self.include_html,
            "include_children": self.include_children,
            "audit": dict(self.audit),
        }
        if self.path is not None:
            data["path"] = self.path
        return data


# --- scanning -------------------------------------------------------------------------


def _preview(kind: str) -> str:
    return _PLACEHOLDER.format(kind=kind)


def _masked_spans(text: str) -> list[tuple[int, int]]:
    """Spans already occupied by a ``[REDACTED:kind]`` placeholder."""
    return [m.span() for m in _ALREADY.finditer(text)]


def _covered(start: int, end: int, masked: list[tuple[int, int]]) -> bool:
    return any(a <= start and end <= b for a, b in masked)


def _collect_matches(text: str, *, include_pii: bool) -> list[tuple[int, int, str, str]]:
    """Non-overlapping matches as ``(start, end, kind, severity)``, left-to-right.

    Longer / earlier-listed (more specific) patterns win on overlap. Spans that
    sit inside an existing placeholder are ignored.
    """
    masked = _masked_spans(text)
    raw: list[tuple[int, int, str, str]] = []
    specs = _COMPILED_SECRETS + (_COMPILED_PII if include_pii else ())
    for kind, severity, pat in specs:
        for m in pat.finditer(text):
            start, end = m.span()
            if start == end or _covered(start, end, masked):
                continue
            raw.append((start, end, kind, severity))
    raw.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: list[tuple[int, int, str, str]] = []
    used: list[tuple[int, int]] = []
    for start, end, kind, severity in raw:
        if any(start < b and end > a for a, b in used):
            continue
        kept.append((start, end, kind, severity))
        used.append((start, end))
    kept.sort(key=lambda h: h[0])
    return kept


def _replace_secrets(text: str, *, include_pii: bool) -> tuple[str, int]:
    hits = _collect_matches(text, include_pii=include_pii)
    if not hits:
        return text, 0
    out = text
    for start, end, kind, _severity in reversed(hits):
        out = out[:start] + _preview(kind) + out[end:]
    return out, len(hits)


def _walk(obj: Any, path: str, step_idx: int | None, include_pii: bool) -> list[SecretFinding]:
    found: list[SecretFinding] = []
    if isinstance(obj, str):
        for _start, _end, kind, severity in _collect_matches(obj, include_pii=include_pii):
            found.append(
                SecretFinding(
                    step_idx=step_idx,
                    field_path=path or ".",
                    kind=kind,
                    severity=severity,
                    redacted_preview=_preview(kind),
                )
            )
        return found
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            child_step = step_idx
            if path == "steps" and isinstance(value, dict) and isinstance(value.get("idx"), int):
                child_step = value["idx"]
            found.extend(_walk(value, child, child_step, include_pii))
        return found
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            child = f"{path}[{i}]"
            child_step = step_idx
            if path == "steps" and isinstance(value, dict) and isinstance(value.get("idx"), int):
                child_step = value["idx"]
            found.extend(_walk(value, child, child_step, include_pii))
        return found
    return found


def _redact_obj(obj: Any, *, include_pii: bool) -> tuple[Any, int]:
    if isinstance(obj, str):
        return _replace_secrets(obj, include_pii=include_pii)
    if isinstance(obj, dict):
        total = 0
        out = {}
        for key, value in obj.items():
            new, n = _redact_obj(value, include_pii=include_pii)
            out[key] = new
            total += n
        return out, total
    if isinstance(obj, list):
        total = 0
        items = []
        for value in obj:
            new, n = _redact_obj(value, include_pii=include_pii)
            items.append(new)
            total += n
        return items, total
    if isinstance(obj, tuple):
        items, total = _redact_obj(list(obj), include_pii=include_pii)
        return tuple(items), total
    return obj, 0


def audit_secrets(run: Run, *, include_pii: bool = False) -> SecretAuditReport:
    """Deep-scan ``run`` for leaked secret patterns (and optional PII).

    Findings record ``step_idx``, ``field_path``, ``kind``, ``severity`` and a
    ``redacted_preview``. The raw secret is never included in the report.
    """
    data = run.model_dump(mode="json")
    findings = _walk(data, "", None, include_pii)
    return SecretAuditReport(
        run_id=run.run_id,
        name=run.name,
        include_pii=include_pii,
        findings=findings,
    )


def redact_run(
    run: Run,
    *,
    include_pii: bool = False,
    store: Store | None = None,
    in_place: bool = False,
) -> tuple[Run, int]:
    """Replace detected secrets (and optional PII) with ``[REDACTED:<kind>]``.

    Returns ``(redacted_run, replacement_count)``. With ``in_place=True`` the
    redacted run is also written back to ``store`` (required).
    """
    if in_place and store is None:
        raise ValueError("in_place=True requires a store")
    data = run.model_dump(mode="json")
    redacted_data, count = _redact_obj(data, include_pii=include_pii)
    redacted = Run.model_validate(redacted_data)
    if in_place and store is not None:
        _persist_run(store, redacted)
        redacted = store.load_run(redacted.run_id)
    return redacted, count


def _persist_run(store: Store, run: Run) -> None:
    """Replace a stored run's metadata and cassette with ``run``."""
    # Drop the previous rows / directory so a rewritten cassette cannot sit
    # next to unredacted history (diagnoses, flags, old steps).
    existing = (
        store.index.get_run_json(run.run_id) is not None or store.run_dir(run.run_id).exists()
    )
    if existing:
        store.delete_run(run.run_id)
    store.begin_run(run)
    for step in run.steps:
        store.add_step(step)
    store.end_run(run)


# --- bundles --------------------------------------------------------------------------


def _export_jsonl(run: Run) -> str:
    """JSONL in the same portable format as ``store.export_run``, no extra pass."""
    from .store import EXPORT_HEADER, EXPORT_VERSION
    from .store.cassette import step_to_line

    head = {
        "type": EXPORT_HEADER,
        "version": EXPORT_VERSION,
        "run": run.model_dump(mode="json", exclude={"steps"}),
    }
    lines = [head] + [step_to_line(s.model_dump(mode="json")) for s in run.steps]
    return "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in lines)


def _scrub_text(text: str) -> str:
    """Drop any secret / PII that might have landed in an error string."""
    return _replace_secrets(text, include_pii=True)[0]


def _diagnosis_payload(run: Run) -> dict[str, Any]:
    if run.diagnosis is not None:
        return run.diagnosis.model_dump(mode="json")
    try:
        from .diagnose import diagnose

        diagnosis = diagnose(run, heuristic=True, save=False, store=None, cache=False)
        return diagnosis.model_dump(mode="json")
    except Exception as exc:
        return {
            "run_id": run.run_id,
            "error": type(exc).__name__,
            "message": _scrub_text(str(exc)),
        }


def _context_payload(runs: list[Run]) -> dict[str, Any]:
    from .context_audit import analyze_context, audit_tool_schemas

    root, *children = runs[0], *runs[1:]
    payload: dict[str, Any] = {
        "root_run_id": root.run_id,
        "context": analyze_context(root).to_dict(),
        "tool_audit": audit_tool_schemas(root).to_dict(),
    }
    if children:
        payload["children"] = [
            {
                "run_id": child.run_id,
                "context": analyze_context(child).to_dict(),
                "tool_audit": audit_tool_schemas(child).to_dict(),
            }
            for child in children
        ]
    return payload


def _audit_summary(reports: list[SecretAuditReport]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_run: list[dict[str, Any]] = []
    total = 0
    for report in reports:
        total += report.count
        by_run.append(
            {
                "run_id": report.run_id,
                "count": report.count,
                "by_kind": report.counts_by_kind(),
                "findings": [
                    {
                        "step_idx": item.step_idx,
                        "field_path": item.field_path,
                        "kind": item.kind,
                        "severity": item.severity,
                        "redacted_preview": item.redacted_preview,
                    }
                    for item in report.findings
                ],
            }
        )
        for kind, n in report.counts_by_kind().items():
            by_kind[kind] = by_kind.get(kind, 0) + n
    return {"count": total, "by_kind": by_kind, "runs": by_run}


def _resolve_out_path(out_path: str | Path, run_id: str) -> Path:
    out = Path(out_path)
    if out.exists() and out.is_dir():
        return out / f"{run_id}.crashlab.zip"
    if not out.suffix:
        return out.with_name(out.name + ".crashlab.zip")
    return out


def export_bundle(
    store: Store,
    run_id: str,
    out_path: str | Path,
    *,
    redact: bool = True,
    include_html: bool = True,
    include_children: bool = True,
) -> BundleManifest:
    """Write a portable ``.crashlab.zip`` debug bundle for ``run_id``.

    The archive contains ``manifest.json``, ``runs/<run_id>.jsonl`` for the root
    run and (by default) every linked child, ``diagnosis.json``,
    ``context.json`` (prompt breakdown + tool-schema audit) and, when
    ``include_html`` is true, ``report.html``. Runs are redacted by default.
    """
    from .otel import linked_runs

    root = store.load_run(run_id)
    runs = linked_runs(root, store=store, include_children=include_children)
    reports = [audit_secrets(item) for item in runs]
    packed: list[Run] = []
    if redact:
        for item in runs:
            redacted, _n = redact_run(item, include_pii=False)
            packed.append(redacted)
    else:
        packed = list(runs)

    dest = _resolve_out_path(out_path, root.run_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = BundleManifest(
        crashlab_version=__version__,
        root_run_id=root.run_id,
        run_ids=[item.run_id for item in packed],
        redacted=redact,
        include_html=include_html,
        include_children=include_children,
        audit=_audit_summary(reports),
        path=str(dest),
    )
    # diagnosis / context / html are taken from the (possibly redacted) copies
    # so a shared bundle cannot leak values the redactor already stripped.
    packed_root = packed[0]
    diagnosis = {rid: _diagnosis_payload(item) for rid, item in ((r.run_id, r) for r in packed)}
    context = _context_payload(packed)
    html = None
    if include_html:
        from .report_html import render_html

        html = render_html(packed_root)

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, default=str) + "\n",
        )
        for item in packed:
            zf.writestr(f"runs/{item.run_id}.jsonl", _export_jsonl(item))
        zf.writestr(
            "diagnosis.json",
            json.dumps(
                {"root_run_id": packed_root.run_id, "diagnoses": diagnosis},
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n",
        )
        zf.writestr(
            "context.json",
            json.dumps(context, indent=2, ensure_ascii=False, default=str) + "\n",
        )
        if html is not None:
            zf.writestr("report.html", html)
    return manifest


def _safe_run_member(name: str) -> str | None:
    """Return the run id for a ``runs/<id>.jsonl`` member, or None if unsafe."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return None
    if len(path.parts) != 2 or path.parts[0] != "runs":
        return None
    if path.suffix != ".jsonl":
        return None
    run_id = path.stem
    if not run_id or "/" in run_id or "\\" in run_id:
        return None
    return run_id


def import_bundle(store: Store, bundle_path: str | Path) -> list[str]:
    """Import every ``runs/*.jsonl`` from a ``.crashlab.zip`` into ``store``.

    Returns the imported run ids (manifest order when present, otherwise the
    archive member order). Existing runs are left alone unless later replaced
    through ``crashlab import --replace`` of a plain JSONL export.
    """
    path = Path(bundle_path)
    if not path.is_file():
        raise BundleError(f"no such bundle: {path}")
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise BundleError(f"not a zip archive: {path}") from exc
    with zf:
        names = zf.namelist()
        members: list[tuple[str, str]] = []
        for name in names:
            run_id = _safe_run_member(name)
            if run_id is not None:
                members.append((name, run_id))
        if not members:
            raise BundleError(f"not a crashlab bundle (no runs/*.jsonl): {path}")
        order: list[str] | None = None
        if "manifest.json" in names:
            try:
                raw = json.loads(zf.read("manifest.json").decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BundleError(f"unreadable manifest in {path}") from exc
            if not isinstance(raw, dict) or raw.get("format") not in (BUNDLE_FORMAT, None):
                # A manifest with an unexpected format still imports the runs;
                # only a corrupt file is fatal (handled above).
                pass
            ids = raw.get("run_ids") if isinstance(raw, dict) else None
            if isinstance(ids, list) and all(isinstance(i, str) for i in ids):
                order = list(ids)
        by_id = {run_id: name for name, run_id in members}
        if order is None:
            wanted = [run_id for _name, run_id in members]
        else:
            wanted = [rid for rid in order if rid in by_id]
            wanted.extend(rid for _name, rid in members if rid not in wanted)
        imported: list[str] = []
        for run_id in wanted:
            name = by_id[run_id]
            data = zf.read(name)
            buf = io.BytesIO(data)
            # import_run reads a path; write a temp file next to the store.
            tmp = store.root / ".bundle-import" / f"{run_id}.jsonl"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp.write_bytes(data)
                rid, _added = import_run(store, tmp)
            except (OSError, ValueError) as exc:
                raise BundleError(_scrub_text(f"{name}: {exc}")) from exc
            finally:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
            imported.append(rid)
            del buf
        tmp_dir = store.root / ".bundle-import"
        with contextlib.suppress(OSError):
            tmp_dir.rmdir()
    return imported


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_VERSION",
    "BundleError",
    "BundleManifest",
    "SecretAuditReport",
    "SecretFinding",
    "audit_secrets",
    "export_bundle",
    "import_bundle",
    "redact_run",
]
