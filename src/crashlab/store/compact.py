"""Compact cassette lines, version 1 (EXPERIMENTAL; writing is off by default).

A chat step's ``request.messages`` usually repeats the previous step's whole
conversation, so a long run grows quadratically. In a compact line, a message list
stores only its new tail, plus a reference to the earlier step that holds the prefix::

    {"seq": 7, ..., "request": {"messages": [<tail>]},
     "compact": {"v": 1, "request.messages": {"base": 5, "keep": 12, "check": "<sha>"}}}

The full list is ``messages(base)[:keep] + tail``. ``check`` is a digest of the full
list, so a wrong reconstruction is caught instead of being replayed.

* Reading is always on. Old-format lines and compact lines can be mixed in one file.
* Writing happens only with ``CRASHLAB_EXPERIMENTAL_COMPACT=1``. crashlab 0.0.2 cannot
  read compact lines (it skips them), and ``export`` always writes the full format.
* A compact line that cannot be expanded (missing or damaged base, a cycle, a bad
  digest, an unknown version) is never dropped. The step is kept with its tail only,
  marked with the attribute ``crashlab.compact.unresolved`` and reported in a warning.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

VERSION = 1
ENV = "CRASHLAB_EXPERIMENTAL_COMPACT"
UNRESOLVED_ATTR = "crashlab.compact.unresolved"
FIELDS = (
    ("request", "messages"),
    ("raw_request", "messages"),
    ("raw_request", "input"),
    ("raw_request", "contents"),
)


def enabled() -> bool:
    return os.environ.get(ENV, "").strip().lower() in ("1", "true", "yes", "on")


def digest(items: list[Any]) -> str:
    data = json.dumps(items, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _get(line: dict[str, Any], top: str, key: str) -> Any:
    obj = line.get(top)
    return obj.get(key) if isinstance(obj, dict) else None


def safe_text(text: str, limit: int = 300) -> str:
    """Text built from names found in a file, made safe for warnings and attributes:
    secrets redacted and long names cut, so a crafted key cannot carry a token out."""
    from .redact import get_redactor

    text = get_redactor().redact_str(text)
    out = []
    for word in text.split(" "):
        out.append(word if len(word) <= 48 else word[:12] + "...")
    text = " ".join(out)
    return text if len(text) <= limit else text[:limit] + "..."


def _normalize_msg(m: Any) -> Any:
    """Dict-ify ``Mapping`` subclasses so a cassette line stores real JSON objects,
    not ``default=str`` object addresses, and so encodings are content-stable."""
    if isinstance(m, Mapping) and not isinstance(m, dict):
        return {k: _normalize_msg(v) if isinstance(v, Mapping) and not isinstance(v, dict) else v
                for k, v in m.items()}  # fmt: skip
    return m


def normalize_line_messages(line: dict[str, Any]) -> dict[str, Any]:
    """Coerce message-list entries that are ``Mapping`` (but not ``dict``) to dicts.

    Applied on every append (compact on or off) so a cassette never stores
    ``repr`` of a user Mapping subclass via ``default=str``.
    """
    out = line
    for top, key in FIELDS:
        obj = out.get(top)
        if not isinstance(obj, dict):
            continue
        msgs = obj.get(key)
        if not isinstance(msgs, list) or not any(
            isinstance(m, Mapping) and not isinstance(m, dict) for m in msgs
        ):
            continue
        if out is line:
            out = dict(line)
        out[top] = {**obj, key: [_normalize_msg(m) for m in msgs]}
    return out


def _dumps_msg(m: Any) -> str:
    # ensure_ascii=False matches :func:`digest`, so a check can be built from the cache.
    return json.dumps(_normalize_msg(m), sort_keys=True, ensure_ascii=False, default=str)


def digest_from_enc(enc: list[str]) -> str:
    """Same digest as :func:`digest` over the objects that produced ``enc``."""
    data = "[" + ", ".join(enc) + "]"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


# Reuse cached encodings only when a content fingerprint still matches. Identity is
# never consulted. Fingerprinting a large string is cheaper than re-JSON-escaping it;
# for small messages dumps is cheaper than sha256, so the fp path applies only when
# the previous encoding exceeded FP_THRESHOLD characters.
FP_THRESHOLD = 2000
_CacheEntry = tuple[int, list[Any], list[str], list[bytes | None]]


def _content_fp(o: Any, h: Any | None = None) -> bytes:
    """Tree fingerprint: equal iff ``_dumps_msg`` would be equal for JSON-ish values."""
    if h is None:
        h = hashlib.sha256()
        _content_fp(o, h)
        return h.digest()
    if o is None:
        h.update(b"N")
    elif o is True:
        h.update(b"T")
    elif o is False:
        h.update(b"F")
    elif isinstance(o, int) and not isinstance(o, bool):
        h.update(b"I" + repr(o).encode())
    elif isinstance(o, float):
        h.update(b"f" + json.dumps(o).encode())
    elif isinstance(o, str):
        b = o.encode("utf-8")
        h.update(b"S" + str(len(b)).encode() + b":" + b)
    elif isinstance(o, Mapping):
        items = list(o.items())
        h.update(b"D" + str(len(items)).encode())
        # Message keys are str; avoid calling _dumps_msg here (it would defeat the
        # large-message fast path that exists to skip dumps).
        for k, v in sorted(items, key=lambda kv: (str(type(kv[0])), str(kv[0]))):
            _content_fp(k, h)
            _content_fp(v, h)
    elif isinstance(o, (list, tuple)):
        h.update(b"L" + str(len(o)).encode())
        for x in o:
            _content_fp(x, h)
    else:
        h.update(b"U" + str(o).encode())
    return h.digest()  # ignored when h was passed; kept for type checkers


def _fp_for(m: Any, enc: str) -> bytes | None:
    return _content_fp(m) if len(enc) >= FP_THRESHOLD else None


def _encode_list(
    full: list[Any], prev: _CacheEntry | None
) -> tuple[list[Any], list[str], list[bytes | None]]:
    """Encode ``full`` by content, reusing immutable cached encodings when a content
    fingerprint still matches (large messages) or when a fresh dumps equals the cache
    (small messages). Never uses object identity. Returns ``(norm, enc, fps)``."""
    norm = [_normalize_msg(m) for m in full]
    if prev is None:
        enc = [_dumps_msg(m) for m in norm]
        return norm, enc, [_fp_for(m, e) for m, e in zip(norm, enc, strict=True)]

    _, _, prev_enc, prev_fps = prev
    enc: list[str] = []
    fps: list[bytes | None] = []
    j = 0
    while j < len(norm) and j < len(prev_enc):
        m = norm[j]
        prev_e = prev_enc[j]
        prev_f = prev_fps[j] if j < len(prev_fps) else None
        if prev_f is not None and len(prev_e) >= FP_THRESHOLD:
            f = _content_fp(m)
            if f == prev_f:
                enc.append(prev_e)
                fps.append(prev_f)
                j += 1
                continue
            break
        e = _dumps_msg(m)
        if e == prev_e:
            enc.append(prev_e)
            fps.append(prev_f if prev_f is not None else _fp_for(m, e))
            j += 1
            continue
        enc.append(e)
        fps.append(_fp_for(m, e))
        j += 1
        break
    else:
        # prefix exhausted without break: encode any new tail only
        for m in norm[j:]:
            e = _dumps_msg(m)
            enc.append(e)
            fps.append(_fp_for(m, e))
        return norm, enc, fps
    # mismatched at j-1 already appended; encode the remainder
    for m in norm[j:]:
        e = _dumps_msg(m)
        enc.append(e)
        fps.append(_fp_for(m, e))
    return norm, enc, fps


class Compactor:
    """Writer side; holds the last full list seen for each field of one cassette.

    Cached per field: ``(seq, messages_ref, encodings, fingerprints)``. Encodings are
    content-based (identity is never trusted). Large-message fingerprints let an
    append-only step reuse immutable encodings without re-escaping; a nested mutation
    changes the fingerprint and forces a fresh dumps. ``encodings`` / ``fingerprints``
    are never mutated in place, so :meth:`snapshot` / :meth:`restore` stay exact.
    """

    def __init__(self) -> None:
        self._last: dict[str, _CacheEntry] = {}

    def snapshot(self) -> dict[str, _CacheEntry]:
        return dict(self._last)

    def restore(self, snap: dict[str, _CacheEntry]) -> None:
        self._last = snap

    def compact(self, line: dict[str, Any]) -> dict[str, Any]:
        seq = line.get("seq")
        if not isinstance(seq, int):
            return line
        meta: dict[str, Any] = {}
        out = dict(line)
        for top, key in FIELDS:
            full = _get(line, top, key)
            if not isinstance(full, list):
                continue
            name = f"{top}.{key}"
            prev = self._last.get(name)
            norm, enc, fps = _encode_list(full, prev)
            self._last[name] = (seq, full, enc, fps)
            out[top] = {**out[top], key: norm}
            if prev is None or prev[0] == seq:
                continue
            base_seq, _, base_enc, _ = prev
            keep = 0
            for a, b in zip(base_enc, enc, strict=False):
                if a != b:
                    break
                keep += 1
            if keep == 0:
                continue
            out[top] = {**out[top], key: norm[keep:]}
            meta[name] = {"base": base_seq, "keep": keep, "check": digest_from_enc(enc)}
        if meta:
            out["compact"] = {"v": VERSION, **meta}
        return out


def expand(lines: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand compact lines in place of the originals. Returns (lines, problems)."""
    by_seq: dict[int, dict[str, Any]] = {}
    for ln in lines:
        if isinstance(ln.get("seq"), int):
            by_seq.setdefault(ln["seq"], ln)
    memo: dict[tuple[int, str], list[Any]] = {}
    problems: list[str] = []

    def resolve(seq: int, name: str) -> list[Any]:
        """Iterative (no recursion): walk base references back to a known list, then
        rebuild forward. Long or out-of-order chains cost O(chain) once, memoised."""
        top, key = name.split(".", 1)
        path: list[tuple[int, list[Any], dict[str, Any] | None]] = []
        seen: set[int] = set()
        cur = seq
        prefix: list[Any] | None = None
        while True:
            if (cur, name) in memo:
                prefix = memo[(cur, name)]
                break
            if cur in seen:
                raise ValueError(f"reference cycle at seq {cur}")
            seen.add(cur)
            ln = by_seq.get(cur)
            if ln is None:
                raise ValueError(f"base step {cur} is missing")
            own = _get(ln, top, key)
            if not isinstance(own, list):
                raise ValueError(f"step {cur} has no list for this field")
            meta = ln.get("compact")
            ref = meta.get(name) if isinstance(meta, dict) else None
            if ref is None:
                memo[(cur, name)] = own
                prefix = own
                break
            if meta.get("v") != VERSION:
                raise ValueError(f"unsupported compact version at seq {cur}")
            if not isinstance(ref, dict):
                raise ValueError(f"bad compact entry at seq {cur}")
            base, keep = ref.get("base"), ref.get("keep")
            if type(base) is not int or type(keep) is not int or keep < 0:
                raise ValueError(f"bad compact entry at seq {cur}")
            path.append((cur, own, ref))
            cur = base
        for cur, own, ref in reversed(path):
            keep = ref["keep"]
            if keep > len(prefix):
                raise ValueError(f"keep={keep} is longer than base step {ref['base']}")
            res = prefix[:keep] + own
            if ref.get("check") != digest(res):  # required: never replay a wrong list
                raise ValueError(f"digest mismatch at seq {cur}")
            memo[(cur, name)] = res
            prefix = res
        return prefix

    out = []
    for ln in lines:
        if "compact" not in ln:
            out.append(ln)
            continue
        new = dict(ln)
        meta = new.pop("compact")
        errors = []
        if not isinstance(meta, dict) or meta.get("v") != VERSION:
            errors.append("unsupported compact version")
        else:
            for name in meta:
                if name == "v":
                    continue
                top, _, key = name.partition(".")
                if (top, key) not in FIELDS:
                    errors.append(f"unknown compact field {name!r}"[:80])
                    continue
                try:
                    res = resolve(new["seq"], name)
                except (ValueError, TypeError, KeyError) as e:
                    errors.append(str(e))
                    continue
                new[top] = {**new[top], key: res}
        if errors:
            attrs = new.get("attributes")
            attrs = dict(attrs) if isinstance(attrs, dict) else {}
            msg = safe_text("; ".join(errors))
            attrs[UNRESOLVED_ATTR] = msg
            new["attributes"] = attrs
            problems.append(f"seq {new.get('seq')}: {msg}")
        out.append(new)
    return out, problems
