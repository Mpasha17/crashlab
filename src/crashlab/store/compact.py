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
    # same options as :func:`digest`, so message encodings line up with the list text
    return json.dumps(_normalize_msg(m), sort_keys=True, ensure_ascii=False, default=str)


# Writer cache, per field: ``(seq, text, ends)``. ``text`` is the JSON text of the last
# list written (exactly what :func:`digest` hashes) and ``ends[i]`` is the offset just past
# message ``i`` in it. Every step re-encodes the whole current list with one
# ``json.dumps`` call, so in-place and nested mutations are always seen: nothing is reused
# because an object is the same object, and nothing is reused because a fingerprint
# matched. Only *text* is compared, and the prefix kept is the longest run of whole
# messages whose encodings are byte-identical to the previous step's.
_CacheEntry = tuple[int, str, tuple[int, ...] | None]


def _list_text(norm: list[Any]) -> str:
    return json.dumps(norm, sort_keys=True, ensure_ascii=False, default=str)


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _common_messages(text: str, prev_text: str, ends: tuple[int, ...]) -> int:
    """How many leading messages of the previous list are encoded identically in
    ``text``. A JSON value ends where its text ends, so equal text up to the end of
    message ``k-1`` followed by a delimiter means the first ``k`` values are equal."""
    n = len(ends)
    if n == 0:
        return 0

    def same(k: int) -> bool:
        e = ends[k - 1]
        return len(text) > e and text.startswith(prev_text[:e]) and text[e] in ",]"

    if same(n):  # the usual agent loop: the whole previous list is still the prefix
        return n
    lo, hi = 0, n - 1  # largest k in [lo, hi] whose prefix text matches
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text.startswith(prev_text[: ends[mid - 1]]):
            lo = mid
        else:
            hi = mid - 1
    # a number value can continue past the old boundary ("1" -> "12"); step back once,
    # the boundary before it is followed by "," in both texts
    return lo if lo == 0 or same(lo) else lo - 1


def _ends_from(
    text: str, norm: list[Any], keep: int, ends: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    """Offsets of every message in ``text``: the first ``keep`` come from the previous
    step, the rest by encoding only the new messages. ``None`` if the encodings do not
    line up (a value whose ``str()`` changes between calls), so nothing refers to it."""
    out = list(ends[:keep]) if ends is not None and keep else []
    pos = out[-1] if out else 1
    if not text.startswith("["):
        return None
    for i in range(keep, len(norm)):
        if i:
            if not text.startswith(", ", pos):
                return None
            pos += 2
        e = _dumps_msg(norm[i])
        if not text.startswith(e, pos):
            return None
        pos += len(e)
        out.append(pos)
    if pos != len(text) - 1 or text[pos] != "]":
        return None
    return tuple(out)


class Compactor:
    """Writer side; holds, per field of one cassette, the JSON text of the last list
    written and its message boundaries (see ``_CacheEntry``). Entries are immutable
    tuples, so :meth:`snapshot` / :meth:`restore` stay exact after a failed write."""

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
            norm = [_normalize_msg(m) for m in full]
            text = _list_text(norm)
            keep, prev_ends = 0, None
            if prev is not None and prev[0] != seq and prev[2] is not None:
                prev_ends = prev[2]
                keep = _common_messages(text, prev[1], prev_ends)
            ends = _ends_from(text, norm, keep, prev_ends)
            if ends is None:
                keep = 0
            self._last[name] = (seq, text, ends)
            out[top] = {**out[top], key: norm}
            if keep == 0 or prev is None:
                continue
            out[top] = {**out[top], key: norm[keep:]}
            meta[name] = {"base": prev[0], "keep": keep, "check": _text_digest(text)}
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
