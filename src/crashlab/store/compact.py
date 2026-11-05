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


class Compactor:
    """Writer side; holds the last full list seen for each field of one cassette."""

    def __init__(self) -> None:
        self._last: dict[str, tuple[int, list[Any], list[str]]] = {}

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
            enc = [json.dumps(m, sort_keys=True, default=str) for m in full]
            prev = self._last.get(name)
            self._last[name] = (seq, full, enc)
            if prev is None or prev[0] == seq:
                continue
            base_seq, _, base_enc = prev
            keep = 0
            for a, b in zip(base_enc, enc, strict=False):
                if a != b:
                    break
                keep += 1
            if keep == 0:
                continue
            out[top] = {**out[top], key: full[keep:]}
            meta[name] = {"base": base_seq, "keep": keep, "check": digest(full)}
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

    def full_list(seq: int, name: str, stack: frozenset[tuple[int, str]]) -> list[Any]:
        k = (seq, name)
        if k in memo:
            return memo[k]
        if k in stack:
            raise ValueError(f"reference cycle at seq {seq}")
        ln = by_seq.get(seq)
        if ln is None:
            raise ValueError(f"base step {seq} is missing")
        top, key = name.split(".", 1)
        own = _get(ln, top, key)
        if not isinstance(own, list):
            raise ValueError(f"step {seq} has no {name} list")
        meta = ln.get("compact")
        ref = meta.get(name) if isinstance(meta, dict) else None
        if ref is None:
            res = own
        else:
            if not isinstance(meta, dict) or meta.get("v") != VERSION:
                raise ValueError(f"unsupported compact version {meta.get('v')!r}")
            if not isinstance(ref, dict):
                raise ValueError(f"bad compact entry for {name}")
            base, keep = ref.get("base"), ref.get("keep")
            if type(base) is not int or type(keep) is not int or keep < 0:
                raise ValueError(f"bad compact entry for {name}")
            prefix = full_list(base, name, stack | {k})
            if keep > len(prefix):
                raise ValueError(f"keep={keep} is longer than base step {base}")
            res = prefix[:keep] + own
            if ref.get("check") != digest(res):  # required: never replay a wrong list
                raise ValueError(f"{name} digest mismatch")
        memo[k] = res
        return res

    out = []
    for ln in lines:
        if "compact" not in ln:
            out.append(ln)
            continue
        new = dict(ln)
        meta = new.pop("compact")
        errors = []
        if not isinstance(meta, dict) or meta.get("v") != VERSION:
            v = meta.get("v") if isinstance(meta, dict) else None
            errors.append(f"unsupported compact version {v!r}")
        else:
            for name in meta:
                if name == "v":
                    continue
                top, _, key = name.partition(".")
                if (top, key) not in FIELDS:
                    errors.append(f"unknown compact field {name!r}")
                    continue
                try:
                    res = full_list(new["seq"], name, frozenset())
                except (ValueError, TypeError, KeyError, RecursionError) as e:
                    errors.append(str(e))
                    continue
                new[top] = {**new[top], key: res}
        if errors:
            attrs = new.get("attributes")
            attrs = dict(attrs) if isinstance(attrs, dict) else {}
            attrs[UNRESOLVED_ATTR] = "; ".join(errors)
            new["attributes"] = attrs
            problems.append(f"seq {new.get('seq')}: {'; '.join(errors)}")
        out.append(new)
    return out, problems
