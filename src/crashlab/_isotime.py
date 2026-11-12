"""ISO-8601 parsing that behaves the same on every supported Python.

Python 3.10's ``datetime.fromisoformat`` rejects a trailing ``Z`` and fractional seconds
that are not exactly 3 or 6 digits. Older pydantic-core wrote ``...28.91893+00:00``
(trailing zero dropped), so stores and exports can hold those."""

from __future__ import annotations

import re
from datetime import datetime

_FRACTION = re.compile(r"(\d{2}:\d{2}:\d{2})[.,](\d+)")


def parse_iso(text: str) -> datetime:
    """``datetime.fromisoformat`` that also takes ``Z`` and 1-9 fraction digits.
    Raises ``ValueError`` like it on anything else."""
    s = str(text).strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    s = _FRACTION.sub(lambda m: f"{m.group(1)}.{m.group(2)[:6].ljust(6, '0')}", s, count=1)
    return datetime.fromisoformat(s)
