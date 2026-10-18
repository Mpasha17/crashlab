"""Small id helpers: ULID-like sortable run ids and OTel-style span ids (no extra deps)."""

from __future__ import annotations

import os
import threading
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_lock = threading.Lock()
_last_ms = -1
_last_rand = 0


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        value, rem = divmod(value, 32)
        out.append(_CROCKFORD[rem])
    return "".join(reversed(out))


def new_ulid() -> str:
    """Return a 26-char Crockford base32 id: 48-bit ms timestamp + 80 random bits.

    Ids are lexicographically sortable by creation time. Within the same millisecond
    the random part is incremented, so ids stay strictly increasing in one process.
    """
    global _last_ms, _last_rand
    with _lock:
        ms = time.time_ns() // 1_000_000
        if ms <= _last_ms:
            ms = _last_ms
            rand = (_last_rand + 1) & ((1 << 80) - 1)
        else:
            rand = int.from_bytes(os.urandom(10), "big")
        _last_ms, _last_rand = ms, rand
    return _encode(ms, 10) + _encode(rand, 16)


def ulid_timestamp_ms(ulid: str) -> int:
    """Decode the millisecond timestamp from a ULID-like id."""
    value = 0
    for ch in ulid[:10].upper():
        value = value * 32 + _CROCKFORD.index(ch)
    return value


def new_span_id() -> str:
    """Return a 16-hex-char span id (same size as an OpenTelemetry span id)."""
    return os.urandom(8).hex()
