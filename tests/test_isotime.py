"""Stored timestamps parse on every supported Python.

Found on Python 3.10 with pydantic < 2.3 (the declared floor at the time): pydantic-core
wrote ``2026-11-12T20:49:28.91893+00:00`` (trailing zero of the microseconds dropped),
and 3.10's ``datetime.fromisoformat`` only takes 3 or 6 fraction digits, so ``crashlab ls``
crashed with "Invalid isoformat string" on about one run in ten."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crashlab._isotime import parse_iso

WANT = datetime(2026, 11, 12, 20, 49, 28, 918930, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "text",
    [
        "2026-11-12T20:49:28.91893+00:00",
        "2026-11-12T20:49:28.91893Z",
        "2026-11-12T20:49:28.918930Z",
        "2026-11-12T20:49:28.9189301+00:00",  # 7 digits (other writers): truncated
        "2026-11-12 20:49:28.91893z",
    ],
)
def test_parse_iso_any_fraction_and_z(text):
    assert parse_iso(text) == WANT


def test_parse_iso_plain_forms_and_errors():
    assert parse_iso("2026-11-12") == datetime(2026, 11, 12)
    assert parse_iso("2026-11-12T20:49:28.5+05:30").microsecond == 500000
    with pytest.raises(ValueError):
        parse_iso("yesterday")


def test_cli_and_scan_helpers_take_short_fractions():
    from crashlab import report_html
    from crashlab._runscan import _as_utc, parse_since
    from crashlab.cli import _local_minute, _local_time

    ts = "2026-11-12T20:49:28.91893+00:00"
    assert _local_time(ts) == WANT.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    assert _local_minute(ts) == WANT.astimezone().strftime("%Y-%m-%d %H:%M")
    assert _as_utc(ts) == WANT
    assert parse_since("2026-11-12T20:49:28.91893Z") == WANT
    assert report_html._local(ts).startswith(WANT.astimezone().strftime("%Y-%m-%d %H:%M:%S"))
