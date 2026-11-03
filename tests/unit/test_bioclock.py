"""bioclock regression tests: the prompt block must be self-consistent.

Covers the incident where Aiko told the user "4:47 AM Vancouver Time /
08:47 UTC" — a local hour paired with a mismatched UTC instant. The block
now carries the UTC offset plus a quote-exactly instruction; these tests
parse a rendered block and prove the local wall time, the offset, and the
UTC instant all describe the same moment, and that the configured timezone
is honoured live (not frozen at import).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from system import bioclock


def _render(tz="America/Vancouver", monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("TIMEZONE", tz)
    else:
        import os

        os.environ["TIMEZONE"] = tz
    return bioclock.current_datetime_block()


def test_block_times_describe_the_same_instant(monkeypatch):
    block = _render(monkeypatch=monkeypatch)
    local_m = re.search(r"Local time \(([^,]+), (UTC[+-]\d{2}:\d{2})\): (\d{2}):(\d{2}) (AM|PM)", block)
    utc_m = re.search(r"UTC: (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", block)
    assert local_m and utc_m, block
    zone_name, offset_s, hh, mm, ap = local_m.groups()
    assert zone_name == "America/Vancouver"

    hour = int(hh) % 12 + (12 if ap == "PM" else 0)
    utc = datetime.strptime(utc_m.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    expected_local = utc.astimezone(ZoneInfo(zone_name))
    # Minute precision: the block's two clocks are read microseconds apart.
    assert (expected_local.hour, expected_local.minute) == (hour, int(mm)), block
    raw_offset = expected_local.strftime("%z")  # e.g. -0700
    assert f"UTC{raw_offset[:3]}:{raw_offset[3:]}" == offset_s, block


def test_configured_timezone_honoured_live(monkeypatch):
    monkeypatch.setenv("TIMEZONE", "Asia/Tokyo")
    assert bioclock.timezone_name() == "Asia/Tokyo"
    block = bioclock.current_datetime_block()
    assert "Asia/Tokyo" in block
    # UTC instant and Tokyo wall time must agree.
    utc_m = re.search(r"UTC: (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", block)
    utc = datetime.strptime(utc_m.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    tokyo = utc.astimezone(ZoneInfo("Asia/Tokyo"))
    assert tokyo.strftime("%I:%M %p") in block, block


def test_unknown_timezone_falls_back_to_utc(monkeypatch):
    monkeypatch.setenv("TIMEZONE", "Not/AZone")
    now = bioclock.local_now()
    assert now.utcoffset().total_seconds() == 0
