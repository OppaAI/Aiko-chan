"""
system/bioclock.py

Single source of truth for "what time is it right now" across every module
that needs to reason about dates or wall-clock time — chat/webchat/agentic
system prompts, proactive check-ins, and system/schedule.py's job timing.

Centralized here so every module resolves timezone the same way instead of
each rolling its own ZoneInfo lookup with its own fallback quirks. Config
lives in config/bioclock.yaml (TIMEZONE key); system.config.load_config() has
already populated it into the process environment by the time this module
is imported, same as every other module's config block.

Callers that need a *different* timezone than the app default for one
specific record (e.g. a schedule.json job saved with its own "timezone"
field) can pass an explicit override to any function here — the override
always wins, the config default is only the fallback when none is given.
"""

from __future__ import annotations

import os
import time
import threading

from system.config import load_config
load_config()
from datetime import datetime, timezone

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from system.log import get_logger

log = get_logger(__name__)


def _configured_timezone() -> str:
    """App-wide default timezone, read live from the environment every call.

    config/bioclock.yaml documents the TIMEZONE key and system.config loads
    it into the process env at boot — but this module must not freeze the
    value in a constant at import time. Import order is not guaranteed (any
    module can import bioclock before main() runs load_config), and a frozen
    fallback would silently stamp UTC wall-clock as local time everywhere.
    """
    return os.getenv("TIMEZONE", "UTC")


# Kept for backwards compatibility; prefer timezone_name(), which re-reads
# the environment instead of freezing the import-time value.
DEFAULT_TIMEZONE = os.getenv("TIMEZONE", "UTC")


def timezone_name(name: str | None = None) -> str:
    """Resolve the effective timezone name: an explicit override (e.g. a
    job's own "timezone" field) takes precedence, otherwise the app-wide
    default from config/bioclock.yaml."""
    name = (name or "").strip()
    return name or _configured_timezone() or DEFAULT_TIMEZONE


def get_timezone(name: str | None = None) -> ZoneInfo:
    """Return a ZoneInfo for the resolved timezone, falling back to UTC
    when the name is invalid/unknown."""
    resolved = timezone_name(name)
    try:
        return ZoneInfo(resolved)
    except ZoneInfoNotFoundError:
        log.warning("[bioclock] unknown timezone %r, falling back to UTC", resolved)
        return ZoneInfo("UTC")


def local_now(name: str | None = None) -> datetime:
    """Timezone-aware 'now' for the resolved timezone (app default unless
    an override is given)."""
    return datetime.now(get_timezone(name))


def current_datetime_block(name: str | None = None) -> str:
    """Rendered <current_datetime> block for injection into any system prompt.

    States the local wall time together with its UTC offset and the matching
    UTC instant, plus an explicit quote-exactly instruction. Small models
    otherwise "recompute" the time while composing a reply — keeping the
    minutes, drifting the hour, and pairing a local hour with the UTC date
    (or vice versa). The offset lets the model sanity-check itself instead
    of doing zone arithmetic from a city name.
    """
    now = local_now(name)
    utc = utc_now()
    offset = now.strftime("%z")
    offset_h = f"UTC{offset[:3]}:{offset[3:]}" if len(offset) == 5 else "UTC"
    return (
        "<current_datetime>\n"
        f"Today is {now.strftime('%A, %B %d, %Y')}. "
        f"Local time ({timezone_name(name)}, {offset_h}): "
        f"{now.strftime('%I:%M %p')}.\n"
        f"UTC: {utc.strftime('%Y-%m-%d %H:%M')}.\n"
        "When you mention the time or date, quote the local time above "
        "exactly as given. Do not convert between zones yourself.\n"
        "</current_datetime>"
    )


def utc_now() -> datetime:
    """Timezone-aware UTC now for persisted timestamps."""
    return datetime.now(timezone.utc)


def monotonic_now() -> float:
    """Monotonic seconds for durations, cooldowns, and polling intervals."""
    return time.monotonic()


def sleep_seconds(seconds: float) -> None:
    """Sleep for a duration; centralized so ticker/polling loops share one clock module."""
    time.sleep(max(0.0, float(seconds)))


def wait_seconds(event: threading.Event, seconds: float) -> bool:
    """Wait on an event for a duration using the centralized clock API."""
    return event.wait(timeout=max(0.0, float(seconds)))
