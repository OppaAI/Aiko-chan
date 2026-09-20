"""
system/resource.py
Cooperative RAM hygiene for the 8GB Jetson box.

CPython rarely returns freed heap to the OS mid-process (arena retention),
so RSS only ever grows until process death. This module gives Aiko a broom:
collect cyclic garbage, then ask glibc to hand fully-free arenas back via
``malloc_trim``. Live objects (fly catalog, embedding matrices, SQLite
handles) are untouched — this reclaims *fragmentation*, not working set.
Expect tens-to-hundreds of MB, not GBs.

Throttle: at most one release per ``RAM_RELEASE_MIN_INTERVAL_S`` (default
300s) no matter how many call sites fire, so per-turn hooks stay cheap.
"""
from __future__ import annotations

import ctypes
import gc
import sys
import threading
import time

from system.config import env_float
from system.log import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_last_release_at: float = 0.0


def rss_mb() -> float | None:
    """Current process RSS in MB, or None when unreadable."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        pass
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        with open("/proc/self/stat") as handle:
            page_size = 4096
            try:
                import os

                page_size = os.sysconf("SC_PAGE_SIZE")
            except Exception:
                pass
        return pages * page_size / (1024 * 1024)
    except Exception:
        return None


def _malloc_trim() -> bool:
    """Ask glibc to return fully-free arenas. No-op off Linux/glibc."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return bool(libc.malloc_trim(0))
    except Exception:
        return False


def release_ram(reason: str = "") -> dict:
    """Collect garbage + trim arenas, throttled. Returns a small report dict.

    Always safe to call: failures are swallowed (best-effort hygiene must
    never break a turn), and calls inside the throttle window are skipped.
    """
    global _last_release_at
    interval = max(0.0, env_float("RAM_RELEASE_MIN_INTERVAL_S", 300.0))
    now = time.monotonic()
    with _lock:
        if interval > 0 and now - _last_release_at < interval:
            return {"ok": False, "skipped": "throttled"}
        _last_release_at = now

    before = rss_mb()
    try:
        collected = gc.collect()
    except Exception:
        collected = -1
    trimmed = _malloc_trim()
    after = rss_mb()
    freed = (before - after) if (before is not None and after is not None) else None
    log.info(
        "ram: released after %s — rss %.0f→%.0f MB (freed %s), gc=%s trim=%s",
        reason or "unspecified",
        before if before is not None else -1,
        after if after is not None else -1,
        f"{freed:.0f} MB" if freed is not None else "n/a",
        collected,
        trimmed,
    )
    return {"ok": True, "before_mb": before, "after_mb": after,
            "freed_mb": freed, "gc_collected": collected, "trimmed": trimmed}
