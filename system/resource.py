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
import os
import sys
import threading
import time

from system.config import env_float, env_str
from system.log import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_last_release_at: float = 0.0

#: Cmdline substrings (case-insensitive) identifying Aiko's companion
#: processes: the llama-server instance(s) serving chat (/v1) and embeddings
#: (/embedding), the MioTTS synthesis server, and optional sherpa-onnx
#: helpers. These run outside the Python process, so system/resource.py's
#: old self-RSS-only view left them completely unwatched — the exact
#: processes most likely to OOM the Jetson. Override with
#: AIKO_COMPANION_PATTERNS (comma-separated).
COMPANION_PATTERNS: tuple[str, ...] = tuple(
    p.strip().lower()
    for p in env_str(
        "AIKO_COMPANION_PATTERNS", "llama-server,mio,sherpa-onnx"
    ).split(",")
    if p.strip()
)


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


def _proc_cmdline(pid: int) -> str:
    """Full cmdline of a pid, or '' when unreadable."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(
                "utf-8", "replace").strip().lower()
    except Exception:
        return ""


def _proc_rss_mb(pid: int) -> float | None:
    """RSS of a pid in MB via /proc, or None when unreadable."""
    try:
        with open(f"/proc/{pid}/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return None


def companion_rss_mb(
    patterns: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """RSS (MB) per companion process, keyed by a short label.

    Matches companion processes by cmdline substring (see
    COMPANION_PATTERNS); the current Python process is always excluded.
    Best-effort: never raises, returns {} when nothing matches or the
    process table is unreadable. Prefers psutil, falls back to /proc.
    """
    pats = tuple(patterns) if patterns else COMPANION_PATTERNS
    if not pats:
        return {}
    self_pid = os.getpid()
    found: dict[str, float] = {}
    try:
        import psutil

        for proc in psutil.process_iter(
            ["pid", "cmdline", "memory_info"]
        ):
            try:
                pid = proc.info["pid"]
                if pid == self_pid:
                    continue
                cmdline = " ".join(
                    proc.info.get("cmdline") or []).lower()
                if not cmdline:
                    continue
                for pat in pats:
                    if pat in cmdline:
                        rss = float(
                            proc.info["memory_info"].rss) / (1024 * 1024)
                        label = f"{pat}:{pid}"
                        found[label] = round(rss, 1)
                        break
            except Exception:
                continue
        return found
    except Exception:
        pass
    # /proc fallback (Linux, no psutil).
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except Exception:
        return {}
    for pid in pids:
        if pid == self_pid:
            continue
        cmdline = _proc_cmdline(pid)
        if not cmdline:
            continue
        for pat in pats:
            if pat in cmdline:
                rss = _proc_rss_mb(pid)
                if rss is not None:
                    found[f"{pat}:{pid}"] = round(rss, 1)
                break
    return found


def box_rss_mb() -> dict:
    """Combined RAM snapshot of the whole Aiko box.

    Returns {"self_mb", "companions": {label: mb}, "companions_mb",
    "total_mb"} — the companions this module previously never watched
    (llama-server, MioTTS, …). Best-effort: values may be None.
    """
    self_mb = rss_mb()
    companions = companion_rss_mb()
    companions_mb = round(sum(companions.values()), 1) if companions else 0.0
    total = (round(self_mb + companions_mb, 1)
             if self_mb is not None else None)
    return {
        "self_mb": round(self_mb, 1) if self_mb is not None else None,
        "companions": companions,
        "companions_mb": companions_mb,
        "total_mb": total,
    }
