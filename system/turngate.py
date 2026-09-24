"""
system/turngate.py

Single process-wide gate shared by orchestrate.py's interactive turn loop
and schedule.py's background scheduler daemon.

Why this exists:
    Aiko runs one model at a time (single llama-server, single AikoMemorize/
    think singleton) — there is no per-user instance. The interactive main
    loop is already single-threaded, so user-vs-user turns are naturally
    serialized via the shared input queue. The scheduler daemon, however,
    runs on its own thread ("aiko-schedule") and fires jobs — some of which
    call directly into the same memorize/think singletons — completely
    unsynchronized with whatever the main loop is doing.

    AIKO_BUSY_LOCK is the one thing both sides take before touching shared
    state (switch_user, inference, memory writes). It makes explicit what
    the main loop already had implicitly (only one turn in flight), and
    extends that same "wait your turn" guarantee to background scheduled
    jobs: a job due mid-turn waits for the turn to finish, and a new turn
    arriving mid-job waits for the job to finish.

    Use an RLock (not a plain Lock) because a single logical "holder" may
    legitimately re-enter — e.g. the scheduler thread nesting a call that
    re-acquires the gate for a sub-step, or a job runner calling into a
    helper that also acquires it defensively.

Timeout (Phase 1 — guard the box):
    The bare lock had no timeout: one wedged turn or scheduled job (e.g. an
    LLM HTTP call hanging past its own timeout) blocked every later turn
    and every job forever. Prefer ``acquire_busy()`` / ``release_busy()``
    over touching AIKO_BUSY_LOCK directly: acquisition gives up after
    ``AIKO_BUSY_TIMEOUT_S`` (default 600s) and reports who is holding the
    lock and for how long, so a wedged holder is loud instead of silent.
    Callers must skip their work (not run unprotected) when acquisition
    fails.

Usage:
    from system.turngate import acquire_busy, release_busy, holder_desc

    if not acquire_busy(owner="scheduler:daily"):
        log.error("skipping job — %s", holder_desc())
        return
    try:
        ...
    finally:
        release_busy()
"""
from __future__ import annotations

import contextlib
import math
import threading
import time

from system.config import env_float
from system.log import get_logger

log = get_logger(__name__)

AIKO_BUSY_LOCK = threading.RLock()

#: Seconds to wait for the busy gate before giving up. A turn or job that
#: cannot acquire the gate within this window is skipped, never run
#: unprotected. Override with AIKO_BUSY_TIMEOUT_S (0 = wait forever,
#: preserving the old blocking behaviour).
BUSY_TIMEOUT_S: float = env_float("AIKO_BUSY_TIMEOUT_S", 600.0)
if not math.isfinite(BUSY_TIMEOUT_S) or BUSY_TIMEOUT_S < 0:
    log.warning("Invalid AIKO_BUSY_TIMEOUT_S; using 600s default")
    BUSY_TIMEOUT_S = 600.0

_meta_lock = threading.Lock()
_holder_ident: int | None = None
_holder_name: str = ""
_holder_since: float = 0.0
_holder_depth: int = 0


def holder_desc() -> str:
    """Human-readable description of the current lock holder."""
    with _meta_lock:
        if _holder_ident is None:
            return "busy lock is free"
        held_for = max(0.0, time.monotonic() - _holder_since)
        return (
            f"busy lock held by thread '{_holder_name}' "
            f"(ident {_holder_ident}) for {held_for:.0f}s "
            f"(depth {_holder_depth})"
        )


def acquire_busy(owner: str = "", timeout: float | None = None) -> bool:
    """Acquire the busy gate, giving up after ``timeout`` seconds.

    Returns True when the gate is held by the caller (re-entrant: a thread
    that already holds it succeeds immediately and bumps the depth).
    Returns False on timeout — the caller must skip its work, never
    proceed unprotected. A timeout is logged at error level with the
    current holder's identity so a wedged holder is visible.
    """
    wait = BUSY_TIMEOUT_S if timeout is None else timeout
    if not math.isfinite(wait) or wait < 0:
        raise ValueError("busy-gate timeout must be finite and non-negative")
    if wait and wait > 0:
        ok = AIKO_BUSY_LOCK.acquire(timeout=wait)
    else:
        AIKO_BUSY_LOCK.acquire()
        ok = True
    if not ok:
        log.error(
            "busy-gate timeout after %.0fs for %s — %s; skipping work",
            wait, owner or "unnamed owner", holder_desc(),
        )
        return False
    global _holder_ident, _holder_name, _holder_since, _holder_depth
    ident = threading.get_ident()
    with _meta_lock:
        if _holder_ident != ident:
            _holder_ident = ident
            try:
                _holder_name = threading.current_thread().name
            except Exception:
                _holder_name = "?"
            _holder_since = time.monotonic()
            _holder_depth = 1
        else:
            _holder_depth += 1
    return True


def release_busy() -> None:
    """Release one level of the busy gate (pairs with ``acquire_busy``)."""
    global _holder_ident, _holder_name, _holder_since, _holder_depth
    ident = threading.get_ident()
    with _meta_lock:
        if _holder_ident == ident:
            _holder_depth -= 1
            if _holder_depth <= 0:
                _holder_ident = None
                _holder_name = ""
                _holder_since = 0.0
                _holder_depth = 0
        # else: released by a non-holder (or raw AIKO_BUSY_LOCK use);
        # still forward the release so the underlying RLock stays balanced.
    AIKO_BUSY_LOCK.release()


@contextlib.contextmanager
def busy_gate(owner: str = "", timeout: float | None = None):
    """Context manager yielding True when the gate was acquired.

    ``with busy_gate(owner="scheduler:daily") as acquired:`` — check
    ``acquired`` and skip the guarded work when False.
    """
    acquired = acquire_busy(owner=owner, timeout=timeout)
    try:
        yield acquired
    finally:
        if acquired:
            release_busy()
