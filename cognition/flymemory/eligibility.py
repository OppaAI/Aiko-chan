"""Minimal eligibility trace for delayed credit (Stage 4 temporal learning).

Immediate teaching (`reinforce()` on the current text) can't credit the
trajectory that led to an outcome ("search → bad result → retry → good
result → 'Perfect!'"). This module keeps a short per-user trail of recent
KC encodings; when a praise/correction/stop outcome arrives, the reward is
also applied to prior steps with exponential decay.

Scope is deliberately small: in-memory only (plastic deltas still persist
via the normal `flush_mb` path), bounded steps, decay-weighted, never
raises. Gated on `MEMORY_FLYMB_MODE=live` and `FLY_ELIGIBILITY=1`.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque

log = logging.getLogger("aiko.flymemory.eligibility")

_TRACES: dict[str, deque] = {}
_LAST: dict[str, dict] = {}
_lock = threading.RLock()


def _uid(user_id: str | None) -> str:
    return (user_id or "").strip() or "default"


def _mb_mode() -> str:
    try:
        from system.config import env_str
        return env_str("MEMORY_FLYMB_MODE", "off").strip().lower()
    except Exception:
        return (os.getenv("MEMORY_FLYMB_MODE") or "off").strip().lower()


def _enabled() -> bool:
    try:
        return (os.getenv("FLY_ELIGIBILITY", "1") or "1").strip().lower() not in (
            "0", "off", "false", "no",
        )
    except Exception:
        return True


def _steps() -> int:
    try:
        return max(1, min(12, int(os.getenv("FLY_ELIGIBILITY_STEPS", "5"))))
    except Exception:
        return 5


def _decay() -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv("FLY_ELIGIBILITY_DECAY", "0.7"))))
    except Exception:
        return 0.7


def record_step(user_id: str | None, text: str) -> bool:
    """Append the current turn's KC encoding to the trail. Returns stored?"""
    if not _enabled() or _mb_mode() != "live":
        return False
    t = (text or "").strip()
    if not t:
        return False
    try:
        from cognition.fly_registry import get_flymb
        from cognition.flymemory.circuit import text_features

        mb = get_flymb(user_id)
        if mb is None:
            return False
        kc = mb.encode(text_features(t))
        key = _uid(user_id)
        with _lock:
            buf = _TRACES.setdefault(key, deque())
            buf.append((kc.copy(), time.time()))
            while len(buf) > _steps():
                buf.popleft()
        # Phase 10A: mark the pattern's eligibility trace in the rate-based
        # credit engine (bounded per-user in-memory store, event-stepped).
        try:
            from cognition.flymemory import credit as _credit

            _credit.mark_trace(user_id, kc, value=1.0, scope="real")
        except Exception:
            pass
        return True
    except Exception as exc:
        log.debug("eligibility record skipped: %s", exc)
        return False


def assign_credit(user_id: str | None, reward: float) -> dict:
    """Credit `reward` backward through the marked eligibility traces.

    Phase 10A: one dopamine event over the already-marked traces —
    ``credit_event(user_id, reward, kc=None)``. The reward (consequence)
    multiplies each trace by its current eligibility (decayed since the
    action was marked); unrelated patterns have decayed to ~0 and learn
    nothing. The current text's own trace is the caller's business (the
    unified path marks it in its own event), so kc=None here on purpose.
    """
    out: dict = {"taught": False, "steps": 0, "delta": 0.0, "reward": float(reward), "flushed": False}
    if not _enabled() or _mb_mode() != "live":
        out["reason"] = "disabled_or_off"
        return out
    try:
        r = max(-1.0, min(1.0, float(reward)))
    except Exception:
        out["reason"] = "bad_reward"
        return out
    if r == 0.0:
        out["reason"] = "zero_reward"
        return out
    try:
        from cognition.fly_registry import get_fly_store
        from cognition.flymemory import credit as _credit

        res = _credit.credit_event(
            user_id, r, kc=None, text=None, source="eligibility", scope="real"
        )
        credited = int(res.get("n_applied_traces") or 0)
        total = float(res.get("delta") or 0.0)
        try:
            from cognition.fly_registry import get_flymb

            mb = get_flymb(user_id)
            store = get_fly_store(user_id)
            if res.get("applied") and store is not None and mb is not None:
                store.flush_mb(mb)
                out["flushed"] = True
        except Exception:
            pass
        out.update({"taught": bool(res.get("applied")), "steps": credited,
                    "delta": round(total, 4), "reason": str(res.get("reason") or "")})
        with _lock:
            _LAST[_uid(user_id)] = {"reward": r, "steps": credited, "delta": out["delta"]}
        log.debug("eligibility credit user=%s reward=%+.2f traces=%d delta=%.4f",
                  _uid(user_id)[:12], r, credited, total)
    except Exception as exc:
        log.debug("eligibility credit failed: %s", exc)
        out["reason"] = str(exc)
    return out


def stats(user_id: str | None = None) -> dict:
    """Last credit assignment + trail depth (for harness/observability)."""
    with _lock:
        trail = _TRACES.get(_uid(user_id), ())
        last = dict(_LAST.get(_uid(user_id), {}))
    last["trail_depth"] = len(trail)
    try:
        from cognition.flymemory import credit as _credit

        last["credit"] = _credit.stats(user_id, scope="real")
    except Exception:
        pass
    return last


def clear(user_id: str | None = None) -> None:
    """Drop trail state (tests / identity switch)."""
    with _lock:
        _TRACES.pop(_uid(user_id), None)
        _LAST.pop(_uid(user_id), None)
    try:
        from cognition.flymemory import credit as _credit

        _credit.clear(user_id, scope=None)
    except Exception:
        pass
