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
        return True
    except Exception as exc:
        log.debug("eligibility record skipped: %s", exc)
        return False


def assign_credit(user_id: str | None, reward: float) -> dict:
    """Spread `reward` over prior trail steps with exponential decay.

    Age-1 (previous turn) gets `reward*decay`, age-2 `reward*decay^2`, etc.
    Age-0 (current text) is handled by the caller's immediate reinforce, so
    it is intentionally skipped here to avoid double-crediting.
    """
    out: dict = {"taught": False, "steps": 0, "delta": 0.0, "reward": float(reward)}
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
        from cognition.fly_registry import get_flymb, get_fly_store

        mb = get_flymb(user_id)
        if mb is None:
            out["reason"] = "mb_unavailable"
            return out
        decay = _decay()
        with _lock:
            trail = list(_TRACES.get(_uid(user_id), ()))
        total = 0.0
        credited = 0
        for age, (kc, _ts) in enumerate(reversed(trail), start=1):
            w = decay ** age
            if w < 0.05:
                break
            try:
                try:
                    from cognition.flymemory.dopamine import pulse
                    res = pulse(r, user_id=user_id, kc=kc, weight=w, source="eligibility")
                    total += float(res.get("delta") or 0.0)
                except Exception:
                    total += float(mb.reinforce(kc, r * w) or 0.0)
                credited += 1
            except Exception:
                continue
        try:
            store = get_fly_store(user_id)
            if store is not None:
                store.flush_mb(mb)
        except Exception:
            pass
        out.update({"taught": credited > 0, "steps": credited, "delta": round(total, 4), "reason": "ok"})
        with _lock:
            _LAST[_uid(user_id)] = {"reward": r, "steps": credited, "delta": out["delta"]}
        log.debug("eligibility credit user=%s reward=%+.2f steps=%d delta=%.4f",
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
    return last


def clear(user_id: str | None = None) -> None:
    """Drop trail state (tests / identity switch)."""
    with _lock:
        _TRACES.pop(_uid(user_id), None)
        _LAST.pop(_uid(user_id), None)
