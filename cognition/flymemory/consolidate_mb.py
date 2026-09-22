"""Sleep consolidation for MB plastic weights (Stage 4).

When sleep_pressure is high (or forced), shrink weak KC→MBON deltas and
slightly protect strong ones. Cheap, bounded, never raises.

Gated on MEMORY_FLYMB_MODE=live and FLY_MB_CONSOLIDATE=1.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("aiko.flymemory.consolidate_mb")


def _enabled() -> bool:
    try:
        return (os.getenv("FLY_MB_CONSOLIDATE", "1") or "1").strip().lower() not in (
            "0", "off", "false", "no",
        )
    except Exception:
        return True


def _mb_mode() -> str:
    try:
        from system.config import env_str
        return env_str("MEMORY_FLYMB_MODE", "off").strip().lower()
    except Exception:
        return (os.getenv("MEMORY_FLYMB_MODE") or "off").strip().lower()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def consolidate(
    user_id: str | None = None,
    *,
    sleep_pressure: float | None = None,
    force: bool = False,
) -> dict:
    """Run one consolidation pass on MB._plastic."""
    out: dict = {"ran": False, "mode": _mb_mode(), "before": 0.0, "after": 0.0, "n_touched": 0}
    if not _enabled() or out["mode"] != "live":
        out["reason"] = "disabled_or_off"
        return out
    try:
        if sleep_pressure is None:
            from cognition.neural_state import get_neural_state
            sleep_pressure = float(get_neural_state(user_id).sleep_pressure or 0.0)
        sp = float(sleep_pressure)
        out["sleep_pressure"] = round(sp, 3)
        min_sp = _float_env("FLY_MB_CONSOLIDATE_MIN_SLEEP", 0.55)
        if not force and sp < min_sp:
            out["reason"] = "sleep_low"
            return out

        from cognition.fly_registry import get_flymb, get_fly_store
        import numpy as np

        mb = get_flymb(user_id)
        if mb is None or not hasattr(mb, "_plastic"):
            out["reason"] = "mb_unavailable"
            return out
        plastic = np.asarray(mb._plastic, dtype=float)
        out["before"] = float(np.abs(plastic).sum())
        weak_thr = _float_env("FLY_MB_WEAK_THR", 0.02)
        strong_thr = _float_env("FLY_MB_STRONG_THR", 0.12)
        weak_decay = _float_env("FLY_MB_WEAK_DECAY", 0.85)
        mid_decay = _float_env("FLY_MB_MID_DECAY", 0.95)
        strong_keep = _float_env("FLY_MB_STRONG_KEEP", 0.99)

        abs_w = np.abs(plastic)
        new = plastic.copy()
        weak = abs_w < weak_thr
        strong = abs_w >= strong_thr
        mid = ~(weak | strong)
        new[weak] *= weak_decay
        new[mid] *= mid_decay
        new[strong] *= strong_keep
        new[np.abs(new) < 1e-6] = 0.0
        mb._plastic = np.clip(new, -0.5, 0.5)
        out["after"] = float(np.abs(mb._plastic).sum())
        out["n_touched"] = int((weak | mid | strong).sum())
        out["ran"] = True
        out["reason"] = "ok"
        try:
            store = get_fly_store(user_id)
            if store is not None:
                store.flush_mb(mb)
        except Exception:
            pass
        try:
            from cognition.neural_state import get_neural_state
            get_neural_state(user_id).record_influence(
                {
                    "kind": "mb_consolidate",
                    "sleep_pressure": out["sleep_pressure"],
                    "before": round(out["before"], 4),
                    "after": round(out["after"], 4),
                }
            )
        except Exception:
            pass
        log.debug(
            "mb consolidate user=%s sleep=%.2f mass %.4f→%.4f",
            (user_id or "default")[:12],
            sp,
            out["before"],
            out["after"],
        )
    except Exception as exc:
        log.debug("mb consolidate skipped: %s", exc)
        out["reason"] = str(exc)
    return out
