"""Sleep-pressure → maintenance / dream urgency (MaleCNS clock/CX motif).

Does not replace the nightly schedule. When MEMORY_FLYSLEEP_MODE is live and
sleep_pressure is high, marks NeuralState and optionally expands dream boost
weight. Schedule still owns when dream() runs; this biases *how much*.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("aiko.fly_behavior.sleep_sched")

_maintenance_users: set[str] = set()
_maintenance_lock = threading.RLock()


def _uid(user_id: str | None) -> str:
    return (user_id or "").strip() or "default"


def _mode() -> str:
    try:
        from system.config import env_str
        return env_str("MEMORY_FLYSLEEP_MODE", "off").strip().lower()
    except Exception:
        return "off"


def dream_boost_multiplier(user_id: str | None = None) -> float:
    """1.0 normal; up to ~1.5 when live + high sleep pressure."""
    mode = _mode()
    if mode not in ("shadow", "live"):
        return 1.0
    try:
        from cognition.neural_state import get_neural_state
        sp = float(get_neural_state(user_id).sleep_pressure or 0.0)
    except Exception:
        return 1.0
    mult = 1.0 + max(0.0, min(0.5, (sp - 0.4) * 1.0))
    if mode == "shadow":
        log.debug("flysleep dream_boost mode=shadow sleep=%.2f would_mult=%.2f", sp, mult)
        return 1.0
    log.debug("flysleep dream_boost mode=live sleep=%.2f mult=%.2f", sp, mult)
    return mult


def should_prefer_maintenance(user_id: str | None = None) -> bool:
    """True when live sleep says run heavier consolidate / lighter agent work."""
    mode = _mode()
    if mode not in ("shadow", "live"):
        return False
    try:
        from cognition.fly_behavior.turn import maintenance_level

        level = maintenance_level(user_id)
        predicted_level = level
        if mode == "shadow":
            from cognition.fly_behavior.turn import _maintenance_level_from_pressure
            from cognition.neural_state import get_neural_state

            sleep_pressure = float(get_neural_state(user_id).sleep_pressure or 0.0)
            predicted_level = _maintenance_level_from_pressure(sleep_pressure)
    except Exception:
        level = "normal"
        predicted_level = "normal"
    if mode == "shadow":
        log.debug(
            "flysleep prefer_maintenance mode=shadow would=%s action_level=%s",
            predicted_level,
            level,
        )
        return False
    return level in ("reduced", "maintenance")


def maybe_consolidate_mb(user_id: str | None = None) -> dict:
    """Run MB consolidation once per transition into maintenance."""
    out = {"ran": False}
    try:
        key = _uid(user_id)
        if not should_prefer_maintenance(user_id):
            with _maintenance_lock:
                _maintenance_users.discard(key)
            out["reason"] = "not_maintenance"
            return out
        with _maintenance_lock:
            if key in _maintenance_users:
                out["reason"] = "already_consolidated"
                return out
            _maintenance_users.add(key)
        from cognition.flymemory.consolidate_mb import consolidate
        return consolidate(user_id)
    except Exception as exc:
        log.debug("maybe_consolidate_mb skipped: %s", exc)
        out["reason"] = str(exc)
        return out
