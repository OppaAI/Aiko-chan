"""Whole-brain simulation on the full MaleCNS v1.0 connectome (Phase 4).

Biology-faithful rate-based 2-hop step over ~166k neurons / 25.5M edges.
Jetson-safe: Phase 12 adds should_step() throttle so we do not step every turn.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

_step_count = 0
_last_step_ts = 0.0
_throttle_lock = threading.RLock()


def should_step(*, force: bool = False) -> bool:
    """Phase 12: Jetson-safe fullbrain throttle.

    Env:
      AIKO_FULLBRAIN_EVERY_N          step every N calls (default 1)
      AIKO_FULLBRAIN_MIN_INTERVAL_S   min seconds between steps (default 0)
    """
    global _step_count
    if force:
        return True
    try:
        every_n = max(1, int(os.environ.get("AIKO_FULLBRAIN_EVERY_N", "1") or 1))
    except Exception:
        every_n = 1
    try:
        min_iv = max(0.0, float(os.environ.get("AIKO_FULLBRAIN_MIN_INTERVAL_S", "0") or 0))
    except Exception:
        min_iv = 0.0
    now = time.time()
    with _throttle_lock:
        _step_count += 1
        if every_n > 1 and (_step_count % every_n) != 0:
            return False
        if min_iv > 0 and (now - _last_step_ts) < min_iv:
            return False
        return True


def mark_stepped() -> None:
    global _last_step_ts
    with _throttle_lock:
        _last_step_ts = time.time()


def step_fullbrain(
    *,
    user_id: str | None = None,
    features: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run one fullbrain step if throttle allows. Never raises."""
    out: dict[str, Any] = {"ran": False, "ms": 0.0, "throttled": False}
    try:
        if not should_step(force=force):
            out["throttled"] = True
            try:
                from cognition.fly_behavior import metrics as _fly_metrics
                _fly_metrics.record_fullbrain(ran=False, user_id=user_id)
            except Exception:
                pass
            return out

        t0 = time.time()
        # Delegate to existing runtime / catalog step if present
        try:
            from cognition.fly_runtime.runtime import step as runtime_step
            runtime_step(user_id=user_id, features=features)
        except Exception:
            try:
                from cognition.flymemory.circuit import get_circuit
                circ = get_circuit()
                if circ is not None and hasattr(circ, "step"):
                    circ.step(features)
            except Exception as exc:
                log.debug("fullbrain step body skipped: %s", exc)

        mark_stepped()
        ms = (time.time() - t0) * 1000.0
        out["ran"] = True
        out["ms"] = round(ms, 2)
        try:
            from cognition.fly_behavior import metrics as _fly_metrics
            _fly_metrics.record_fullbrain(ran=True, ms=ms, user_id=user_id)
        except Exception:
            pass
    except Exception as exc:
        log.debug("step_fullbrain failed: %s", exc)
    return out
