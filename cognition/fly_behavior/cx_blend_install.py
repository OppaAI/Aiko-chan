"""Install semantic+motion blend on FlyCompass.step (Stage 5).

Avoids editing the large attention.py: wrap the per-identity compass once
so the existing flycx_state_for_record path picks up Harrier features and
T4/T5 motion automatically.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("aiko.fly.cx_blend_install")

_lock = threading.Lock()
_wrapped: set[str] = set()


def install(user_id: str | None = None) -> bool:
    """Wrap get_flycx(user_id).step once. Safe to call every turn."""
    key = (user_id or "").strip() or "default"
    with _lock:
        if key in _wrapped:
            return True
        try:
            from cognition.fly_registry import get_flycx

            cx = get_flycx(user_id)
            if cx is None or not hasattr(cx, "step"):
                return False
            if getattr(cx, "_stage5_blend_wrapped", False):
                _wrapped.add(key)
                return True
            orig = cx.step

            def step(features, pen_drive: float = 0.0, fatigue: float = 0.0):
                feats, pen, fat = features, pen_drive, fatigue
                try:
                    from cognition.fly_behavior.cx_features import blend_cx_features

                    feats, pen, fat = blend_cx_features(
                        list(features), pen_drive, fatigue, "", user_id
                    )
                except Exception:
                    pass
                return orig(feats, pen, fat)

            cx.step = step  # type: ignore[method-assign]
            cx._stage5_blend_wrapped = True
            _wrapped.add(key)
            log.debug("stage5 CX blend installed for %s", key[:12])
            return True
        except Exception as exc:
            log.debug("cx blend install skipped: %s", exc)
            return False
