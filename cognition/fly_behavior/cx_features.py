"""Compose CX step inputs: affect state + semantic text + motion (Stage 5)."""
from __future__ import annotations

import logging
import os

log = logging.getLogger("aiko.fly.cx_features")


def blend_cx_features(
    affect_feats: list[float],
    pen: float,
    fatigue: float,
    text: str,
    user_id: str | None = None,
) -> tuple[list[float], float, float]:
    """Blend semantic 8-D into affect features; motion boosts pen_drive."""
    mode = (os.getenv("MEMORY_FLYCX_MODE") or "off").strip().lower()
    if mode != "live":
        return affect_feats, pen, fatigue

    feats = [float(x) for x in affect_feats]
    if len(feats) < 8:
        feats = (feats + [0.0] * 8)[:8]
    try:
        from cognition.fly_behavior.semantic_features import (
            blend_weight,
            last_semantic_features,
            semantic_features,
        )
        sem = (
            semantic_features(text, user_id=user_id)
            if (text or "").strip()
            else last_semantic_features(user_id)
        )
        if sem is not None and len(sem) >= 8:
            w = blend_weight()
            feats = [(1.0 - w) * feats[i] + w * float(sem[i]) for i in range(8)]
    except Exception as exc:
        log.debug("semantic blend skipped: %s", exc)

    try:
        from cognition.neural_state import peek_neural_state
        st = peek_neural_state(user_id)
        if st is not None:
            motion = float(getattr(st, "motion_salience", 0.0) or 0.0)
            mw = float(os.getenv("FLY_CX_MOTION_W", "0.35"))
            pen = max(-1.0, min(1.0, float(pen) + mw * motion))
    except Exception as exc:
        log.debug("motion pen blend skipped: %s", exc)

    return feats, float(pen), float(fatigue)
