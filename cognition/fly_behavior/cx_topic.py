"""CX topic drive (Stage 5 — semantic).

Replaces SHA-1 observe-only heading with Harrier → 8-D → continuous heading.
Does NOT step FlyCompass (attention already steps once per turn). Stashes
semantic features for cx_features.blend_cx_features and records influence.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("aiko.fly.cx_topic")


def _mode() -> str:
    try:
        from system.config import env_str
        return env_str("MEMORY_FLYCX_MODE", "off").strip().lower()
    except Exception:
        return (os.getenv("MEMORY_FLYCX_MODE") or "off").strip().lower()


def apply_topic_drive(text: str, *, user_id: str | None = None) -> dict:
    out = {
        "applied": False,
        "heading": None,
        "sharpness": None,
        "reason": "",
        "mode": _mode(),
    }
    t = (text or "").strip()
    if len(t) < 4:
        out["reason"] = "too_short"
        return out
    try:
        from cognition.fly_behavior.semantic_features import (
            heading_from_features,
            semantic_features,
            sharpness_from_features,
        )

        try:
            from cognition.fly_behavior.cx_blend_install import install as _install_cx_blend

            _install_cx_blend(user_id)
        except Exception:
            pass
        feats = semantic_features(t, user_id=user_id)
        heading = heading_from_features(feats)
        sharp = sharpness_from_features(feats)
        out["heading"] = round(heading, 2)
        out["sharpness"] = round(sharp, 4)
        out["applied"] = True
        out["reason"] = "semantic"
        try:
            from cognition.neural_state import get_neural_state

            st = get_neural_state(user_id)
            if out["mode"] == "live" and hasattr(st, "publish_cx"):
                try:
                    st.publish_cx(
                        heading_deg=heading,
                        sharpness=sharp,
                        source="cx_topic_semantic",
                    )
                except TypeError:
                    st.publish_cx(heading_deg=heading, sharpness=sharp)
            st.record_influence(
                {
                    "kind": "cx_topic",
                    "heading": out["heading"],
                    "sharpness": out["sharpness"],
                    "preview": t[:40],
                    "mode": "semantic",
                }
            )
        except Exception:
            pass
    except Exception as exc:
        log.debug("apply_topic_drive skipped: %s", exc)
        out["reason"] = "error"
    return out
