"""CX topic drive (Stage 3 / A–H D).

Observe-only topic diagnostic for the turn. FlyCompass exposes only
step()/reset()/readout()/summary() and attention already advances it once
per turn (flycx_state_for_record), so this must NOT step the compass a
second time — that would double-count sleep pressure. Best-effort;
never raises.
"""
from __future__ import annotations

import hashlib
import logging

log = logging.getLogger("aiko.fly.cx_topic")


def apply_topic_drive(text: str, *, user_id: str | None = None) -> dict:
    out = {"applied": False, "heading": None, "reason": ""}
    t = (text or "").strip()
    if len(t) < 4:
        out["reason"] = "too_short"
        return out
    try:
        digest = hashlib.sha1(t.encode("utf-8", errors="ignore")).digest()
        heading = int.from_bytes(digest[:2], "little") % 360
        # No compass mutation here: FlyCompass has no steer/set_heading/bump
        # API, and the shared compass is already stepped once per turn by
        # attention.flycx_state_for_record. Record the diagnostic heading
        # for the causal trail only.
        out["heading"] = heading
        out["applied"] = False
        out["reason"] = "observe_only"
        try:
            from cognition.neural_state import get_neural_state
            st = get_neural_state(user_id)
            st.record_influence({"kind": "cx_topic", "heading": heading, "preview": t[:40], "mode": "observe"})
        except Exception:
            pass
    except Exception as exc:
        log.debug("apply_topic_drive skipped: %s", exc)
        out["reason"] = "error"
    return out
