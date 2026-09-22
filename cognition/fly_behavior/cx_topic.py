"""CX topic drive (Stage 3 / A–H D).

Nudges compass heading from the current utterance tokens so focus is not
only leftover sleep/drowsiness state. Best-effort; never raises.
"""
from __future__ import annotations

import hashlib
import logging

log = logging.getLogger("aiko.fly.cx_topic")


def apply_topic_drive(text: str, *, user_id: str | None = None) -> dict:
    out = {"applied": False, "heading": None}
    t = (text or "").strip()
    if len(t) < 4:
        return out
    try:
        from cognition.fly_registry import get_flycx
        cx = get_flycx(user_id)
        if cx is None:
            return out
        digest = hashlib.sha1(t.encode("utf-8", errors="ignore")).digest()
        heading = int.from_bytes(digest[:2], "little") % 360
        if hasattr(cx, "bump") or hasattr(cx, "set_heading") or hasattr(cx, "steer"):
            for name in ("steer", "set_heading", "bump"):
                fn = getattr(cx, name, None)
                if callable(fn):
                    try:
                        fn(heading)
                    except TypeError:
                        try:
                            fn(heading / 360.0)
                        except Exception:
                            pass
                    break
        out["heading"] = heading
        out["applied"] = True
        try:
            from cognition.neural_state import get_neural_state
            st = get_neural_state(user_id)
            if hasattr(st, "publish_cx"):
                st.publish_cx(heading=heading / 360.0, source="topic")
            st.record_influence({"kind": "cx_topic", "heading": heading, "preview": t[:40]})
        except Exception:
            pass
    except Exception as exc:
        log.debug("apply_topic_drive skipped: %s", exc)
    return out
