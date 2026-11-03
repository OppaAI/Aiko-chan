"""Apply DN vigor to TTS (Stage 6 — GF-aware).

Reads NeuralState motor_vigor / action_drive and maps them onto Speaker
rate/volume. Honors global GF cancel (skip/mute speech). Safe no-op if
speaker or state is missing.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("aiko.fly.dn_tts")


def apply_dn_prosody(speaker, *, user_id: str | None = None) -> dict:
    out = {"applied": False, "rate": 1.0, "volume": 1.0, "cancelled": False}
    if (os.getenv("MEMORY_FLYDN_MODE", "off") or "off").strip().lower() != "live":
        return out
    try:
        from cognition.fly_behavior.gf_global import should_cancel_tts
        if should_cancel_tts(user_id):
            out["cancelled"] = True
            out["rate"] = 0.0
            out["volume"] = 0.0
            for name in ("stop", "mute", "cancel"):
                fn = getattr(speaker, name, None)
                if callable(fn):
                    try:
                        fn()
                        out["applied"] = True
                        break
                    except Exception:
                        pass
            return out
    except Exception:
        pass
    try:
        from cognition.fly_behavior.dn_body import body_drive
        bd = body_drive(user_id=user_id)
        rate = float(bd.get("rate_mult") or 1.0)
        vol = float(bd.get("vol_mult") or 1.0)
        out["rate"] = round(rate, 3)
        out["volume"] = round(vol, 3)
        if hasattr(speaker, "set_expression"):
            speaker.set_expression(rate=rate, volume=vol)
            out["applied"] = True
        elif hasattr(speaker, "set_speech_rate"):
            speaker.set_speech_rate(rate)
            out["applied"] = True
        if hasattr(speaker, "set_volume"):
            try:
                speaker.set_volume(vol)
            except Exception:
                pass
    except Exception as exc:
        log.debug("apply_dn_prosody skipped: %s", exc)
    return out
