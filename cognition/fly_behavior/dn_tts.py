"""Apply DN vigor to TTS (Stage 3 / A–H F).

Reads NeuralState motor_vigor / action_drive and maps them onto Speaker
rate/volume. Safe no-op if speaker or state is missing.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("aiko.fly.dn_tts")


def apply_dn_prosody(speaker, *, user_id: str | None = None) -> dict:
    out = {"applied": False, "rate": 1.0, "volume": 1.0}
    if (os.getenv("MEMORY_FLYDN_MODE", "off") or "off").strip().lower() != "live":
        return out
    try:
        from cognition.neural_state import peek_neural_state
        st = peek_neural_state(user_id)
        if st is None:
            return out
        raw_vigor = getattr(st, "motor_vigor", 1.0)
        raw_drive = getattr(st, "action_drive", 0.5)
        vigor = float(1.0 if raw_vigor is None else raw_vigor)
        drive = float(0.5 if raw_drive is None else raw_drive)
        rate = max(0.85, min(1.15, 0.92 + 0.18 * vigor * (0.5 + 0.5 * drive)))
        vol = max(0.75, min(1.15, 0.90 + 0.20 * drive))
        out["rate"] = round(rate, 3)
        out["volume"] = round(vol, 3)
        if hasattr(speaker, "set_expression"):
            speaker.set_expression(rate=rate, volume=vol)
            out["applied"] = True
        elif hasattr(speaker, "set_speech_rate"):
            speaker.set_speech_rate(rate)
            out["applied"] = True
    except Exception as exc:
        log.debug("apply_dn_prosody skipped: %s", exc)
    return out
