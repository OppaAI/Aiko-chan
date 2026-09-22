"""Apply DN vigor to TTS (Stage 3 / A–H F).

Reads NeuralState motor_vigor / action_drive and maps them onto Speaker
rate/volume. Safe no-op if speaker or state is missing.
"""
from __future__ import annotations

import logging

log = logging.getLogger("aiko.fly.dn_tts")


def apply_dn_prosody(speaker, *, user_id: str | None = None) -> dict:
    out = {"applied": False, "rate": 1.0, "volume": 1.0}
    try:
        from cognition.neural_state import peek_neural_state
        st = peek_neural_state(user_id)
        if st is None:
            return out
        vigor = float(getattr(st, "motor_vigor", 1.0) or 1.0)
        drive = float(getattr(st, "action_drive", 0.5) or 0.5)
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
