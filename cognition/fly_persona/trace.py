"""Phase 11 — explainability trace: trait → circuit parameter → effect.

Each turn may emit one JSON-serializable record describing how the
persona modulated the circuits. This is data for a future Fly Studio
panel ("why did Aiko behave differently"); the panel itself is not
part of this phase.

The record answers three questions:
  1. What were the traits?            (state.describe)
  2. What gains did they produce?     (persona_gains, and whether they
                                      were applied or shadowed)
  3. What circuit parameters changed? (the effective per-drive gain /
                                      decay / threshold factors)

Never raises. Recording into NeuralState's influence ring is best-effort.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger("aiko.fly.persona.trace")


def explain_turn(user_id: str | None = None) -> dict:
    """Build the JSON-serializable persona trace for this turn."""
    rec: dict = {
        "kind": "persona",
        "ts": time.time(),
        "traits": {},
        "gains": {},
        "applied": False,
        "mode": "off",
        "effects": [],
    }
    try:
        from cognition.fly_persona import persona_mode
        from cognition.fly_persona.modulators import persona_gains
        from cognition.fly_persona.state import get_personality

        mode = persona_mode()
        rec["mode"] = mode
        st = get_personality(user_id)
        rec["traits"] = {k: round(v, 4) for k, v in st.traits.items()}
        gains = persona_gains(user_id) or {}
        rec["gains"] = gains
        rec["applied"] = mode == "live"

        cx = gains.get("cx", {}) if isinstance(gains, dict) else {}
        # Human-readable effect lines: only non-identity modulations.
        for drive, params in cx.items():
            if not isinstance(params, dict):
                continue
            for p, v in params.items():
                try:
                    f = float(v)
                except Exception:
                    continue
                if abs(f - 1.0) >= 0.01:
                    direction = "raised" if f > 1.0 else "lowered"
                    rec["effects"].append(
                        f"cx.{drive}.{p} {direction} ×{f:.2f}"
                    )
        for chan in ("mb_plasticity", "gf_urgency", "dn_vigor"):
            try:
                f = float(gains.get(chan, 1.0))
            except Exception:
                continue
            if abs(f - 1.0) >= 0.01:
                direction = "raised" if f > 1.0 else "lowered"
                rec["effects"].append(f"{chan} {direction} ×{f:.2f}")
        if mode != "live" and rec["effects"]:
            rec["effects"].append("(shadow: computed, not applied)")
    except Exception as exc:
        log.debug("persona explain_turn skipped: %s", exc)
        rec["error"] = str(exc)[:120]
    return rec


def record_persona_trace(user_id: str | None = None) -> dict:
    """Build the trace and append it to the NeuralState influence ring."""
    rec = explain_turn(user_id)
    try:
        from cognition.neural_state import get_neural_state

        get_neural_state(user_id).record_influence(rec)
    except Exception as exc:
        log.debug("persona trace record skipped: %s", exc)
    return rec
