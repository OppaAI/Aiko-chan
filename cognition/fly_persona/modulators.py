"""Phase 11 — trait → circuit modulators.

NON-NEGOTIABLE: the persona affects neural INITIAL CONDITIONS AND GAINS,
never direct action scores. Every entry in the mapping below is a gain
(multiplier on an existing circuit parameter), a decay, or a threshold
factor. There is intentionally no path from a trait to a vote, a score,
or a candidate preference.

Mapping table (trait → circuit parameter):
  curiosity   → CX curiosity-drive gain ↑, threshold ↓ (novelty sensitivity);
                curiosity-drive decay ↑ (lingering novelty interest)
  exploration → CX heading decay ↑ (exploration persistence: the heading
                vector lingers, sustaining directed attention); curiosity
                decay ↑ slightly
  playfulness → CX engagement-drive gain ↑; DN vigor gain ↑;
                MB plasticity gain ↑ (rewarding novel interactions stick)
  attachment  → DN vigor gain ↑ (approach energy); MB plasticity gain ↑
                (socially rewarding interactions stick)
  calmness    → GF urgency gain ↓ (functionally a higher interrupt
                threshold: urgency must push harder to fire); CX urgency
                gain ↓; urgency decay ↑ (faster recovery from urgency)

All multipliers are clamped to [0.5, 1.5]. Combined channels (dn_vigor,
mb_plasticity) multiply their trait factors, then clamp.

persona_gains() computes the raw trait-derived gains (always safe to call:
pure math, no I/O). applied_gains() returns the identity gains unless
AIKO_FLY_PERSONA_MODE=live — shadow computes and logs but never modulates.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("aiko.fly.persona.modulators")

_GAIN_LO, _GAIN_HI = 0.5, 1.5


def _mult(trait_value: float, k: float) -> float:
    """Linear trait→gain map: 1 + k·(t−0.5)·2, clamped to [0.5, 1.5].

    k > 0: trait raises the gain. k < 0: trait lowers it (e.g. calmness
    on urgency). t = 0.5 (neutral) always yields exactly 1.0.
    """
    try:
        t = max(0.0, min(1.0, float(trait_value)))
        kk = float(k)
    except Exception:
        return 1.0
    return max(_GAIN_LO, min(_GAIN_HI, 1.0 + kk * (t - 0.5) * 2.0))


def persona_gains(user_id: str | None = None) -> dict:
    """Raw trait-derived gain structure for one identity. Never raises."""
    try:
        from cognition.fly_persona.state import get_personality

        tr = get_personality(user_id).traits
        c = tr["curiosity"]
        p = tr["playfulness"]
        a = tr["attachment"]
        m = tr["calmness"]
        e = tr["exploration"]

        cx_cur_gain = _mult(c, 0.6)
        return {
            "cx": {
                # per-drive: gain/decay multipliers; thresh multiplies the
                # base threshold (<1 = more sensitive).
                "curiosity": {
                    "gain": round(cx_cur_gain, 4),
                    "decay": round(_mult(c, 0.2) * _mult(e, 0.2), 4),
                    "thresh": round(max(_GAIN_LO, min(_GAIN_HI, 1.0 / cx_cur_gain)), 4),
                },
                "engagement": {
                    "gain": round(_mult(p, 0.6), 4),
                    "decay": 1.0,
                    "thresh": 1.0,
                },
                "rest": {"gain": 1.0, "decay": 1.0, "thresh": 1.0},
                "heading": {
                    "gain": 1.0,
                    "decay": round(_mult(e, 0.3), 4),
                },
                "urgency": {
                    # calmness lowers the gain (harder to startle) AND the
                    # retention factor (faster urgency recovery): both use a
                    # negative k so higher calmness shrinks the multiplier.
                    "gain": round(_mult(m, -0.6), 4),
                    "decay": round(_mult(m, -0.3), 4),
                },
            },
            "mb_plasticity": round(
                max(_GAIN_LO, min(_GAIN_HI, _mult(p, 0.5) * _mult(a, 0.5))), 4
            ),
            "gf_urgency": round(_mult(m, -0.6), 4),
            "dn_vigor": round(
                max(_GAIN_LO, min(_GAIN_HI, _mult(p, 0.4) * _mult(a, 0.4))), 4
            ),
        }
    except Exception as exc:
        log.debug("persona_gains skipped: %s", exc)
        return {}


def _identity_gains() -> dict:
    return {
        "cx": {
            "curiosity": {"gain": 1.0, "decay": 1.0, "thresh": 1.0},
            "engagement": {"gain": 1.0, "decay": 1.0, "thresh": 1.0},
            "rest": {"gain": 1.0, "decay": 1.0, "thresh": 1.0},
            "heading": {"gain": 1.0, "decay": 1.0},
            "urgency": {"gain": 1.0, "decay": 1.0},
        },
        "mb_plasticity": 1.0,
        "gf_urgency": 1.0,
        "dn_vigor": 1.0,
    }


def applied_gains(user_id: str | None = None) -> dict:
    """Gains the circuits should actually use this turn.

    live   → trait-derived gains.
    shadow → computed for the trail, but identity (no modulation).
    off    → identity, and callers should skip persona work entirely.
    Never raises.
    """
    try:
        from cognition.fly_persona import persona_mode

        if persona_mode() == "live":
            g = persona_gains(user_id)
            return g if g else _identity_gains()
        return _identity_gains()
    except Exception as exc:
        log.debug("applied_gains skipped: %s", exc)
        return _identity_gains()
