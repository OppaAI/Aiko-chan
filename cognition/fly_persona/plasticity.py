"""Phase 11 — trait plasticity through the Phase-10A dopamine machinery ONLY.

This module is the single sanctioned writer of trait values. No other
module may mutate NeuralPersonalityState (its traits property returns a
copy; _apply_delta is private to this module's use).

Update rule (documented Hebbian-style: trait × circuit-activity × dopamine):

    Δtrait = LR_P · da · signal(trait)

  LR_P            = 0.05  (AIKO_FLY_PERSONA_LR)
  per-event clamp |Δ| ≤ 0.02
  trait clamp     [default−0.30, default+0.30] ∩ [0, 1]
  cooldown        ≥ 3 credit events between trait updates per identity

Per-trait circuit-activity signals, read best-effort at credit time:
  curiosity   ← CX curiosity-drive activation
  playfulness ← CX engagement-drive activation
  exploration ← CX heading-vector magnitude (sustained directed attention)
  attachment  ← positive MB valence (approach), from NeuralState
  calmness    ← (0.5 − urgency trace): good outcomes without urgency raise
                calmness; urgent good outcomes lower it (the world rewards
                vigilance, not calm)

da is the Phase-10A dopamine signal (prediction-error-scaled), so traits
move only on *surprising* outcomes, and only in proportion to how active
their circuit channel was. Negative da reverses the direction.

Scope guard: only scope="real" reshapes personality. Simulated (FlyWorld)
and replay credit never touch traits — synthetic or consolidated
experience must not rewrite who she is becoming.

Fail-soft: never raises.
"""
from __future__ import annotations

import logging
import math
import os

log = logging.getLogger("aiko.fly.persona.plasticity")

_LR = 0.05
_MAX_DELTA = 0.02
_COOLDOWN_EVENTS = 3


def _lr() -> float:
    try:
        return max(0.0, min(0.5, float(os.getenv("AIKO_FLY_PERSONA_LR", str(_LR)))))
    except Exception:
        return _LR


def _signals(user_id: str | None) -> dict[str, float]:
    """Circuit-activity signals per trait, all in [0, 1] except calmness'."""
    sig = {
        "curiosity": 0.0,
        "playfulness": 0.0,
        "exploration": 0.0,
        "attachment": 0.0,
        "calmness": 0.0,
    }
    try:
        from cognition.centralcomplex.temporal import get_temporal_cx

        tcx = get_temporal_cx(user_id)
        try:
            sig["curiosity"] = float(tcx.activation("curiosity"))
        except Exception:
            pass
        try:
            sig["playfulness"] = float(tcx.activation("engagement"))
        except Exception:
            pass
        try:
            hx, hy = tcx.heading
            sig["exploration"] = max(
                0.0, min(1.0, math.sqrt(hx * hx + hy * hy) / math.sqrt(2.0))
            )
        except Exception:
            pass
        try:
            sig["calmness"] = 0.5 - float(tcx.urgency)
        except Exception:
            pass
    except Exception as exc:
        log.debug("persona plasticity cx signals skipped: %s", exc)
    try:
        from cognition.neural_state import peek_neural_state

        st = peek_neural_state(user_id)
        if st is not None:
            sig["attachment"] = max(0.0, min(1.0, float(st.valence or 0.0)))
    except Exception as exc:
        log.debug("persona plasticity valence signal skipped: %s", exc)
    return sig


def on_credit_outcome(
    user_id: str | None,
    *,
    reward: float = 0.0,
    pe: float = 0.0,
    da: float = 0.0,
    scope: str | None = "real",
) -> dict:
    """Apply one dopamine event to the trait vector. Never raises.

    Called exclusively from cognition.flymemory.credit.credit_event.
    """
    out = {"updated": False, "deltas": {}, "reason": ""}
    try:
        from cognition.fly_persona import persona_mode
        from cognition.fly_persona.state import get_personality
        from cognition.fly_persona import state as _state_mod

        if persona_mode() != "live":
            out["reason"] = "persona_not_live"
            return out
        if (scope or "real").strip() != "real":
            out["reason"] = "scope_guarded"
            return out
        try:
            da_f = max(-1.0, min(1.0, float(da)))
        except Exception:
            out["reason"] = "bad_da"
            return out
        if abs(da_f) < 1e-9:
            out["reason"] = "no_drive"
            return out

        st = get_personality(user_id)
        st._event_clock += 1
        if st._event_clock - st._last_update_event < _COOLDOWN_EVENTS:
            out["reason"] = "cooldown"
            return out

        sig = _signals(user_id)
        lr = _lr()
        deltas: dict[str, float] = {}
        # state._lock is the module-level RLock guarding all personality
        # states; plasticity is the single sanctioned writer.
        with _state_mod._lock:
            cur = dict(st._traits)
            for trait, s in sig.items():
                try:
                    s_f = float(s)
                except Exception:
                    continue
                if not math.isfinite(s_f):
                    continue
                d = max(-_MAX_DELTA, min(_MAX_DELTA, lr * da_f * s_f))
                if abs(d) < 1e-9:
                    continue
                cur[trait] = st._clamp(trait, cur.get(trait, 0.5) + d)
                deltas[trait] = round(d, 5)
            if deltas:
                st._traits = cur
                st._last_update_event = st._event_clock
                st.save()
                out["updated"] = True
                out["deltas"] = deltas
                out["reason"] = "ok"
            else:
                out["reason"] = "no_delta"
    except Exception as exc:
        log.debug("persona plasticity skipped: %s", exc)
        out["reason"] = f"error: {exc}"
    return out
