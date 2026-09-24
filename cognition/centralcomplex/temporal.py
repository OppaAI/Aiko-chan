"""Phase 7 — CX persistent temporal / navigation dynamics (rate-based).

Cross-turn state the compass bump alone doesn't carry in closed form:
  * persistent heading vector (2-D, bounded) — where attention "points",
    carried turn to turn and nudged by motion / voice / urgency cues
  * competing drives — curiosity / engagement / rest, each with its own
    decay rate, arbitrated into Phase 5 action-selection votes
  * GF urgency as a decaying trace instead of per-turn spikes

Update equations — all explicit, bounded, deterministic given inputs
(no ad-hoc persistent variables, no randomness, no wall-clock in the math):

    heading_{t+1} = clip2(heading_t * D_H + cue_t * G_H)      # [-1,1]^2
    urgency_{t+1} = clip01(urgency_t * D_U + fresh_t * G_U)   # [0,1]
    drive_{t+1}   = clip01(drive_t * D_d + in_t * G_d)        # [0,1]
    valence_{t+1} = clip11(mb_valence_t)                     # [-1,1]

CX<->MB feedback (also explicit):
    thr_eff(d) = clip(thr_base_d - valence * K_V, 0.05, 0.95)  # MB->CX:
        positive valence lowers drive thresholds (approach),
        negative valence raises them (withdraw)
    act(d) = clip01((drive_d - thr_eff(d)) / (1 - thr_eff(d)))
    teach_gain = clip(1 + 0.4*act_cur - 0.5*act_rest, 0.5, 1.25)  # CX->MB:
        curiosity up-regulates teaching, rest down-regulates it

Modes (AIKO_FLY_CX_MODE): off | shadow | live. Default shadow:
compute + record influence, never steer. In live mode the drive biases
enter Phase 5 votes (additive, bounded, never overriding), the urgency
trace warms the GF line, and teaching strength is gated.

Per-turn cost is a few dozen float ops; no ML weights, no numpy.
Never raises — any failure degrades to neutral (zero bias) output.
"""
from __future__ import annotations

import logging
import math
import os
import threading

log = logging.getLogger("aiko.fly.cx_temporal")


def _env(name: str, default: str) -> str:
    try:
        from system.config import env_str

        return env_str(name, default)
    except Exception:
        return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return default


# ── config (env-overridable, read once at import like the other fly modules) ──

_MODE = _env("AIKO_FLY_CX_MODE", "shadow").strip().lower()

_D_H = _env_float("AIKO_FLY_CX_DECAY_HEADING", 0.92)   # heading persistence
_G_H = _env_float("AIKO_FLY_CX_GAIN_HEADING", 0.50)    # cue gain
_D_U = _env_float("AIKO_FLY_CX_DECAY_URGENCY", 0.88)   # urgency trace decay
_G_U = _env_float("AIKO_FLY_CX_GAIN_URGENCY", 0.60)    # fresh urgency gain
_K_V = _env_float("AIKO_FLY_CX_VALENCE_K", 0.20)        # MB valence → threshold
_DRIVE_W = _env_float("AIKO_FLY_CX_DRIVE_WEIGHT", 0.10)  # additive vote weight

# name → (decay, gain, base threshold)
_DRIVES: dict[str, tuple[float, float, float]] = {
    "curiosity": (
        _env_float("AIKO_FLY_CX_DECAY_CURIOSITY", 0.90),
        _env_float("AIKO_FLY_CX_GAIN_CURIOSITY", 0.35),
        _env_float("AIKO_FLY_CX_THRESH_CURIOSITY", 0.35),
    ),
    "engagement": (
        _env_float("AIKO_FLY_CX_DECAY_ENGAGEMENT", 0.85),
        _env_float("AIKO_FLY_CX_GAIN_ENGAGEMENT", 0.40),
        _env_float("AIKO_FLY_CX_THRESH_ENGAGEMENT", 0.30),
    ),
    "rest": (
        _env_float("AIKO_FLY_CX_DECAY_REST", 0.95),
        _env_float("AIKO_FLY_CX_GAIN_REST", 0.30),
        _env_float("AIKO_FLY_CX_THRESH_REST", 0.40),
    ),
}


def cx_mode() -> str:
    """Current CX temporal mode: off | shadow | live (default shadow)."""
    return _MODE if _MODE in ("off", "shadow", "live") else "shadow"


def drive_weight() -> float:
    return max(0.0, min(0.5, _DRIVE_W))


# ── math helpers ────────────────────────────────────────────────────────────

def _clip01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def _clip11(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(-1.0, min(1.0, v))


def _clip_vec2(x: float, y: float) -> tuple[float, float]:
    return _clip11(x), _clip11(y)


# ── state ───────────────────────────────────────────────────────────────────

class TemporalCX:
    """Per-identity persistent CX temporal state. Call step() once per turn.

    The math is pure: deterministic given the input dict, no I/O, no clock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.heading: tuple[float, float] = (0.0, 0.0)
        self.drives: dict[str, float] = {name: 0.0 for name in _DRIVES}
        self.urgency: float = 0.0
        self.valence: float = 0.0

    def step(self, inputs: dict | None) -> dict:
        """Advance one turn. inputs keys (all optional floats):

        cue_x, cue_y   2-D direction cue (compass heading × salience)
        fresh_urgency  this turn's GF urgency [0,1]
        novelty        LH novelty [0,1]            → curiosity
        engagement_in  user engagement [0,1]       → engagement
        rest_in        sleep pressure [0,1]        → rest
        mb_valence     mushroom-body valence [-1,1]
        """
        try:
            inp = inputs or {}
            hx, hy = self.heading
            cx = _clip11(inp.get("cue_x", 0.0))
            cy = _clip11(inp.get("cue_y", 0.0))
            self.heading = _clip_vec2(hx * _D_H + cx * _G_H,
                                      hy * _D_H + cy * _G_H)

            fresh = _clip01(inp.get("fresh_urgency", 0.0))
            self.urgency = _clip01(self.urgency * _D_U + fresh * _G_U)

            novelty = _clip01(inp.get("novelty", 0.0))
            eng_in = _clip01(inp.get("engagement_in", 0.0))
            rest_in = _clip01(inp.get("rest_in", 0.0))
            drive_inputs = {
                "curiosity": novelty,
                "engagement": eng_in,
                "rest": rest_in,
            }
            for name, (decay, gain, _thr) in _DRIVES.items():
                d = self.drives.get(name, 0.0)
                self.drives[name] = _clip01(d * decay + drive_inputs[name] * gain)

            self.valence = _clip11(inp.get("mb_valence", 0.0))
            return self.readout()
        except Exception as exc:
            log.debug("TemporalCX.step skipped: %s", exc)
            return self.readout()

    # ── derived readouts ────────────────────────────────────────────────

    def threshold(self, name: str) -> float:
        """Valence-modulated drive threshold (MB → CX)."""
        _decay, _gain, base = _DRIVES[name]
        return max(0.05, min(0.95, base - self.valence * _K_V))

    def activation(self, name: str) -> float:
        """Normalized suprathreshold drive activation in [0,1]."""
        thr = self.threshold(name)
        d = _clip01(self.drives.get(name, 0.0))
        return _clip01((d - thr) / max(1.0 - thr, 1e-6))

    def drive_bias(self, *, kind: str = "", energy: float = 0.0,
                   novelty: float = 0.5) -> float:
        """Additive Phase-5 vote bias in [-1,1] from drive activations.

        Engagement favors replying; curiosity favors novel candidates;
        rest damps high-energy candidates. Always computed — the caller
        gates application by mode.
        """
        try:
            act_c = self.activation("curiosity")
            act_e = self.activation("engagement")
            act_r = self.activation("rest")
            nov = _clip01(novelty)
            en = _clip11(energy)
            b = 0.0
            if str(kind) == "reply":
                b += 0.6 * act_e
            b += 0.6 * act_c * (2.0 * nov - 1.0)
            b -= 0.6 * act_r * max(0.0, en)
            return _clip11(b)
        except Exception as exc:
            log.debug("drive_bias skipped: %s", exc)
            return 0.0

    def teach_gain(self) -> float:
        """CX → MB teaching gate: bounded factor on teach strength."""
        try:
            g = 1.0 + 0.4 * self.activation("curiosity") \
                - 0.5 * self.activation("rest")
            return max(0.5, min(1.25, g))
        except Exception as exc:
            log.debug("teach_gain skipped: %s", exc)
            return 1.0

    def heading_deg(self) -> float:
        hx, hy = self.heading
        if hx * hx + hy * hy < 1e-12:
            return 0.0
        return float(math.degrees(math.atan2(hy, hx)) % 360.0)

    def reset_urgency(self) -> None:
        with self._lock:
            self.urgency = 0.0

    def readout(self) -> dict:
        with self._lock:
            return {
                "heading": [round(self.heading[0], 4), round(self.heading[1], 4)],
                "heading_deg": round(self.heading_deg(), 2),
                "drives": {k: round(_clip01(v), 4) for k, v in self.drives.items()},
                "activations": {k: round(self.activation(k), 4) for k in _DRIVES},
                "thresholds": {k: round(self.threshold(k), 4) for k in _DRIVES},
                "urgency": round(self.urgency, 4),
                "valence": round(self.valence, 4),
                "teach_gain": round(self.teach_gain(), 4),
            }


# ── per-identity registry ───────────────────────────────────────────────────

_states: dict[str, TemporalCX] = {}
_states_lock = threading.RLock()


def _key(user_id: str | None) -> str:
    try:
        from cognition.fly_registry import _norm_id

        return _norm_id(user_id)
    except Exception:
        return (user_id or "").strip() or "default"


def get_temporal_cx(user_id: str | None = None) -> TemporalCX:
    key = _key(user_id)
    with _states_lock:
        st = _states.get(key)
        if st is None:
            st = TemporalCX()
            _states[key] = st
        return st


def clear_temporal_cx(user_id: str | None = None) -> None:
    with _states_lock:
        _states.pop(_key(user_id), None)


# ── readouts for other modules (all best-effort) ────────────────────────────

def drive_bias_for_candidate(
    user_id: str | None,
    *,
    kind: str = "",
    energy: float = 0.0,
    novelty: float = 0.5,
) -> float:
    """Phase-5 vote bias for one candidate. Never raises."""
    try:
        return get_temporal_cx(user_id).drive_bias(
            kind=kind, energy=energy, novelty=novelty
        )
    except Exception as exc:
        log.debug("drive_bias_for_candidate skipped: %s", exc)
        return 0.0


def get_urgency_trace(user_id: str | None = None) -> float:
    """Current decaying GF-urgency trace. Never raises."""
    try:
        return _clip01(get_temporal_cx(user_id).urgency)
    except Exception:
        return 0.0


def reset_urgency(user_id: str | None = None) -> None:
    """Drop the urgency trace (e.g. when the interrupt is cleared)."""
    try:
        get_temporal_cx(user_id).reset_urgency()
    except Exception as exc:
        log.debug("reset_urgency skipped: %s", exc)


def teach_gain_for(user_id: str | None = None) -> float:
    """CX → MB teaching gate for this identity. Never raises."""
    try:
        return get_temporal_cx(user_id).teach_gain()
    except Exception:
        return 1.0


# ── per-turn tick (wiring; gathers inputs, steps state, records) ─────────────

def _compass_heading(user_id: str | None) -> float | None:
    """Best-effort current compass heading in degrees."""
    try:
        from cognition.fly_registry import get_flycx

        cx = get_flycx(user_id)
        if cx is None:
            return None
        ro = cx.readout() or {}
        return float(ro.get("heading_deg") or 0.0)
    except Exception as exc:
        log.debug("compass heading skipped: %s", exc)
        return None


def tick_cx_temporal(
    text: str,
    *,
    user_id: str | None = None,
    fresh_urgency: float = 0.0,
    mb_valence: float = 0.0,
    sleep_pressure: float = 0.0,
    novelty: float = 0.5,
    user_energy: float = 0.5,
    voice_energy: float = 0.0,
) -> dict:
    """Step the persistent CX state for one turn. Never raises.

    Gathers the direction cue (compass heading × salience) and motion
    salience best-effort, then runs the explicit update equations. In
    shadow mode (default) everything is computed and recorded; steering
    is applied by callers only in live mode.
    """
    out: dict = {"mode": cx_mode(), "ok": False}
    try:
        if cx_mode() == "off":
            return out

        motion = 0.0
        try:
            from cognition.neural_state import get_neural_state

            st = get_neural_state(user_id)
            motion = _clip01(float(getattr(st, "motion_salience", 0.0) or 0.0))
        except Exception:
            st = None

        # Direction cue: compass heading, scaled by the strongest salient
        # channel (motion / voice / fresh urgency). Silent turns contribute
        # nothing and the heading simply decays.
        cue_scale = _clip01(
            max(motion, _clip01(voice_energy), _clip01(fresh_urgency))
        )
        cue_x, cue_y = 0.0, 0.0
        if cue_scale > 0.0:
            hdg = _compass_heading(user_id)
            if hdg is not None:
                rad = math.radians(hdg % 360.0)
                cue_x = math.cos(rad) * cue_scale
                cue_y = math.sin(rad) * cue_scale

        tcx = get_temporal_cx(user_id)
        with tcx._lock:
            readout = tcx.step(
                {
                    "cue_x": cue_x,
                    "cue_y": cue_y,
                    "fresh_urgency": fresh_urgency,
                    "novelty": novelty,
                    "engagement_in": user_energy,
                    "rest_in": sleep_pressure,
                    "mb_valence": mb_valence,
                }
            )
        out.update({"ok": True, **readout})

        if st is not None:
            try:
                st.record_influence(
                    {
                        "kind": "cx_temporal",
                        "mode": cx_mode(),
                        "heading_deg": readout["heading_deg"],
                        "drives": readout["drives"],
                        "activations": readout["activations"],
                        "urgency": readout["urgency"],
                        "teach_gain": readout["teach_gain"],
                    }
                )
            except Exception:
                pass
        return out
    except Exception as exc:
        log.debug("tick_cx_temporal skipped: %s", exc)
        return out
