"""DN → motor command stage (Phase 10B).

Descending-neuron command bottleneck, modeled the way the rest of the fly
stack is modeled: as a *functional abstraction*, not a neuron simulation.

Takes the fly action-selection output (score_candidates record) plus the
current NeuralState readouts (vigor, drive, urgency, valence) and emits
discrete, named, bounded **motor primitives** — e.g. prosody, expression,
gaze, gesture, pose, vigor — rather than raw VRM numbers or TTS floats.

Design rules (shared with the rest of the fly brain):
  - deterministic given the same record + state (no wall clock, no RNG)
  - fail-soft: never raises; on any problem returns [] (no primitives)
  - conscience invariant: a vetoed or missing winner emits NOTHING. The body
    layer must never bypass a guardrail veto; a refused action produces no
    motor primitives.
  - persona/personality do NOT enter here (Phase 11 scope); parameters are
    explicit constants below.

The primitives are consumed by vnc_coordinator (sequencing/arbitration) and
then by the actuator backends in body.py. Nothing here touches hardware.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger("aiko.fly.motor_primitives")


# ── mode ────────────────────────────────────────────────────────────────

def _body_mode() -> str:
    try:
        from system.config import env_str
        return env_str("AIKO_FLY_BODY_MODE", "shadow").strip().lower()
    except Exception:
        return (os.getenv("AIKO_FLY_BODY_MODE") or "shadow").strip().lower()


def body_mode() -> str:
    """Current body-layer mode: off | shadow | live (default shadow)."""
    m = _body_mode()
    return m if m in ("off", "shadow", "live") else "shadow"


# ── primitive vocabulary ────────────────────────────────────────────────
# Actuators Aiko actually has today: TTS prosody, VRM avatar channels, and
# the agent's own step pacing. No physical body (see body.py's stub).

ACTUATORS = (
    "tts",            # TTS prosody hints
    "vrm.expression", # avatar face blend
    "vrm.gaze",       # avatar eye/head direction
    "vrm.gesture",    # avatar hand/arm motion
    "vrm.pose",       # avatar whole-body pose
    "agent",          # agent loop pacing (step budget, retry energy)
)

# name -> (actuator, priority, duration_turns, cooldown_turns)
# Priority: higher wins when two in-flight primitives target one actuator.
_PRIMITIVE_SPEC: dict[str, tuple[str, int, int, int]] = {
    "prosody":    ("tts",            30, 1, 0),
    "expression": ("vrm.expression", 20, 2, 1),
    "gaze":       ("vrm.gaze",       25, 1, 0),
    "gesture":    ("vrm.gesture",    15, 1, 2),
    "pose":       ("vrm.pose",       5,  3, 1),
    "vigor":      ("agent",          10, 2, 0),
}

# Candidate kinds coming out of action_select.Candidate.
_KIND_REPLY = "reply"
_KIND_TOOL = "tool"
_KIND_ROUTE = "route"


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = lo
    if v != v:  # NaN
        v = lo
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class MotorPrimitive:
    """One discrete motor command. All params are bounded and JSON-safe."""

    name: str
    params: dict = field(default_factory=dict)
    priority: int = 0
    duration_turns: int = 1
    cooldown_turns: int = 0
    actuator: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "actuator": self.actuator,
            "priority": self.priority,
            "duration_turns": self.duration_turns,
            "cooldown_turns": self.cooldown_turns,
            "params": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in self.params.items()},
        }


def _neural_snapshot(user_id: str | None):
    """Read the current NeuralState; None on any failure (fail-soft)."""
    try:
        from cognition.neural_state import peek_neural_state
        return peek_neural_state(user_id)
    except Exception:
        return None


def _expression_name(valence: float) -> str:
    if valence > 0.25:
        return "happy"
    if valence < -0.25:
        return "sad"
    return "neutral"


def emit_primitives(
    record: dict | None,
    *,
    user_id: str | None = None,
    state=None,
) -> list[MotorPrimitive]:
    """Emit motor primitives for one scored action-selection record.

    `record` is the dict returned by action_select.score_candidates
    (keys: winner, candidates[{kind, veto, ...}], mode, applied).
    `state` may be a NeuralState; when omitted it is peeked.

    Returns [] when: body mode is off, there is no winner, the winner was
    vetoed (conscience invariant), a GF interrupt is active, or anything
    fails. Never raises.
    """
    try:
        return _emit(record, user_id=user_id, state=state)
    except Exception as exc:
        log.debug("emit_primitives skipped: %s", exc)
        return []


def _emit(record: dict | None, *, user_id: str | None, state) -> list[MotorPrimitive]:
    if body_mode() == "off":
        return []
    if not isinstance(record, dict):
        return []
    winner = record.get("winner")
    if not winner:
        return []
    cands = record.get("candidates") or {}
    cand = cands.get(winner) or {}
    # Conscience invariant: vetoed / refused actions drive no body.
    if cand.get("veto"):
        log.debug("emit_primitives: winner %s vetoed; no primitives", winner)
        return []

    st = state if state is not None else _neural_snapshot(user_id)
    if st is None:
        return []
    # GF interrupt is authoritative: no new primitives while it is set.
    try:
        from cognition.fly_behavior.gf_global import should_cancel_output
        if should_cancel_output(user_id):
            return []
    except Exception:
        pass
    if bool(getattr(st, "interrupt", False)):
        return []

    vigor = _clamp(getattr(st, "motor_vigor", 1.0), 0.5, 1.5)
    drive = _clamp(getattr(st, "action_drive", 0.5), 0.0, 1.0)
    urgency = _clamp(getattr(st, "urgency", 0.0), 0.0, 1.0)
    valence = _clamp(getattr(st, "valence", 0.0), -1.0, 1.0)
    approach = _clamp(getattr(st, "approach", 0.0), 0.0, 1.0)
    avoidance = _clamp(getattr(st, "avoidance", 0.0), 0.0, 1.0)

    kind = str(cand.get("kind") or _KIND_REPLY)
    out: list[MotorPrimitive] = []

    def _mk(name: str, params: dict) -> MotorPrimitive:
        actuator, prio, dur, cd = _PRIMITIVE_SPEC[name]
        return MotorPrimitive(
            name=name,
            params={k: (round(_clamp(v, -2.0, 2.0), 4)
                        if isinstance(v, (int, float)) else v)
                    for k, v in params.items()},
            priority=prio,
            duration_turns=dur,
            cooldown_turns=cd,
            actuator=actuator,
        )

    # Urgency reshapes delivery: faster, tenser, less gestural.
    urg_boost = 1.0 + 0.25 * urgency

    if kind == _KIND_REPLY:
        out.append(_mk("prosody", {
            "rate": _clamp(0.92 * vigor * urg_boost, 0.85, 1.15),
            "volume": _clamp(0.90 + 0.20 * drive, 0.75, 1.15),
            "emphasis": _clamp(abs(valence) * 0.6 + 0.2 * urgency, 0.0, 1.0),
        }))
        out.append(_mk("expression", {
            "name": _expression_name(valence),
            "intensity": _clamp(0.45 + 0.35 * max(approach, avoidance)
                                + 0.15 * urgency, 0.0, 1.0),
        }))
        out.append(_mk("gaze", {
            "target": "user",
            "speed": _clamp(1.0 * urg_boost, 0.35, 1.4),
        }))
        out.append(_mk("pose", {
            "name": "engaged" if drive > 0.55 else "idle",
            "intensity": _clamp(drive, 0.0, 1.0),
        }))
    elif kind == _KIND_TOOL:
        out.append(_mk("gesture", {
            "name": "work",
            "intensity": _clamp(0.40 + 0.40 * (vigor - 0.5)
                                - 0.25 * urgency, 0.0, 1.0),
        }))
        out.append(_mk("gaze", {
            "target": "task",
            "speed": _clamp(0.9 * urg_boost, 0.35, 1.4),
        }))
        out.append(_mk("pose", {
            "name": "thinking",
            "intensity": _clamp(0.5 + 0.3 * drive, 0.0, 1.0),
        }))
        out.append(_mk("vigor", {
            "mult": _clamp(vigor, 0.4, 1.5),
        }))
    else:  # route / unknown kinds: minimal, observable body language
        out.append(_mk("gaze", {
            "target": "user",
            "speed": _clamp(1.0, 0.35, 1.4),
        }))
        out.append(_mk("pose", {
            "name": "idle",
            "intensity": _clamp(0.4 * drive + 0.2, 0.0, 1.0),
        }))

    return out


def primitive_names() -> list[str]:
    """The fixed primitive vocabulary (stable API for Studio/tests)."""
    return sorted(_PRIMITIVE_SPEC.keys())


def spec(name: str) -> dict:
    """Static spec for one primitive (actuator/priority/duration/cooldown)."""
    s = _PRIMITIVE_SPEC.get(name)
    if not s:
        return {}
    actuator, prio, dur, cd = s
    return {"actuator": actuator, "priority": prio,
            "duration_turns": dur, "cooldown_turns": cd}
