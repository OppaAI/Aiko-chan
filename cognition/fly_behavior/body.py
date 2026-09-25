"""Actuator/body abstraction (Phase 10B).

The mandatory translation layer between motor primitives and Aiko's actual
outputs. MaleVNC-derived signals do NOT map 1:1 to VRM movement — so nothing
here maps them 1:1 either. Primitives are translated into per-actuator
commands with clamping and per-actuator rate limits, then handed to backends.

Concrete backends (Aiko has no physical body):
  - vrm    → extends the dn_body.py packet (avatar expression/gaze/gesture/pose)
  - tts    → TTS prosody hints (rate/volume/emphasis)
  - agent  → agent action-vigor (step pacing)
  - null   → log-only, for headless operation
  - physical → STUB ONLY. Aiko has no physical body. This is a documented
    interface showing where a future hardware backend would plug in; it is
    never selected and applies nothing.

Mode (AIKO_FLY_BODY_MODE=off|shadow|live, default shadow):
  - off:    the whole layer is a no-op
  - shadow: primitives + commands are computed and logged (NeuralState
    influence), nothing is applied
  - live:   backends apply; the VRM backend feeds dn_body's packet

Entry points:
  - on_scored(record, user_id=...) — hook called at the end of
    action_select.score_candidates; keeps the latest record per user
  - drive(user_id=...) — run the full pipeline for one turn
  - tts_hints(user_id=...) / agent_command(user_id=...) — backend outputs
    for existing consumers (dn_tts, agent_step_budget)

Never raises. Per-turn cost is bounded (few primitives, capped queues).
"""
from __future__ import annotations

import logging
import os
import threading
from collections import deque

log = logging.getLogger("aiko.fly.body")


def _env_mode() -> str:
    try:
        from system.config import env_str
        return env_str("AIKO_FLY_BODY_MODE", "shadow").strip().lower()
    except Exception:
        return (os.getenv("AIKO_FLY_BODY_MODE") or "shadow").strip().lower()


def body_mode() -> str:
    m = _env_mode()
    return m if m in ("off", "shadow", "live") else "shadow"


# ── translation: primitive params → per-actuator commands ─────────────
# Per-actuator max change per turn (rate limits keep motion smooth and
# prevent any single primitive from whipping the avatar around).
_RATE_LIMITS = {
    "vrm.expression": 0.25,
    "vrm.gaze": 0.30,
    "vrm.gesture": 0.30,
    "vrm.pose": 0.20,
    "tts": 0.10,
    "agent": 0.20,
}

# Which scalar of each command the rate limit applies to.
_RATE_KEYS = {
    "vrm.expression": "intensity",
    "vrm.gaze": "speed",
    "vrm.gesture": "intensity",
    "vrm.pose": "intensity",
    "tts": "rate",
    "agent": "vigor",
}

_VALID_GAZE_TARGETS = ("user", "task", "away")
_VALID_EXPRESSIONS = ("happy", "sad", "neutral")
_VALID_GESTURES = ("work", "greet", "emphasize", "none")
_VALID_POSES = ("engaged", "thinking", "idle")


def _clamp01(x) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v != v:
        v = 0.0
    return max(0.0, min(1.0, v))


def _pick(value, valid: tuple, default: str) -> str:
    v = str(value or "")
    return v if v in valid else default


_lock = threading.RLock()
_last_cmd: dict[str, dict[str, dict]] = {}   # user -> actuator -> command
_latest_record: dict[str, deque] = {}        # user -> deque[record]
_last_drive: dict[str, dict] = {}            # user -> last drive() result


def _key(user_id: str | None) -> str:
    try:
        from cognition.fly_registry import _norm_id
        return _norm_id(user_id)
    except Exception:
        return (user_id or "").strip() or "default"


def clear(user_id: str | None = None) -> None:
    """Drop body-layer state for a user (tests / reset). Never raises."""
    try:
        with _lock:
            _last_cmd.pop(_key(user_id), None)
            _latest_record.pop(_key(user_id), None)
            _last_drive.pop(_key(user_id), None)
    except Exception:
        pass


def on_scored(record: dict | None, *, user_id: str | None = None) -> None:
    """Hook: remember the latest action-selection record per user.

    Called at the end of action_select.score_candidates (fail-soft).
    Bounded: keeps the last 4 records only.
    """
    try:
        if not isinstance(record, dict):
            return
        with _lock:
            dq = _latest_record.get(_key(user_id))
            if dq is None:
                dq = deque(maxlen=4)
                _latest_record[_key(user_id)] = dq
            dq.append(record)
    except Exception as exc:
        log.debug("body on_scored skipped: %s", exc)


def latest_record(user_id: str | None = None) -> dict | None:
    try:
        with _lock:
            dq = _latest_record.get(_key(user_id))
            return dq[-1] if dq else None
    except Exception:
        return None


def translate(active: list[dict], *, user_id: str | None = None) -> dict:
    """Turn coordinated in-flight primitives into per-actuator commands.

    `active`: list of {"name","actuator","params"} from the VNC coordinator.
    Returns {actuator: command}. Applies clamping + per-actuator rate
    limits against the last-applied command. Deterministic. Never raises.
    """
    try:
        return _translate(active, user_id=user_id)
    except Exception as exc:
        log.debug("body translate skipped: %s", exc)
        return {}


def _translate(active: list[dict], *, user_id: str | None) -> dict:
    cmds: dict[str, dict] = {}
    for p in active or []:
        name = p.get("name", "")
        params = p.get("params") or {}
        if name == "prosody":
            cmds["tts"] = {
                "rate": max(0.85, min(1.15, float(params.get("rate", 1.0)))),
                "volume": max(0.75, min(1.15, float(params.get("volume", 1.0)))),
                "emphasis": _clamp01(params.get("emphasis", 0.0)),
            }
        elif name == "expression":
            cmds["vrm.expression"] = {
                "name": _pick(params.get("name"), _VALID_EXPRESSIONS, "neutral"),
                "intensity": _clamp01(params.get("intensity", 0.5)),
            }
        elif name == "gaze":
            cmds["vrm.gaze"] = {
                "target": _pick(params.get("target"), _VALID_GAZE_TARGETS, "user"),
                "speed": max(0.35, min(1.4, float(params.get("speed", 1.0)))),
            }
        elif name == "gesture":
            cmds["vrm.gesture"] = {
                "name": _pick(params.get("name"), _VALID_GESTURES, "none"),
                "intensity": _clamp01(params.get("intensity", 0.0)),
            }
        elif name == "pose":
            cmds["vrm.pose"] = {
                "name": _pick(params.get("name"), _VALID_POSES, "idle"),
                "intensity": _clamp01(params.get("intensity", 0.5)),
            }
        elif name == "vigor":
            cmds["agent"] = {
                "vigor": max(0.4, min(1.5, float(params.get("mult", 1.0)))),
            }

    # Rate limits: cap the per-turn change of each actuator's scalar.
    with _lock:
        prev = _last_cmd.setdefault(_key(user_id), {})
        for actuator, cmd in cmds.items():
            key = _RATE_KEYS.get(actuator)
            limit = _RATE_LIMITS.get(actuator)
            old = prev.get(actuator)
            if key and limit and isinstance(old, dict) and key in old:
                try:
                    delta = float(cmd[key]) - float(old[key])
                    if abs(delta) > limit:
                        cmd[key] = round(float(old[key]) + limit
                                         * (1.0 if delta > 0 else -1.0), 4)
                except Exception:
                    pass
            prev[actuator] = dict(cmd)
    return cmds


# ── backends ──────────────────────────────────────────────────────────

class _Backend:
    name = "base"

    def apply(self, commands: dict, *, user_id: str | None = None) -> dict:
        raise NotImplementedError


class VRMBackend(_Backend):
    """Avatar backend: folds actuator commands into the dn_body packet shape.

    Extends (does not replace) dn_body.body_drive: the packet keys Aiko's
    WebUI already consumes (expression_intensity, gesture_intensity,
    gaze_speed, rate_mult, ...) keep their meaning; primitive-derived
    fields (expression_name, gaze_target, gesture_name, pose_name) are
    additive.
    """

    name = "vrm"

    def apply(self, commands: dict, *, user_id: str | None = None) -> dict:
        out = {
            "backend": "vrm",
            "applied": False,
            "expression_name": "neutral",
            "expression_intensity": 0.5,
            "gaze_target": "user",
            "gaze_speed": 1.0,
            "gesture_name": "none",
            "gesture_intensity": 0.0,
            "pose_name": "idle",
            "pose_intensity": 0.5,
            "rate_mult": 1.0,
            "vol_mult": 1.0,
        }
        try:
            expr = commands.get("vrm.expression") or {}
            gaze = commands.get("vrm.gaze") or {}
            gest = commands.get("vrm.gesture") or {}
            pose = commands.get("vrm.pose") or {}
            tts = commands.get("tts") or {}
            out.update({
                "expression_name": expr.get("name", "neutral"),
                "expression_intensity": round(_clamp01(expr.get("intensity", 0.5)), 4),
                "gaze_target": gaze.get("target", "user"),
                "gaze_speed": round(max(0.35, min(1.4, float(gaze.get("speed", 1.0)))), 4),
                "gesture_name": gest.get("name", "none"),
                "gesture_intensity": round(_clamp01(gest.get("intensity", 0.0)), 4),
                "pose_name": pose.get("name", "idle"),
                "pose_intensity": round(_clamp01(pose.get("intensity", 0.5)), 4),
                "rate_mult": round(max(0.85, min(1.15, float(tts.get("rate", 1.0)))), 4),
                "vol_mult": round(max(0.75, min(1.15, float(tts.get("volume", 1.0)))), 4),
            })
        except Exception as exc:
            log.debug("vrm backend skipped: %s", exc)
        return out


class TTSBackend(_Backend):
    """TTS prosody hints backend: rate/volume/emphasis for the speaker."""

    name = "tts"

    def apply(self, commands: dict, *, user_id: str | None = None) -> dict:
        tts = commands.get("tts") or {}
        try:
            return {
                "backend": "tts",
                "applied": False,
                "rate": round(max(0.85, min(1.15, float(tts.get("rate", 1.0)))), 3),
                "volume": round(max(0.75, min(1.15, float(tts.get("volume", 1.0)))), 3),
                "emphasis": round(_clamp01(tts.get("emphasis", 0.0)), 3),
            }
        except Exception as exc:
            log.debug("tts backend skipped: %s", exc)
            return {"backend": "tts", "applied": False,
                    "rate": 1.0, "volume": 1.0, "emphasis": 0.0}


class AgentBackend(_Backend):
    """Agent pacing backend: action-vigor for step budgets / retry energy."""

    name = "agent"

    def apply(self, commands: dict, *, user_id: str | None = None) -> dict:
        ag = commands.get("agent") or {}
        try:
            vigor = max(0.4, min(1.5, float(ag.get("vigor", 1.0))))
        except Exception:
            vigor = 1.0
        return {"backend": "agent", "applied": False,
                "vigor": round(vigor, 4)}


class NullBackend(_Backend):
    """Log-only backend for headless operation: computes, applies nothing."""

    name = "null"

    def apply(self, commands: dict, *, user_id: str | None = None) -> dict:
        log.debug("body null backend: %d actuator commands logged",
                  len(commands or {}))
        return {"backend": "null", "applied": False,
                "logged": True, "n_commands": len(commands or {})}


class PhysicalBodyBackend(_Backend):
    """STUB — Aiko has no physical body.

    This class exists only to document where a future hardware backend
    would plug in. It is never selected by `drive()` and applies nothing.
    A real implementation would subclass _Backend, talk to the hardware
    driver, and register itself in _BACKENDS — plus go through its own
    safety review before ever being enabled.
    """

    name = "physical"

    def apply(self, commands: dict, *, user_id: str | None = None) -> dict:
        return {"backend": "physical", "applied": False, "supported": False,
                "reason": "no physical body configured"}


_BACKENDS: dict[str, _Backend] = {
    "vrm": VRMBackend(),
    "tts": TTSBackend(),
    "agent": AgentBackend(),
    "null": NullBackend(),
    "physical": PhysicalBodyBackend(),  # stub; never selected
}

# Backends the live path actually drives. "physical" is deliberately absent.
_LIVE_BACKENDS = ("vrm", "tts", "agent")


def backends() -> list[str]:
    """Registered backend names (stable API for Studio/tests)."""
    return sorted(_BACKENDS.keys())


# ── full pipeline ─────────────────────────────────────────────────────

def drive(
    record: dict | None = None,
    *,
    user_id: str | None = None,
    tick: int | None = None,
) -> dict:
    """Run DN → primitives → VNC coordination → actuators → backends.

    `record`: score_candidates record; when omitted the latest record from
    on_scored() is used. Shadow computes everything and logs; live applies
    the vrm/tts/agent backends. Never raises.
    """
    try:
        return _drive(record, user_id=user_id, tick=tick)
    except Exception as exc:
        log.debug("body drive skipped: %s", exc)
        return {"mode": body_mode(), "applied": False, "primitives": [],
                "actuators": {}, "backends": {}}


def _drive(record: dict | None, *, user_id: str | None, tick: int | None) -> dict:
    mode = body_mode()
    out: dict = {
        "mode": mode,
        "applied": False,
        "primitives": [],
        "coordinator": {},
        "actuators": {},
        "backends": {},
        "cancelled": False,
    }
    if mode == "off":
        return out

    rec = record if isinstance(record, dict) else latest_record(user_id)

    from cognition.fly_behavior import motor_primitives as mp
    from cognition.fly_behavior import vnc_coordinator as vnc
    from cognition.fly_behavior.gf_global import should_cancel_output

    interrupt = bool(should_cancel_output(user_id))
    prims = [] if interrupt else mp.emit_primitives(rec, user_id=user_id)
    out["primitives"] = [p.as_dict() for p in prims]

    coord = vnc.coordinate(prims, user_id=user_id, tick=tick,
                           interrupt=interrupt)
    out["coordinator"] = {
        "tick": coord.get("tick"),
        "started": coord.get("started", []),
        "cancelled": coord.get("cancelled", []),
        "dropped": coord.get("dropped", []),
    }
    out["cancelled"] = interrupt or bool(coord.get("cancelled"))

    actuators = translate(coord.get("active", []), user_id=user_id)
    out["actuators"] = actuators

    live = mode == "live"
    for name in _LIVE_BACKENDS:
        res = _BACKENDS[name].apply(actuators, user_id=user_id)
        res["applied"] = live
        out["backends"][name] = res
    # The null backend always runs in shadow as the headless record.
    null_res = _BACKENDS["null"].apply(actuators, user_id=user_id)
    out["backends"]["null"] = null_res
    out["applied"] = live

    # Observability: what the body layer did this turn.
    try:
        from cognition.neural_state import get_neural_state
        get_neural_state(user_id).record_influence({
            "kind": "body_drive",
            "mode": mode,
            "n_primitives": len(prims),
            "n_active": len(coord.get("active", [])),
            "cancelled": out["cancelled"],
            "actuators": sorted(actuators.keys()),
            "applied": live,
        })
    except Exception:
        pass
    with _lock:
        _last_drive[_key(user_id)] = out
    return out


def _cached_drive(user_id: str | None) -> dict:
    """Last drive() result for a user, or run the pipeline once."""
    try:
        with _lock:
            cached = _last_drive.get(_key(user_id))
        if isinstance(cached, dict) and cached:
            return cached
    except Exception:
        pass
    return drive(user_id=user_id)


def tts_hints(user_id: str | None = None) -> dict:
    """TTS prosody hints from the latest drive (for dn_tts consumers)."""
    try:
        res = _cached_drive(user_id)["backends"].get("tts", {})
        return {"rate": res.get("rate", 1.0), "volume": res.get("volume", 1.0),
                "emphasis": res.get("emphasis", 0.0),
                "applied": bool(res.get("applied", False))}
    except Exception:
        return {"rate": 1.0, "volume": 1.0, "emphasis": 0.0, "applied": False}


def agent_command(user_id: str | None = None) -> dict:
    """Agent pacing command from the latest drive (for agent_step_budget)."""
    try:
        res = _cached_drive(user_id)["backends"].get("agent", {})
        return {"vigor": res.get("vigor", 1.0),
                "applied": bool(res.get("applied", False))}
    except Exception:
        return {"vigor": 1.0, "applied": False}
