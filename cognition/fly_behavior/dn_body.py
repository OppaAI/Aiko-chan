"""DN → embodied Aiko (Stage 6).

Reads NeuralState motor_vigor / action_drive (and optional interrupt) and
returns a body-drive packet for TTS, VRM expression, gaze, gesture, and
agent action vigor. Never raises.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("aiko.fly.dn_body")


def _mode() -> str:
    try:
        from system.config import env_str
        return env_str("MEMORY_FLYDN_MODE", "off").strip().lower()
    except Exception:
        return (os.getenv("MEMORY_FLYDN_MODE") or "off").strip().lower()


def body_drive(*, user_id: str | None = None) -> dict:
    """Return drive dict for speech + virtual body."""
    mode = _mode()
    out = {
        "mode": mode,
        "applied": False,
        "cancelled": False,
        "rate_mult": 1.0,
        "vol_mult": 1.0,
        "expression_intensity": 0.5,
        "gesture_intensity": 0.4,
        "gaze_speed": 1.0,
        "action_vigor": 1.0,
        "arousal": 0.5,
    }
    if mode not in ("shadow", "live"):
        return out
    try:
        from cognition.fly_behavior.gf_global import should_cancel_output
        if should_cancel_output(user_id):
            out["cancelled"] = True
            out["rate_mult"] = 0.0
            out["vol_mult"] = 0.0
            out["gesture_intensity"] = 0.0
            out["expression_intensity"] = 0.15
            out["applied"] = mode == "live"
            return out
    except Exception:
        pass
    try:
        from cognition.neural_state import peek_neural_state
        st = peek_neural_state(user_id)
        if st is None:
            return out
        vigor = float(getattr(st, "motor_vigor", 1.0) or 1.0)
        drive = float(getattr(st, "action_drive", 0.5) or 0.5)
        v = (drive - 0.5) * 2.0
        out.update(
            {
                "arousal": round(drive, 4),
                "rate_mult": round(max(0.85, min(1.15, vigor)), 4),
                "vol_mult": round(max(0.75, min(1.15, 0.90 + 0.20 * drive)), 4),
                "expression_intensity": round(max(0.0, min(1.0, 0.45 + 0.35 * v)), 4),
                "gesture_intensity": round(max(0.0, min(1.0, 0.40 + 0.40 * v)), 4),
                "gaze_speed": round(max(0.35, min(1.4, 1.0 + 0.30 * v)), 4),
                "action_vigor": round(max(0.4, min(1.5, vigor)), 4),
                "applied": mode == "live",
            }
        )
        try:
            from cognition.neural_state import get_neural_state
            get_neural_state(user_id).record_influence(
                {
                    "kind": "dn_body",
                    "mode": mode,
                    "rate_mult": out["rate_mult"],
                    "expression": out["expression_intensity"],
                    "gesture": out["gesture_intensity"],
                    "gaze": out["gaze_speed"],
                    "action_vigor": out["action_vigor"],
                    "cancelled": out["cancelled"],
                }
            )
        except Exception:
            pass
    except Exception as exc:
        log.debug("body_drive skipped: %s", exc)
    # Phase 10B: fold the body layer's primitive-derived fields into the
    # packet (additive only — all existing keys keep their meaning).
    try:
        from cognition.fly_behavior import body as _body_layer
        bl = _body_layer.drive(user_id=user_id) or {}
        vrm = (bl.get("backends") or {}).get("vrm") or {}
        out.update({
            "body_mode": bl.get("mode", "off"),
            "body_applied": bool(bl.get("applied", False)),
            "primitives": bl.get("primitives", []),
            "expression_name": vrm.get("expression_name", "neutral"),
            "gaze_target": vrm.get("gaze_target", "user"),
            "gesture_name": vrm.get("gesture_name", "none"),
            "pose_name": vrm.get("pose_name", "idle"),
            "emphasis": ((bl.get("backends") or {}).get("tts") or {}).get(
                "emphasis", 0.0),
        })
    except Exception as exc:
        log.debug("body_drive 10B layer skipped: %s", exc)
    return out


def agent_step_budget(*, user_id: str | None = None, base: int = 8) -> int:
    """Scale agent max steps by action_vigor (low energy → fewer steps)."""
    try:
        bd = body_drive(user_id=user_id)
        if bd.get("cancelled"):
            return 0
        # Phase 10B: when the body layer is live, its agent backend is the
        # authoritative vigor source (primitive-driven pacing).
        if bd.get("body_applied"):
            try:
                from cognition.fly_behavior import body as _body_layer
                cmd = _body_layer.agent_command(user_id=user_id) or {}
                if cmd.get("applied"):
                    vig = float(cmd.get("vigor") or 1.0)
                    return max(1, int(round(base * max(0.4, min(1.3, vig)))))
            except Exception:
                pass
        vig = float(bd.get("action_vigor") or 1.0)
        return max(1, int(round(base * max(0.4, min(1.3, vig)))))
    except Exception:
        return base
