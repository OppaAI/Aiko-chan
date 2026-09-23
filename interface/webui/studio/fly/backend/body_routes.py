"""Stage 6.1 body route helpers for Fly Studio."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_body_payload(uid: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {}
    try:
        from cognition.fly_behavior.dn_body import body_drive

        body = body_drive(user_id=uid)
    except Exception:
        logger.exception("Body drive failed")
        body = {"mode": "off"}
    intents: list[dict[str, Any]] = []
    try:
        cancelled = bool(body.get("cancelled"))
        expr = float(body.get("expression_intensity") or 0.5)
        gest = float(body.get("gesture_intensity") or 0.4)
        gaze = float(body.get("gaze_speed") or 1.0)
        vigor = float(body.get("action_vigor") or 1.0)
        if cancelled:
            intents.append({
                "kind": "expression",
                "name": "neutral",
                "intensity": 0.15,
                "reason": "gf_cancel",
            })
        else:
            intents.append({
                "kind": "expression",
                "name": "happy" if expr >= 0.55 else ("sorrow" if expr <= 0.35 else "neutral"),
                "intensity": round(expr, 3),
                "reason": "dn_expression",
            })
            intents.append({
                "kind": "gesture",
                "name": "idle_amp",
                "intensity": round(gest, 3),
                "reason": "dn_gesture",
            })
            intents.append({
                "kind": "gaze",
                "name": "speed",
                "intensity": round(gaze, 3),
                "reason": "dn_gaze",
            })
            intents.append({
                "kind": "action",
                "name": "vigor",
                "intensity": round(vigor, 3),
                "reason": "dn_action",
            })
    except Exception:
        pass
    return {"body": body, "avatar_intents": intents}
