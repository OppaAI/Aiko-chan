"""Typed, consent-aware observations and a bounded VRM output adapter."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensoryObservation:
    modality: str
    salience: float
    consented: bool = True
    metadata: dict | None = None

    def drive(self, seed: str) -> dict[str, float]:
        return {seed: max(0.0, min(1.0, self.salience))} if self.consented else {}


class AvatarMotorController:
    """Maps vetted intents—not neural IDs—to the existing WebUI VRM protocol."""

    ALLOWED = {"expression", "pose", "viseme"}

    def dispatch(self, intent: dict) -> bool:
        kind = str(intent.get("kind", ""))
        if kind not in self.ALLOWED:
            return False
        try:
            from interface.webui.webui import webui_bridge
            bridge = webui_bridge()
            if bridge is None:
                return False
            if kind == "expression":
                bridge.set_expression(str(intent.get("name", "neutral")), float(intent.get("intensity", 0.5)))
            elif kind == "viseme":
                bridge.set_viseme(str(intent.get("name", "A")), float(intent.get("weight", 0.0)))
            else:
                bridge.set_pose(str(intent.get("name", "thinking")), bool(intent.get("active", True)))
            return True
        except Exception:
            return False
