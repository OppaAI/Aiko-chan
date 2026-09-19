"""Translate bounded fly readouts into safe, inspectable avatar intents.

This is deliberately an avatar controller, not a claim that a MaleVNC neuron
can drive a VRM bone. It only proposes reversible, browser-local body language.
Speech, messages, hardware, and other external effects remain approval-gated.
"""
from __future__ import annotations

from dataclasses import dataclass


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class AvatarIntent:
    """A non-verbal virtual-body proposal with an explanation."""

    kind: str
    name: str
    intensity: float
    reason: str

    def as_dict(self) -> dict[str, str | float]:
        return {"kind": self.kind, "name": self.name, "intensity": round(self.intensity, 3), "reason": self.reason}


class AvatarEmbodiment:
    """Create a small, observable virtual-body vocabulary from readouts."""

    def propose(self, *, valence: float, arousal: float, motion_salience: float, output_drive: float) -> list[AvatarIntent]:
        valence = max(-1.0, min(1.0, float(valence)))
        arousal, motion_salience, output_drive = _unit(arousal), _unit(motion_salience), _unit(output_drive)
        expression = "happy" if valence > 0.2 else "sad" if valence < -0.2 else "neutral"
        pose = "alert" if motion_salience > 0.65 or arousal > 0.7 else "thinking" if output_drive > 0.55 else "idle"
        return [
            AvatarIntent("expression", expression, max(abs(valence), arousal * 0.35), "bounded affect readout"),
            AvatarIntent("pose", pose, max(arousal, output_drive), "bounded salience/output readout"),
        ]
