"""Descending-neuron output drive (Stage 6 — embodied).

DNs (1,342 neurons, 484 types in MaleCNS v1.0) are the final common path:
the brain decides, DNs gate whether and how vigorously the body acts.

Stage 6 maps (energy, decisiveness, affect) to:
  TTS rate/volume, expression intensity, gesture intensity, gaze speed,
  agent action vigor.

REAL: DN population scale (documented gain magnitudes).
SYNTHETIC: the mapping itself (functional DN-gain abstraction).
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "dn_stats.json"


class FlyDN:
    def __init__(
        self,
        rate_gain: float = 0.08,
        vol_gain: float = 0.10,
        expr_gain: float = 0.35,
        gesture_gain: float = 0.40,
        gaze_gain: float = 0.30,
        data_path: str | Path = DATA_PATH,
    ) -> None:
        with open(data_path, encoding="utf-8") as f:
            stats = json.load(f)
        self.n_dn = int(stats.get("n_dn", 1342))
        self.rate_gain = float(rate_gain)
        self.vol_gain = float(vol_gain)
        self.expr_gain = float(expr_gain)
        self.gesture_gain = float(gesture_gain)
        self.gaze_gain = float(gaze_gain)

    def drive(
        self,
        energy: float = 0.5,
        decisiveness: float = 0.5,
        affect: float = 0.0,
    ) -> dict:
        """Body + speech drive. Multipliers around 1.0; intensities in [0,1]."""
        e = max(0.0, min(1.0, float(energy)))
        d = max(0.0, min(1.0, float(decisiveness)))
        a = max(-1.0, min(1.0, float(affect)))
        arousal = 0.5 * e + 0.3 * d + 0.2 * (a * 0.5 + 0.5)
        v = (arousal - 0.5) * 2.0
        return {
            "rate_mult": 1.0 + self.rate_gain * v,
            "vol_mult": 1.0 + self.vol_gain * v,
            "arousal": round(arousal, 4),
            "expression_intensity": round(max(0.0, min(1.0, 0.45 + self.expr_gain * v)), 4),
            "gesture_intensity": round(max(0.0, min(1.0, 0.40 + self.gesture_gain * v)), 4),
            "gaze_speed": round(max(0.35, min(1.4, 1.0 + self.gaze_gain * v)), 4),
            "action_vigor": round(max(0.4, min(1.5, 1.0 + 0.45 * v)), 4),
        }

    def summary(self) -> dict:
        return {"n_dn": self.n_dn}
