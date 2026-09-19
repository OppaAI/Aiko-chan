"""Descending-neuron output drive (functional, stats-backed).

DNs (1,342 neurons, 484 types in MaleCNS v1.0) are the final common path:
the brain decides, DNs gate whether and how vigorously the body acts. Aiko
analogue: arousal-gated prosody/energy of speech output. This module maps
(energy, decisiveness, affect) to small rate/volume multipliers; the
caller's existing clamps stay authoritative.

REAL: DN population scale (drives the documented gain magnitudes).
SYNTHETIC: the mapping itself (functional DN-gain abstraction).
Runtime deps: none (stdlib only).
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "dn_stats.json"


class FlyDN:
    def __init__(self, rate_gain: float = 0.08, vol_gain: float = 0.10,
                 data_path: str | Path = DATA_PATH) -> None:
        with open(data_path, encoding="utf-8") as f:
            stats = json.load(f)
        self.n_dn = int(stats.get("n_dn", 1342))
        self.rate_gain = float(rate_gain)
        self.vol_gain = float(vol_gain)

    def drive(self, energy: float = 0.5, decisiveness: float = 0.5,
              affect: float = 0.0) -> dict:
        """Small multipliers around 1.0. All inputs clipped to valid ranges."""
        e = max(0.0, min(1.0, float(energy)))
        d = max(0.0, min(1.0, float(decisiveness)))
        a = max(-1.0, min(1.0, float(affect)))
        arousal = 0.5 * e + 0.3 * d + 0.2 * (a * 0.5 + 0.5)
        return {
            "rate_mult": 1.0 + self.rate_gain * (arousal - 0.5) * 2.0,
            "vol_mult": 1.0 + self.vol_gain * (arousal - 0.5) * 2.0,
            "arousal": round(arousal, 4),
        }

    def summary(self) -> dict:
        return {"n_dn": self.n_dn}
