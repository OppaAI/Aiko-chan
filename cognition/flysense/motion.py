"""T4/T5 motion energy (Reichardt detectors + LPTC pooling).

In the fly, T4 (ON) and T5 (OFF) subtypes a/b/c/d tile four cardinal
directions; HS cells pool horizontal, VS vertical (MaleCNS v1.0 confirms:
HS<-T4a/T5a, VS<-T4d/T5d, 258k synapses). This module runs that computation
on downscaled frames: opponent correlation per direction, pooled with the
real subtype convergence weights.

REAL: subtype convergence counts + LPTC pooling weights from the connectome.
SYNTHETIC: the Reichardt delay-and-compare dynamics (functional), frame
  downscaling, thresholding. The JS mirror in companion.js implements the
  same math for the in-browser reflex arc.
Runtime deps: numpy only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).resolve().parent / "data" / "motion_stats.json"

# Canonical direction semantics from lobula-plate layer anatomy.
SUBTYPE_DIR = {"a": "h", "b": "h", "c": "v", "d": "v"}


def _subtype_gains(lptc_gain: dict) -> dict:
    """Collapse per-LPTC pooling into horizontal/vertical channel weights."""
    gains = {"h": 0.0, "v": 0.0}
    for post, subs in lptc_gain.items():
        total = sum(subs.values()) or 1
        for sub, w in subs.items():
            gains["h" if str(post).startswith("HS") else "v"] += w / total
    n_hs = sum(1 for p in lptc_gain if str(p).startswith("HS")) or 1
    n_vs = sum(1 for p in lptc_gain if str(p).startswith("VS")) or 1
    return {"h": gains["h"] / n_hs, "v": gains["v"] / n_vs}


def emd_energy(prev: np.ndarray, curr: np.ndarray, gains: dict | None = None) -> dict:
    """Opponent Reichardt energy between two grayscale frames (any size).

    Returns {h, v, total}: horizontal/vertical/total motion energy in
    arbitrary units ~[0, 1+] (brighter change = more). Pure function.
    """
    p = np.asarray(prev, dtype=np.float64)
    c = np.asarray(curr, dtype=np.float64)
    if p.shape != c.shape:
        n = (min(p.shape[0], c.shape[0]), min(p.shape[1], c.shape[1]))
        p, c = p[:n[0], :n[1]], c[:n[0], :n[1]]
    on_p, off_p = np.maximum(p, 0.0), np.maximum(-p, 0.0)
    on_c, off_c = np.maximum(c, 0.0), np.maximum(-c, 0.0)

    def opponent(a_c, a_p, axis: int) -> float:
        # right/down vs left/up correlation imbalance (mean-removed frames).
        fwd = np.roll(a_p, 1, axis=axis)
        bwd = np.roll(a_p, -1, axis=axis)
        return float(abs((a_c * fwd).mean() - (a_c * bwd).mean()))

    h = opponent(on_c, on_p, 1) + opponent(off_c, off_p, 1)
    v = opponent(on_c, on_p, 0) + opponent(off_c, off_p, 0)
    g = gains or {"h": 1.0, "v": 1.0}
    h, v = h * g["h"], v * g["v"]
    # Wide-field change energy (decorrelated large moves / appearance):
    # LPTC-pooling analogue for displacements beyond Reichardt range.
    change = float(abs(c - p).mean())
    return {"h": h, "v": v, "total": h + v, "change": change}


class FlyMotion:
    """Stateful gate: holds the last thumbnail, scores change per frame."""

    def __init__(self, size: int = 48, threshold: float = 0.02,
                 change_threshold: float = 0.10,
                 data_path: str | Path = DATA_PATH) -> None:
        with open(data_path, encoding="utf-8") as f:
            stats = json.load(f)
        self.gains = _subtype_gains(stats.get("lptc_gain", {}))
        self.size = int(size)
        self.threshold = float(threshold)
        self.change_threshold = float(change_threshold)
        self._last: np.ndarray | None = None

    def downscale(self, frame: np.ndarray) -> np.ndarray:
        """Mean-pool any grayscale frame to (size, size), zero-mean/unit-peak."""
        f = np.asarray(frame, dtype=np.float64)
        if f.ndim == 3:
            f = f[..., :3].mean(axis=2)
        h, w = f.shape
        sh, sw = max(1, h // self.size), max(1, w // self.size)
        small = f[:sh * self.size, :sw * self.size].reshape(self.size, sh, self.size, sw).mean(axis=(1, 3))
        small = small - small.mean()
        peak = float(abs(small).max())
        return small / peak if peak > 0 else small

    def score(self, frame: np.ndarray) -> dict:
        """Score change vs the previous frame; first frame arms the detector."""
        small = self.downscale(frame)
        if self._last is None:
            self._last = small
            return {"h": 0.0, "v": 0.0, "total": 0.0, "novel": True}
        out = emd_energy(self._last, small, self.gains)
        out["novel"] = False
        self._last = small
        return out

    def moved(self, frame: np.ndarray) -> tuple[bool, dict]:
        """Gate helper: (changed_enough, energies)."""
        e = self.score(frame)
        return bool(e["total"] >= self.threshold or e.get("change", 0.0) >= self.change_threshold or e["novel"]), e

    def reset(self) -> None:
        self._last = None
