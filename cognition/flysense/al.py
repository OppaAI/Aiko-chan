"""Antennal-lobe divisive normalization for embeddings.

In the fly, ~50 glomeruli fan out to Kenyon cells while broad local neurons
(lLN/vLN) divide every channel by the pooled drive — gain control that keeps
odors discriminable across concentrations. Same trick ports to embeddings:
project into 49 real glomerular channels, divide by real inhibition ratios
(MaleCNS v1.0, median 0.39), project back, renormalize.

REAL: 49 glomeruli + per-glomerulus inhibition ratios from the connectome.
SYNTHETIC: the embedding<->glomerulus projections (fixed seeded matrices).
Runtime deps: numpy only. Deterministic per embedding dim.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).resolve().parent / "data" / "al_gains.json"


class FlyAL:
    def __init__(self, sigma: float = 0.5, pool_gain: float = 1.0, seed: int = 20260608,
                 data_path: str | Path = DATA_PATH) -> None:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        gloms = data["glomeruli"]
        self.glomeruli = sorted(gloms)
        self.inh = np.array([gloms[g]["inhibition_ratio"] for g in self.glomeruli], dtype=np.float64)
        self.sigma = float(sigma)
        self.pool_gain = float(pool_gain)
        self.seed = int(seed)
        self._proj: dict[int, np.ndarray] = {}

    def _matrix(self, dim: int) -> np.ndarray:
        m = self._proj.get(dim)
        if m is None:
            rng = np.random.default_rng((self.seed, dim))
            m = rng.normal(0, 1, (dim, len(self.glomeruli)))
            m /= max(np.abs(m).sum(), 1e-9)
            self._proj[dim] = m
        return m

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        """Gain-control one embedding, preserving dtype/shape conventions (L2 unit)."""
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if np.all(v == 0):
            return np.asarray(vec).copy()
        P = self._matrix(v.shape[0])
        drive = np.maximum(P.T @ v, 0.0)
        pool = float(drive.mean())
        out = drive / (self.sigma + self.inh * drive + self.pool_gain * pool)
        back = P @ out
        n = float(np.linalg.norm(back))
        if n <= 0:
            return np.asarray(vec).copy()
        return (back / n).astype(np.asarray(vec).dtype if hasattr(vec, "dtype") else np.float32)

    def normalize_batch(self, mat: np.ndarray) -> np.ndarray:
        m = np.asarray(mat, dtype=np.float64)
        if m.ndim == 1:
            return self.normalize(m)
        return np.stack([self.normalize(r) for r in m])

    def summary(self) -> dict:
        return {"glomeruli": len(self.glomeruli),
                "inhibition_median": round(float(np.median(self.inh)), 4)}
