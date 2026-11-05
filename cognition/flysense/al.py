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
import threading
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
        if not np.isfinite(self.sigma) or self.sigma <= 0 or not np.isfinite(self.pool_gain) or self.pool_gain < 0:
            raise ValueError("AL gains need positive finite sigma and nonnegative finite pool gain")
        if not self.glomeruli or not np.isfinite(self.inh).all() or (self.inh < 0).any():
            raise ValueError("AL inhibition ratios must be finite and nonnegative")
        self._lock = threading.RLock()
        self._proj: dict[int, np.ndarray] = {}

    def _matrix(self, dim: int) -> np.ndarray:
        with self._lock:
            m = self._proj.pop(dim, None)
            if m is None:
                rng = np.random.default_rng((self.seed, dim))
                m = rng.normal(0, 1, (dim, len(self.glomeruli)))
                m /= max(np.abs(m).sum(), 1e-9)
            self._proj[dim] = m
            while len(self._proj) > 4:
                self._proj.pop(next(iter(self._proj)))
            return m

    @staticmethod
    def _dtype(array):
        return array.dtype if array.dtype.kind == "f" else np.dtype(np.float32)

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        """Gain-control one finite vector; return floating, L2-unit output."""
        original = np.asarray(vec)
        if original.ndim != 1 or original.size > 16384 or original.dtype.kind not in "biuf":
            raise ValueError("AL input must be a real vector of at most 16384 values")
        v = original.astype(np.float64)
        if not np.isfinite(v).all():
            raise ValueError("AL input must be finite")
        dtype = self._dtype(original)
        if not v.size or np.all(v == 0):
            return original.astype(dtype, copy=True)
        peak = float(np.abs(v).max())
        if peak > 1e100:
            v /= peak
        P = self._matrix(v.size)
        drive = np.maximum(P.T @ v, 0.0)
        pool = float(drive.mean())
        out = drive / (self.sigma + self.inh * drive + self.pool_gain * pool)
        back = P @ out
        n = float(np.linalg.norm(back))
        if not np.isfinite(n) or n <= 0:
            baseline = v / np.abs(v).max()
            return (baseline / np.linalg.norm(baseline)).astype(dtype)
        return (back / n).astype(dtype)

    def normalize_batch(self, mat: np.ndarray) -> np.ndarray:
        m = np.asarray(mat)
        if m.ndim == 1:
            return self.normalize(m)
        if m.ndim != 2 or m.shape[1] > 16384 or m.dtype.kind not in "biuf":
            raise ValueError("AL batch must contain real embedding vectors")
        if not m.shape[0]:
            return np.empty(m.shape, dtype=self._dtype(m))
        return np.stack([self.normalize(row) for row in m])

    def summary(self) -> dict:
        return {"glomeruli": len(self.glomeruli),
                "inhibition_median": round(float(np.median(self.inh)), 4)}
