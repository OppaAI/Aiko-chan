"""Fruit-fly central-complex (CX) compass + sleep pressure for Aiko.

Biology (adult Drosophila, MaleCNS v1.0, Berg et al. 2025):
  EPG neurons tile the ellipsoid body in 16 wedges and hold a bump of
  activity = head direction (ring attractor). PENs rotate the bump with
  angular velocity. ER/ExR ring neurons gate visual/feature context in.
  PFL output neurons steer action from the bump. EB R5 (ER5) neurons track
  sleep drive (Liu et al. 2016). Aiko mapping: bump heading/sharpness =
  focus direction/confidence, PFL imbalance = decisiveness, ER5 accumulator
  = drowsiness pressure on arousal.

What is real vs synthetic (see README.md):
  REAL: every neuron/edge/weight (RING/PEN/PEG/EPG/PF/INPUT), EPG wedge
    order parsed from real instance suffixes (_R1..8/_L1..8), soma sides
    for the PFL L/R steering split.
  SYNTHETIC: feature->INPUT/RING sensory projections (fixed seeded matrix),
    PEN angular drive + fatigue scalars (caller-provided), Mexican-hat-free
    single-step recurrent update with real EPG<->EPG weights, normalisation.

Runtime deps: numpy only. Deterministic given `seed`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).resolve().parent / "data" / "cx_circuit.npz"
N_WEDGES = 16


def load_circuit(path: str | Path = DATA_PATH) -> dict:
    """Load the baked MaleCNS CX slice. Pure numpy, no pandas needed."""
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _csr(pre: np.ndarray, post: np.ndarray, w: np.ndarray,
         n_pre: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(pre, kind="stable")
    pre_s, post_s, w_s = pre[order], post[order], w[order]
    counts = np.bincount(pre_s, minlength=n_pre)
    indptr = np.zeros(n_pre + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    return indptr, post_s.astype(np.int64), w_s.astype(np.float64)


def _csrmatvec(indptr, indices, data, x: np.ndarray, n_post: int) -> np.ndarray:
    out = np.zeros(n_post, dtype=np.float64)
    for i in range(len(indptr) - 1):
        s, e = indptr[i], indptr[i + 1]
        if e > s and x[i] != 0.0:
            out[indices[s:e]] += x[i] * data[s:e]
    return out


@dataclass
class FlyCompass:
    """Ring-attractor focus compass with sleep pressure. Call step() per turn."""

    n_features: int = 8
    rec_gain: float = 0.5
    pen_shift: int = 3          # max bump positions per step at |pen_drive|=1
    sleep_rate: float = 0.08
    sleep_decay: float = 0.02
    seed: int = 20260608
    data_path: str | Path = DATA_PATH

    bump: np.ndarray = field(init=False, default=None)
    sleep_pressure: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        circ = load_circuit(self.data_path)
        roles = circ["node_roles"]
        types = circ["node_types"]
        self._role_idx = {r: np.flatnonzero(roles == r)
                          for r in ("INPUT", "RING", "PEN", "PEG", "EPG", "PF")}
        self.n_epg = len(self._role_idx["EPG"])

        # EPG ring order from REAL wedge suffixes; unknown wedges trail.
        wedges = circ["node_wedges"][self._role_idx["EPG"]] % N_WEDGES
        epg_bodies = circ["node_ids"][self._role_idx["EPG"]]
        self._order = np.lexsort((epg_bodies, wedges))
        self._inv = np.empty_like(self._order)
        self._inv[self._order] = np.arange(self.n_epg)
        self._angles = (wedges[self._order].astype(np.float64) / N_WEDGES) * 2 * np.pi

        pos = np.full(len(roles), -1, dtype=np.int64)
        for r, idx in self._role_idx.items():
            pos[idx] = np.arange(len(idx))
        pre, post = circ["edge_pre"], circ["edge_post"]
        w = circ["edge_weight"].astype(np.float64)
        rpre, rpost = roles[pre], roles[post]

        def pathway(r1: str, r2: str):
            m = (rpre == r1) & (rpost == r2)
            p, q, v = pos[pre[m]], pos[post[m]], w[m].copy()
            if len(v):
                sums = np.bincount(q, weights=v, minlength=len(self._role_idx[r2]))
                v = v / np.maximum(sums[q], 1e-9)
            return _csr(p, q, v, len(self._role_idx[r1]))

        self._ring_epg = pathway("RING", "EPG")
        self._pen_epg = pathway("PEN", "EPG")
        self._peg_epg = pathway("PEG", "EPG")
        self._in_epg = pathway("INPUT", "EPG")
        self._epg_epg = pathway("EPG", "EPG")
        self._epg_pf = pathway("EPG", "PF")

        # ER5/R5 sleep-drive units (real neurons, Liu et al. 2016 proxy).
        ring_types = types[self._role_idx["RING"]]
        self._er5 = np.flatnonzero(ring_types == "ER5")
        # PFL steering split by REAL soma side.
        pf_types = types[self._role_idx["PF"]]
        pf_sides = circ["node_sides"][self._role_idx["PF"]]
        is_pfl = np.array([t.startswith("PFL") for t in pf_types])
        self._pfl_l = np.flatnonzero(is_pfl & (pf_sides == "L"))
        self._pfl_r = np.flatnonzero(is_pfl & (pf_sides == "R"))
        self.n_pfl_l, self.n_pfl_r = len(self._pfl_l), len(self._pfl_r)

        rng = np.random.default_rng(self.seed)
        n_in, n_ring = len(self._role_idx["INPUT"]), len(self._role_idx["RING"])
        self._sens_epg = rng.normal(0, 1, (self.n_features, n_in))
        self._sens_epg /= max(np.abs(self._sens_epg).sum(), 1e-9)
        self._sens_ring = rng.normal(0, 1, (self.n_features, n_ring))
        self._sens_ring /= max(np.abs(self._sens_ring).sum(), 1e-9)
        self.reset()

    def reset(self) -> None:
        self.bump = np.full(self.n_epg, 1.0 / self.n_epg)
        self.sleep_pressure = 0.0

    def step(self, features, pen_drive: float = 0.0, fatigue: float = 0.0) -> dict:
        """Advance one turn. features: 8-dim vector. Returns focus readouts."""
        f = np.asarray(features, dtype=np.float64).reshape(-1)
        if f.shape[0] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {f.shape[0]}")
        n_epg_total = self.n_epg
        # Real-weight drive into EPG (position order).
        def to_pos(vec_global_epg: np.ndarray) -> np.ndarray:
            return vec_global_epg[self._inv]
        drive = _csrmatvec(*self._ring_epg, np.maximum(self._sens_ring.T @ f, 0.0),
                           n_epg_total)
        drive += _csrmatvec(*self._pen_epg, np.full(len(self._role_idx["PEN"]), 0.2 + 0.8 * abs(float(pen_drive))),
                            n_epg_total)
        drive += _csrmatvec(*self._peg_epg, np.full(len(self._role_idx["PEG"]), 0.1), n_epg_total)
        drive += _csrmatvec(*self._in_epg, np.maximum(self._sens_epg.T @ f, 0.0), n_epg_total)
        b = to_pos(drive) + self.rec_gain * to_pos(
            _csrmatvec(*self._epg_epg, self.bump[self._order], n_epg_total))
        b = np.maximum(b, 0.0)
        # PEN angular-velocity rotation of the bump (real function, synthetic gain).
        s = int(round(float(np.clip(pen_drive, -1.0, 1.0)) * self.pen_shift))
        if s:
            b = np.roll(b, s)
        tot = b.sum()
        self.bump = b / tot if tot > 0 else np.full(self.n_epg, 1.0 / self.n_epg)
        # ER5 sleep-drive accumulation (real R5 proxy units).
        ring_act = np.maximum(self._sens_ring.T @ f, 0.0)
        er5 = float(ring_act[self._er5].mean()) if len(self._er5) else 0.0
        self.sleep_pressure = float(np.clip(
            self.sleep_pressure + self.sleep_rate * (0.5 * float(np.clip(fatigue, 0, 1)) + 0.5 * er5)
            - self.sleep_decay * self.sleep_pressure, 0.0, 1.0))
        return self.readout()

    def readout(self) -> dict:
        b_global = self.bump[self._order]
        pfl = _csrmatvec(*self._epg_pf, b_global, len(self._role_idx["PF"]))
        pl = float(pfl[self._pfl_l].sum()) if self.n_pfl_l else 0.0
        pr = float(pfl[self._pfl_r].sum()) if self.n_pfl_r else 0.0
        vec = (self.bump * np.exp(1j * self._angles)).sum()
        heading = float(np.degrees(np.angle(vec)) % 360.0)
        sharpness = float(min(1.0, abs(vec)))
        return {
            "heading_deg": heading,
            "sharpness": sharpness,
            "decisiveness": float(abs(pl - pr) / (pl + pr + 1e-9)),
            "pfl_drive": float(pl + pr),
            "sleep_pressure": float(self.sleep_pressure),
        }

    def summary(self) -> dict:
        return {
            "n_epg": self.n_epg, "n_pfl_l": self.n_pfl_l, "n_pfl_r": self.n_pfl_r,
            "n_er5": len(self._er5), "sleep_pressure": float(self.sleep_pressure),
        }
