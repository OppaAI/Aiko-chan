"""Fruit-fly mushroom-body (MB) associative memory for Aiko.

Biology (adult Drosophila, MaleCNS v1.0, Berg et al. 2025):
  sensory INPUT (PNs) -> ~4k Kenyon cells (sparse ~5% expansion) ->
  MBON readouts, with DAN (dopamine) teaching signals and APL/DPM global
  inhibition. PAM DANs pair with approach MBON compartments, PPL1 DANs with
  avoidance compartments; the opponent MBON sum steers behaviour.

What is real here vs synthetic (read README.md for details):
  REAL: every neuron, edge and weight downstream of INPUT comes from the
    MaleCNS v1.0 connectome slice shipped in data/mb_microcircuit.npz
    (KC->MBON, DAN->MBON, DAN->KC, APL->KC, MBON->MBON, MBON->DAN).
    MBON valence signs are derived from real PAM-vs-PPL1 innervation.
  SYNTHETIC: the feature->INPUT sensory projection (fixed seeded matrix),
    the DAN reward source (Aiko's valence_score), plasticity rule
    (functional opponent Hebbian abstraction), normalisation constants.

Runtime deps: numpy only. Deterministic given `seed`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).resolve().parent / "data" / "mb_microcircuit.npz"

# Fraction of Kenyon cells active per event (~5% in the fly).
KC_SPARSITY = 0.05


def load_circuit(path: str | Path = DATA_PATH) -> dict:
    """Load the baked MaleCNS MB slice. Pure numpy, no pandas needed."""
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _csr(pre: np.ndarray, post: np.ndarray, w: np.ndarray,
         n_pre: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiny CSR builder (pre-major). No scipy dependency."""
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
class FlyMB:
    """Opponent-valence associative memory wired with real fly connectivity."""

    n_features: int = 8
    kc_sparsity: float = KC_SPARSITY
    lr: float = 0.05
    apl_gain: float = 1.0
    rec_gain: float = 0.3
    seed: int = 20260608
    data_path: str | Path = DATA_PATH

    n_input: int = field(init=False, default=0)
    n_kc: int = field(init=False, default=0)
    n_mbon: int = field(init=False, default=0)
    mbon_sign: np.ndarray = field(init=False, default=None)  # +1 approach / -1 avoid / 0 neutral
    mbon_types: np.ndarray = field(init=False, default=None)

    def __post_init__(self) -> None:
        circ = load_circuit(self.data_path)
        roles = circ["node_roles"]
        self._role_idx = {r: np.flatnonzero(roles == r) for r in ("INPUT", "KC", "MBON", "DAN", "APLDPM")}
        self.n_input = len(self._role_idx["INPUT"])
        self.n_kc = len(self._role_idx["KC"])
        self.n_mbon = len(self._role_idx["MBON"])
        self.mbon_types = circ["node_types"][self._role_idx["MBON"]]

        # Global node ids -> compact per-role positions for each pathway matrix.
        pos = np.full(len(roles), -1, dtype=np.int64)
        for r, idx in self._role_idx.items():
            pos[idx] = np.arange(len(idx))
        pre, post, w = circ["edge_pre"], circ["edge_post"], circ["edge_weight"].astype(np.float64)
        rpre = roles[pre]
        rpost = roles[post]

        def pathway(r1: str, r2: str, normalise: str = "post"):
            m = (rpre == r1) & (rpost == r2)
            p = pos[pre[m]]
            q = pos[post[m]]
            v = w[m].copy()
            if normalise == "post" and len(v):
                sums = np.bincount(q, weights=v, minlength=len(self._role_idx[r2]))
                v = v / np.maximum(sums[q], 1e-9)
            elif normalise == "max" and len(v):
                v = v / max(v.max(), 1e-9)
            n1 = len(self._role_idx[r1])
            return _csr(p, q, v, n1), len(self._role_idx[r2])

        (self._ikc, _) = pathway("INPUT", "KC", "post")
        (self._akc, _) = pathway("APLDPM", "KC", "max")
        (self._kcm, _) = pathway("KC", "MBON", "post")
        (self._dkm, _) = pathway("DAN", "MBON", "max")
        (self._mm, _) = pathway("MBON", "MBON", "max")
        # Plastic overlay on KC->MBON (the learned association layer).
        _, kc_mbon_idx, kc_mbon_dat = self._kcm
        self._plastic = np.zeros_like(kc_mbon_dat)

        # Opponent signs from REAL DAN innervation: PAM≈reward, PPL1≈punishment.
        types = circ["node_types"]
        dan_types = types[self._role_idx["DAN"]]
        is_pam = np.array([t.startswith("PAM") for t in dan_types])
        is_ppl = np.array([t.startswith("PPL1") for t in dan_types])
        dind, didx, ddat = self._dkm
        pam_w = np.zeros(self.n_mbon)
        ppl_w = np.zeros(self.n_mbon)
        for j in range(len(dind) - 1):
            s, e = dind[j], dind[j + 1]
            if e > s:
                if is_pam[j]:
                    pam_w[didx[s:e]] += ddat[s:e]
                elif is_ppl[j]:
                    ppl_w[didx[s:e]] += ddat[s:e]
        self.mbon_sign = np.sign(pam_w - ppl_w).astype(np.float64)

        rng = np.random.default_rng(self.seed)
        self._sensory = rng.normal(0, 1, (self.n_features, self.n_input)).astype(np.float64)
        self._sensory /= max(np.abs(self._sensory).sum(axis=1, keepdims=True).max(), 1e-9)

    # -- core circuit ----------------------------------------------------
    def encode(self, features) -> np.ndarray:
        """Project features -> INPUT -> sparse KC pattern (dense float array)."""
        f = np.asarray(features, dtype=np.float64).reshape(-1)
        if f.shape[0] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {f.shape[0]}")
        drive_in = np.maximum(self._sensory.T @ f, 0.0)
        kc = _csrmatvec(*self._ikc, drive_in, self.n_kc)
        # APL/DPM global divisive inhibition keeps activity sparse and bounded.
        apl = _csrmatvec(*self._akc, np.ones(4), self.n_kc)
        kc = kc / (1.0 + self.apl_gain * (apl.mean() + kc.mean()))
        k = max(1, int(round(self.n_kc * self.kc_sparsity)))
        if k < self.n_kc:
            thr = np.partition(kc, -k)[-k]
            kc = np.where(kc >= thr, kc, 0.0)
        return kc

    def readout(self, kc: np.ndarray) -> tuple[float, float]:
        """Opponent (approach, avoid) MBON sums for an active KC pattern."""
        ind, idx, dat = self._kcm
        w = np.clip(dat + self._plastic, 0.0, None)
        mbon = np.zeros(self.n_mbon)
        for i in range(len(ind) - 1):
            s, e = ind[i], ind[i + 1]
            if e > s and kc[i] != 0.0:
                mbon[idx[s:e]] += kc[i] * w[s:e]
        # One recurrent MBON<->MBON step (real feedback weights).
        mbon = np.maximum(mbon + self.rec_gain * _csrmatvec(*self._mm, mbon, self.n_mbon), 0.0)
        approach = float(mbon[self.mbon_sign > 0].sum())
        avoid = float(mbon[self.mbon_sign < 0].sum())
        return approach, avoid

    def reinforce(self, kc: np.ndarray, reward: float) -> float:
        """DAN-like opponent update. reward in [-1, 1]; returns applied delta mass."""
        r = float(np.clip(reward, -1.0, 1.0))
        if r == 0.0:
            return 0.0
        active = np.flatnonzero(kc)
        if len(active) == 0:
            return 0.0
        ind, idx, _ = self._kcm
        total = 0.0
        for i in active:
            s, e = ind[i], ind[i + 1]
            if e <= s:
                continue
            tgt = idx[s:e]
            # Reward: potentiate approach MBONs, depress avoid MBONs (and mirror).
            direction = np.where(self.mbon_sign[tgt] >= 0, 1.0, -1.0) * np.sign(r)
            dw = self.lr * abs(r) * kc[i] * direction
            self._plastic[s:e] += dw
            total += float(np.abs(dw).sum())
        self._plastic = np.clip(self._plastic, -0.5, 0.5)
        return total

    def valence_bias(self, features, reward: float | None = None) -> float:
        """One-call helper: encode -> (optionally) reinforce -> opponent bias in [-1, 1]."""
        kc = self.encode(features)
        if reward is not None:
            self.reinforce(kc, reward)
        approach, avoid = self.readout(kc)
        denom = approach + avoid + 1e-9
        return float((approach - avoid) / denom)

    def summary(self) -> dict:
        return {
            "n_input": self.n_input, "n_kc": self.n_kc, "n_mbon": self.n_mbon,
            "kc_sparsity": self.kc_sparsity,
            "mbon_approach": int((self.mbon_sign > 0).sum()),
            "mbon_avoid": int((self.mbon_sign < 0).sum()),
            "mbon_neutral": int((self.mbon_sign == 0).sum()),
            "plastic_mass": float(np.abs(self._plastic).sum()),
        }
