"""Whole-brain simulation on the full MaleCNS v1.0 connectome (Phase 4).

Biology (Berg et al. 2025, CC-BY): ~166.4k neurons, ~25.6M directed
connections — exactly the paper's quantify-neuron-connections.ipynb
counting (both ends valid-superclass, no weight threshold).

What is real vs synthetic:
  REAL: every neuron, edge and signed weight in data/malecns_full.npz;
    NT signs from Janelia neurotransmitter predictions (GABA -> inhibitory).
  SYNTHETIC: the KC seed (comes from FlyMB.encode), the 2-hop ReLU
    propagation abstraction, readout groupings (DN/clock).

Runtime deps: numpy only. The full graph is ~310MB RAM; it loads ONCE via
the get_fullbrain() singleton (Phase 4 rule 3). No LIF — rate-based,
matching the rest of the fly brain.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import numpy as np

try:
    from system.log import get_logger as _get_logger
    log = _get_logger(__name__)
except Exception:
    import logging as _logging
    log = _logging.getLogger("aiko.fullbrain")

DATA_NAME = "malecns_full.npz"
# Set after the GitHub release upload; env override always wins.
# Default remote for the prebuilt artifact (~80MB, not committed to git).
# NOTE: the asset must be uploaded to this release once (see Phase 4 PR notes);
# until then the download 404s and the runtime degrades gracefully.
DEFAULT_RELEASE_URL = (
    "https://github.com/OppaAI/Aiko-chan/releases/download/"
    "flybrain-data-v1/malecns_full.npz"
)
RELEASE_URL = os.environ.get("AIKO_FULLBRAIN_URL", DEFAULT_RELEASE_URL).strip()

_lock = threading.RLock()
_instance: "FullBrain | None" = None
_missing_logged = False
_load_started = False


def ensure_data() -> Path | None:
    """Locate or prepare validated data; call only off the turn path."""
    candidates = []
    env = os.environ.get("AIKO_FULLBRAIN_PATH", "").strip()
    if env:
        candidates.append(Path(env))
    candidates.append(Path.home() / ".aiko" / "data" / DATA_NAME)
    candidates.append(Path(__file__).resolve().parent / "data" / DATA_NAME)
    for c in candidates:
        if c.is_file():
            try:
                load_fullbrain(c)
                return c
            except Exception as exc:
                log.warning("fullbrain: invalid artifact %s: %s", c, exc)
    if RELEASE_URL:
        dest = Path.home() / ".aiko" / "data" / DATA_NAME
        temporary = None
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            log.info("fullbrain: downloading %s", RELEASE_URL)
            with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=f".{DATA_NAME}.", suffix=".npz", delete=False) as tmp:
                temporary = Path(tmp.name)
            urllib.request.urlretrieve(RELEASE_URL, temporary)
            load_fullbrain(temporary)
            os.replace(temporary, dest)
            return dest
        except Exception as exc:
            log.warning("fullbrain: download failed: %s", exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return None


def load_fullbrain(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    required = {"indptr", "indices", "data", "pre_sorted", "node_ids", "node_types", "node_sign"}
    if missing := required - out.keys():
        raise ValueError(f"fullbrain artifact missing arrays: {sorted(missing)}")
    n = len(out["node_ids"])
    edges = len(out["indices"])
    if (
        any(out[key].ndim != 1 for key in required)
        or len(out["indptr"]) != n + 1
        or len(out["node_types"]) != n
        or len(out["node_sign"]) != n
        or len(out["data"]) != edges
        or len(out["pre_sorted"]) != edges
        or out["indptr"][0] != 0
        or out["indptr"][-1] != edges
    ):
        raise ValueError("fullbrain artifact has inconsistent array dimensions")
    # meta_json is a 0-d object array; normalise to str
    m = out.get("meta_json")
    if m is not None:
        out["meta_json"] = str(m)
    return out


# Clock-neuron type prefixes in MaleCNS (dorsal/lateral clock network).
_CLOCK_PREFIXES = ("DN1a", "DN1p", "DN2", "DN3", "LNd", "s-LNv", "l-LNv", "LPN")


def _is_clock(t: str) -> bool:
    return t.startswith(_CLOCK_PREFIXES)


def _is_descending(t: str) -> bool:
    return t.startswith("DN") and not _is_clock(t)


class FullBrain:
    """Two-hop whole-brain propagation seeded by the MB KC pattern."""

    def __init__(self, path: str | Path | None = None) -> None:
        d = load_fullbrain(path or ensure_data())
        self.pre = np.ascontiguousarray(d["pre_sorted"], dtype=np.int32)
        self.post = np.ascontiguousarray(d["indices"], dtype=np.int32)
        self.w = np.ascontiguousarray(d["data"], dtype=np.float32)
        self.n = int(d["indptr"].shape[0] - 1)
        self.node_ids = d["node_ids"].astype(np.int64)
        types = d["node_types"].astype(str)
        self._clock_mask = np.array([_is_clock(t) for t in types], dtype=bool)
        self._dn_mask = np.array([_is_descending(t) for t in types], dtype=bool)
        self._id2pos = {int(b): i for i, b in enumerate(self.node_ids)}
        try:
            meta = d.get("meta_json", "")
            import json
            self.meta = json.loads(meta) if meta else {}
        except Exception:
            self.meta = {}
        log.info(
            "fullbrain: loaded n=%d edges=%d clock=%d DN=%d",
            self.n, len(self.pre),
            int(self._clock_mask.sum()), int(self._dn_mask.sum()),
        )

    # -- core ---------------------------------------------------------
    def _matvec(self, x: np.ndarray) -> np.ndarray:
        # Vectorised scatter-reduce (Phase 4 rule 2). x float32 in, float32 out.
        # In-place multiply avoids one 25.6M-element temp allocation.
        tmp = x[self.pre]
        tmp *= self.w
        return np.bincount(self.post, weights=tmp, minlength=self.n).astype(np.float32)

    @staticmethod
    def _norm_relu(y: np.ndarray) -> np.ndarray:
        np.maximum(y, 0.0, out=y)
        m = float(y.mean())
        if m > 0:
            y /= (1.0 + m)
        return y

    # Reference total injected "energy" per step. Encode's raw KC values sum
    # to ~1.6 (arbitrary internal scale); normalising the seed decouples the
    # readout magnitudes from that scale: ~1 unit per active KC.
    SEED_ENERGY = 200.0

    def step(self, kc_values: np.ndarray, kc_body_ids: np.ndarray) -> dict:
        """Seed KCs, propagate 2 hops, read out DN / clock / arousal."""
        x = np.zeros(self.n, dtype=np.float32)
        pos = np.array(
            [self._id2pos.get(int(b), -1) for b in kc_body_ids], dtype=np.int64
        )
        ok = pos >= 0
        kval = kc_values[ok].astype(np.float32) if ok.any() else None
        total = float(kval[kval > 0].sum()) if kval is not None else 0.0
        if total <= 0:
            return {"dn_drive": 0.0, "clock_drive": 0.0, "arousal": 0.0,
                    "n_seed": 0}
        x[pos[ok]] = kval * (self.SEED_ENERGY / total)
        y = self._norm_relu(self._matvec(x))
        z = self._norm_relu(self._matvec(y))
        dn = float(z[self._dn_mask].mean()) if self._dn_mask.any() else 0.0
        ck = float(z[self._clock_mask].mean()) if self._clock_mask.any() else 0.0
        return {
            "dn_drive": dn,
            "clock_drive": ck,
            "arousal": float(z.mean()),
            "n_seed": int((kval > 0).sum()),
        }

    def summary(self) -> dict:
        return {
            "n_neurons": self.n,
            "n_edges": int(len(self.pre)),
            "n_clock": int(self._clock_mask.sum()),
            "n_descending": int(self._dn_mask.sum()),
            "ram_mb": round(
                (self.pre.nbytes + self.post.nbytes + self.w.nbytes) / 1e6, 1
            ),
            "meta": self.meta,
        }


def _load_fullbrain_background() -> None:
    global _instance, _missing_logged
    path = ensure_data()
    brain = None
    if path is not None:
        try:
            brain = FullBrain(path)
        except Exception as exc:
            log.warning("fullbrain: load failed: %s", exc)
    with _lock:
        _instance = brain
        if brain is None and not _missing_logged:
            log.warning(
                "fullbrain: malecns_full.npz unavailable; whole-brain step "
                "disabled (run cognition/flymemory/tools/extract_full.py)"
            )
            _missing_logged = True


def get_fullbrain() -> FullBrain | None:
    """Return the process-wide brain, preparing it off the turn path once."""
    global _load_started
    start = False
    with _lock:
        if _instance is None and not _load_started:
            _load_started = True
            start = True
        brain = _instance
    if start:
        threading.Thread(target=_load_fullbrain_background, name="fullbrain-loader", daemon=True).start()
    return brain
