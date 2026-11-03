"""Semantic sensory front-end for CX / LH (Stage 5).

Harrier embedding → fixed random projection → 8-D features for FlyCompass,
plus a continuous heading diagnostic (replaces SHA-1 topic hash).

Falls back to flymemory text_features() when embedding is unavailable.
Never raises.
"""
from __future__ import annotations

import logging
import math
import os
import threading

log = logging.getLogger("aiko.fly.semantic_features")

_N = 8
_proj_lock = threading.Lock()
_proj: list[list[float]] | None = None
_LAST_FEATS: dict[str, list[float]] = {}


def _uid(user_id: str | None) -> str:
    return (user_id or "").strip() or "default"


def _projection(dim: int) -> list[list[float]]:
    """Deterministic projection matrix (dim × 8)."""
    global _proj
    with _proj_lock:
        if _proj and len(_proj[0]) == dim:
            return _proj
        import random
        rng = random.Random(20260922)
        cols = []
        for _ in range(_N):
            v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            cols.append([x / n for x in v])
        _proj = cols
        return _proj


def _embed(text: str) -> list[float] | None:
    t = (text or "").strip()
    if not t:
        return None
    try:
        from cognition.memory.preference_store import _cached_vec
        return _cached_vec(t)
    except Exception:
        pass
    try:
        # Phase 2: shared singleton — a fresh HarrierEmbedder per call meant
        # a new requests.Session and an empty TTL cache every time.
        from cognition.memory.preference_store import _shared_embedder
        vec = list(_shared_embedder().embed([t]))[0]
        return [float(x) for x in vec]
    except Exception as exc:
        log.debug("semantic embed skipped: %s", exc)
        return None


def project_to_8d(vec: list[float]) -> list[float]:
    cols = _projection(len(vec))
    out = []
    for j in range(_N):
        s = sum(vec[i] * cols[j][i] for i in range(len(vec)))
        out.append(max(-1.0, min(1.0, s)))
    return out


def semantic_features(text: str, *, user_id: str | None = None) -> list[float]:
    """8-D features for CX INPUT projection. Caches per-user last vector."""
    vec = _embed(text)
    if vec is None:
        try:
            from cognition.flymemory.circuit import text_features
            feats = [float(x) for x in text_features(text or "")]
        except Exception:
            feats = [0.0] * _N
    else:
        feats = project_to_8d(vec)
    _LAST_FEATS[_uid(user_id)] = feats
    return feats


def heading_from_features(feats: list[float]) -> float:
    if len(feats) < 2:
        return 0.0
    return float((math.degrees(math.atan2(feats[1], feats[0])) + 360.0) % 360.0)


def sharpness_from_features(feats: list[float]) -> float:
    n = math.sqrt(sum(x * x for x in feats)) if feats else 0.0
    return float(max(0.0, min(1.0, n / math.sqrt(float(_N)))))


def last_semantic_features(user_id: str | None = None) -> list[float] | None:
    return _LAST_FEATS.get(_uid(user_id))


def blend_weight() -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv("FLY_CX_SEMANTIC_W", "0.55"))))
    except Exception:
        return 0.55
