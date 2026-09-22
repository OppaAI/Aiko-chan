"""Lateral-horn style contextual prior (Stage 5 semantic).

Stage 5: embedding cosine against a short ring of recent contexts when
Harrier is available; falls back to token fingerprint.

Modes (MEMORY_FLYLH_MODE): off | shadow | live
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections import deque

from system.config import env_str, env_float

log = logging.getLogger("aiko.fly_behavior.lateral_horn")

_seen_fp: dict[str, set[str]] = {}
_seen_vec: dict[str, deque] = {}
_VEC_MAX = 32


def _mode() -> str:
    try:
        return env_str("MEMORY_FLYLH_MODE", "off").strip().lower()
    except Exception:
        return "off"


def _tok_key(text: str) -> str:
    toks = sorted({w for w in (text or "").lower().split() if len(w) > 3})[:40]
    h = hashlib.sha1(" ".join(toks).encode("utf-8")).hexdigest()[:16]
    return h


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return max(-1.0, min(1.0, dot / (na * nb)))


def context_prior(text: str, *, user_id: str | None = None, record: bool = True) -> dict:
    """familiarity in [0,1]: higher = more like recently seen context."""
    mode = _mode()
    if mode not in ("shadow", "live"):
        return {"mode": mode, "familiarity": 0.5, "novel": False}
    key = (user_id or "").strip()
    if not key:
        return {"mode": mode, "familiarity": 0.5, "novel": False}

    familiarity = 0.5
    novel = True
    method = "fingerprint"

    try:
        from cognition.memory.preference_store import _cached_vec

        vec = _cached_vec(text or "")
        if vec is not None:
            method = "embedding"
            buf = _seen_vec.setdefault(key, deque(maxlen=_VEC_MAX))
            best = 0.0
            for prev in buf:
                best = max(best, _cosine(vec, prev))
            familiarity = max(0.1, min(0.95, 0.15 + 0.8 * max(0.0, best)))
            novel = best < 0.75
            if record:
                buf.append(list(vec))
    except Exception as exc:
        log.debug("lh semantic skipped: %s", exc)
        method = "fingerprint"

    if method == "fingerprint":
        bucket = _seen_fp.setdefault(key, set()) if record else _seen_fp.get(key, set())
        fp = _tok_key(text)
        familiar = fp in bucket
        if record:
            bucket.add(fp)
            if len(bucket) > 256:
                for i, x in enumerate(list(bucket)):
                    if i % 2 == 0:
                        bucket.discard(x)
        familiarity = 0.85 if familiar else max(0.15, 1.0 - (len(bucket) / 256.0) * 0.5)
        novel = not familiar

    w = env_float("MEMORY_FLYLH_W", 0.05)
    return {
        "mode": mode,
        "familiarity": round(familiarity, 4),
        "novel": bool(novel),
        "method": method,
        "bias": round((familiarity - 0.5) * 2.0 * w, 4) if mode == "live" else 0.0,
    }


def adjust_score(base: float, text: str, *, user_id: str | None = None, record: bool = True) -> float:
    """Apply LH familiarity bias to a ranking score when mode is live."""
    prior = context_prior(text or "", user_id=user_id, record=record)
    mode = prior.get("mode") or _mode()
    if mode == "shadow":
        log.debug(
            "flylh score mode=shadow familiarity=%.2f method=%s",
            prior.get("familiarity", 0.5),
            prior.get("method"),
        )
        return base
    if mode == "live":
        b = float(prior.get("bias") or 0.0)
        log.debug("flylh score mode=live bias=%+.3f method=%s", b, prior.get("method"))
        return base + b
    return base
