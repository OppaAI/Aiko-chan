"""Lateral-horn style contextual prior (functional abstraction).

In the fly, the lateral horn carries innate/contextual olfactory biases in
parallel with the mushroom body's learned associations. Here LH is a *fast
context familiarity prior* — not another memory store.

Modes (MEMORY_FLYLH_MODE): off | shadow | live
"""
from __future__ import annotations

import hashlib
import logging
from system.config import env_str, env_float

log = logging.getLogger("aiko.fly_behavior.lateral_horn")

_seen: dict[str, set[str]] = {}


def _mode() -> str:
    try:
        return env_str("MEMORY_FLYLH_MODE", "off").strip().lower()
    except Exception:
        return "off"


def _tok_key(text: str) -> str:
    toks = sorted({w for w in (text or "").lower().split() if len(w) > 3})[:40]
    h = hashlib.sha1(" ".join(toks).encode("utf-8")).hexdigest()[:16]
    return h


def context_prior(text: str, *, user_id: str | None = None) -> dict:
    """familiarity in [0,1]: higher = more like recently seen context."""
    mode = _mode()
    if mode not in ("shadow", "live"):
        return {"mode": mode, "familiarity": 0.5, "novel": False}
    key = (user_id or "").strip()
    if not key:
        return {"mode": mode, "familiarity": 0.5, "novel": False}
    bucket = _seen.setdefault(key, set())
    fp = _tok_key(text)
    familiar = fp in bucket
    bucket.add(fp)
    if len(bucket) > 256:
        for i, x in enumerate(list(bucket)):
            if i % 2 == 0:
                bucket.discard(x)
    familiarity = 0.85 if familiar else max(0.15, 1.0 - (len(bucket) / 256.0) * 0.5)
    w = env_float("MEMORY_FLYLH_W", 0.05)
    return {
        "mode": mode,
        "familiarity": round(familiarity, 4),
        "novel": not familiar,
        "bias": round((familiarity - 0.5) * 2.0 * w, 4) if mode == "live" else 0.0,
    }


def adjust_score(base: float, text: str, *, user_id: str | None = None) -> float:
    """Apply LH familiarity bias to a ranking score when mode is live.

    Shadow: log only. Off: return base unchanged.
    """
    prior = context_prior(text or "", user_id=user_id)
    mode = prior.get("mode") or _mode()
    if mode == "shadow":
        log.debug("flylh score mode=shadow familiarity=%.2f", prior.get("familiarity", 0.5))
        return base
    if mode == "live":
        b = float(prior.get("bias") or 0.0)
        log.debug("flylh score mode=live bias=%+.3f", b)
        return base + b
    return base
