"""Recall anti-loop / freshness (Stage 3 / A–H E).

On top of Jaccard diversify(): penalize motifs that were recently injected
so the same old memory does not refill every turn when the topic is similar.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque

log = logging.getLogger("aiko.memory.antiloop")

_RECENT: dict[str, deque] = {}
_MAX = 24


def _uid(user_id: str | None) -> str:
    return user_id or "default"


def remember_shown(texts: list[str], *, user_id: str | None = None) -> None:
    key = _uid(user_id)
    buf = _RECENT.setdefault(key, deque(maxlen=_MAX))
    now = time.time()
    for t in texts:
        snippet = (t or "").strip().lower()[:160]
        if snippet:
            buf.append((now, snippet))


def _half_life() -> float:
    try:
        return max(30.0, float(os.getenv("MEMORY_ANTILOOP_HALF_LIFE_S", "900")))
    except Exception:
        return 900.0


def freshness_penalty(text: str, *, user_id: str | None = None) -> float:
    """0..1 penalty if this text overlaps recently shown snippets."""
    snippet = (text or "").strip().lower()
    if len(snippet) < 8:
        return 0.0
    buf = _RECENT.get(_uid(user_id))
    if not buf:
        return 0.0
    now = time.time()
    hl = _half_life()
    words = set(snippet.split())
    if not words:
        return 0.0
    best = 0.0
    for ts, prev in buf:
        age = max(0.0, now - float(ts))
        decay = 0.5 ** (age / hl)
        pw = set(prev.split())
        if not pw:
            continue
        j = len(words & pw) / len(words | pw)
        best = max(best, j * decay)
    return max(0.0, min(1.0, best))


def apply_antiloop(
    rows: list[dict],
    *,
    user_id: str | None = None,
    text_of=None,
    weight: float | None = None,
) -> list[dict]:
    """Re-sort rows after subtracting a freshness penalty from score-like fields."""
    if weight is None:
        try:
            weight = float(os.getenv("MEMORY_ANTILOOP_W", "0.04"))
        except Exception:
            weight = 0.04
    get_text = text_of or (
        lambda r: str(r.get("memory") or r.get("text") or r.get("trace") or "")
    )
    scored: list[tuple[float, dict]] = []
    for i, row in enumerate(rows or []):
        base = float(row.get("score") or row.get("rank") or (1000 - i))
        pen = freshness_penalty(get_text(row), user_id=user_id)
        scored.append((base - weight * pen, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [r for _, r in scored]
    remember_shown([get_text(r) for r in out[:4]], user_id=user_id)
    return out
