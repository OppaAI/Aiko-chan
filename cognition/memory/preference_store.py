"""Durable prefer/avoid facts (Stage 3 / A–H G).

JSON list under the user state dir. Used by teach_api and recall ranking
so a taught topic is more than a one-shot MB reinforce.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path

log = logging.getLogger("aiko.memory.preference_store")

_MAX = 64

# Semantic fallback tuning (Stage 3 follow-up).
# Substring hits keep full weight; token-Jaccard and embedding-cosine hits
# apply scaled weights so paraphrases ("birthday pastry" for "fruit tarts")
# suppress without equating them to literal matches.
_JACCARD_THRESHOLD = 0.45
_COSINE_THRESHOLD = 0.75
_SEMANTIC_SCALE = 0.7
_JACCARD_SCALE = 0.5
_VEC_CACHE_MAX = 512
_EMBED_TIMEOUT_S = 1.5
_EMBED_COOLDOWN_S = 300.0

_vec_cache: dict[str, list[float] | None] = {}
_embed_cooldown_until: float = 0.0

# Phase 2: one shared embedder for the CX/semantic path. Constructing a fresh
# HarrierEmbedder per call meant a new requests.Session and an empty per-
# instance TTL cache every time — the instance cache could never hit.
_embedder_lock = threading.Lock()
_embedder = None


def _shared_embedder():
    """Process-wide HarrierEmbedder for raw (uninstructed) CX embeddings."""
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            from cognition.memory.vecstore import HarrierEmbedder
            _embedder = HarrierEmbedder(timeout=_EMBED_TIMEOUT_S)
        return _embedder


def _semantic_enabled() -> bool:
    try:
        return (os.getenv("PREF_SEMANTIC", "1") or "1").strip().lower() not in (
            "0", "off", "false", "no",
        )
    except Exception:
        return True


def _content_tokens(text: str) -> frozenset[str]:
    try:
        from cognition.memory.diversity import _content_tokens as _tok
        toks = _tok(text)
    except Exception:
        toks = frozenset((text or "").lower().split())
    # Crude plural fold so "tart"/"tarts" match (mirrors circuit lexicon).
    out = set()
    for w in toks:
        out.add(w)
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            out.add(w[:-1])
    return frozenset(out)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _cached_vec(text: str) -> list[float] | None:
    """Best-effort Harrier embedding with in-process cache + cooldown breaker.

    Never raises; returns None when disabled, unavailable, or failed.
    Keeps recall ranking fast: one timeout per cooldown window max, then
    skips until the window expires.
    """
    global _embed_cooldown_until
    t = (text or "").strip()
    if not t or not _semantic_enabled():
        return None
    if t in _vec_cache:
        return _vec_cache[t]
    now = time.time()
    if now < _embed_cooldown_until:
        return None
    try:
        vec = list(_shared_embedder().embed([t]))[0]
        import numpy as _np
        arr = _np.asarray(vec, dtype=_np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError("empty embedding")
        out = [float(x) for x in arr]
    except Exception as exc:
        log.debug("preference embedding skipped: %s", exc)
        _embed_cooldown_until = now + _EMBED_COOLDOWN_S
        return None
    if len(_vec_cache) >= _VEC_CACHE_MAX:
        _vec_cache.pop(next(iter(_vec_cache)), None)
    _vec_cache[t] = out
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    try:
        import math
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / (na * nb)
    except Exception:
        return 0.0


def _path(user_id: str | None) -> Path | None:
    try:
        from system.userspace import user_state_dir
        root = Path(user_state_dir(user_id))
        root.mkdir(parents=True, exist_ok=True)
        return root / "fly_preferences.json"
    except Exception:
        return None


def load_preferences(user_id: str | None = None) -> list[dict]:
    p = _path(user_id)
    if p is None or not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return list(data) if isinstance(data, list) else []
    except Exception as exc:
        log.debug("load_preferences failed: %s", exc)
        return []


def record_preference(topic: str, direction: str, *, user_id: str | None = None) -> dict:
    topic = (topic or "").strip()
    direction = "prefer" if str(direction).lower() in ("prefer", "approach", "like", "want") else "avoid"
    out = {"topic": topic, "direction": direction, "ts": time.time()}
    if not topic:
        return out
    # Pre-warm the semantic cache and persist the topic vector best-effort
    # so later recall turns skip the embed call for the topic side.
    try:
        vec = _cached_vec(topic)
        if vec is not None:
            out["vec"] = vec
    except Exception:
        pass
    p = _path(user_id)
    if p is not None:
        lock_fd = None
        tmp_path = None
        try:
            lock_fd = os.open(p.with_suffix(p.suffix + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            rows = [
                r
                for r in load_preferences(user_id)
                if str(r.get("topic") or "").lower() != topic.lower()
            ]
            rows.append(out)
            rows = rows[-_MAX:]
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=p.parent,
                prefix=f".{p.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                json.dump(rows, tmp, indent=0)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, p)
            tmp_path = None
        except Exception as exc:
            log.debug("record_preference write failed: %s", exc)
        finally:
            if tmp_path is not None:
                with suppress(OSError):
                    tmp_path.unlink(missing_ok=True)
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
    return out


def preference_delta(text: str, *, user_id: str | None = None) -> float:
    """Score nudge: + for prefer-topic overlap, − for avoid-topic overlap.

    Literal substring hits keep full weight. Token-Jaccard (>=0.45) and
    embedding-cosine (>=0.75) fallbacks catch paraphrases at scaled weight.
    Never raises; falls back to literal-only on any embedding failure.
    """
    low = (text or "").lower()
    if not low:
        return 0.0
    text_tokens: frozenset[str] | None = None
    text_vec: list[float] | None | bool = False  # False = not attempted yet
    delta = 0.0
    for row in load_preferences(user_id):
        topic = str(row.get("topic") or "")
        topic_low = topic.lower()
        if len(topic_low) < 2:
            continue
        full = 0.04 if row.get("direction") == "prefer" else -0.08
        if topic_low in low:
            delta += full
            continue
        # Token-Jaccard fallback (cheap, no I/O).
        try:
            if text_tokens is None:
                text_tokens = _content_tokens(low)
            topic_tokens = _content_tokens(topic_low)
            if _jaccard(text_tokens, topic_tokens) >= _JACCARD_THRESHOLD:
                delta += full * _JACCARD_SCALE
                continue
        except Exception:
            pass
        # Embedding-cosine fallback (best-effort, cached, cooldown-guarded).
        try:
            if text_vec is False:
                text_vec = _cached_vec(low)
            if text_vec is None:
                continue
            topic_vec = row.get("vec")
            if not isinstance(topic_vec, list):
                topic_vec = _cached_vec(topic_low)
                if topic_vec is None:
                    continue
            if _cosine(topic_vec, text_vec) >= _COSINE_THRESHOLD:
                delta += full * _SEMANTIC_SCALE
        except Exception:
            continue
    return max(-0.25, min(0.15, delta))
