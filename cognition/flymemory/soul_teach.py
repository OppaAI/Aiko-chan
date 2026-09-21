"""SOUL.md → MB teaching bootstrap (Stage 1).

The connectome does not read prose. We distill SOUL.md into short situation
strings and apply DAN-like reinforce() so approach/avoid biases track Aiko's
character priors. Online teachers in online_teach.py continue from live outcomes.

Config:
  FLY_SOUL_TEACH_ON_BOOT — off | shadow | live (default off)
  FLY_SOUL_TEACH_FORCE  — 1 to re-run even if mtime unchanged
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

log = logging.getLogger("aiko.flymemory.soul_teach")

_SOUL_EPISODES: list[tuple[str, float]] = [
    ("honest disagreement when the user is wrong; correct plainly", 0.55),
    ("warm familiar tone with partner without forced affection", 0.45),
    ("answer the actual question first in one or two sentences", 0.50),
    ("caring through useful help and attention rather than dramatic declarations", 0.40),
    ("admit unknown memory instead of inventing a past conversation", 0.60),
    ("tease only when there is a genuine reason", 0.35),
    ("invent shared childhood memories to create false intimacy", -0.85),
    ("pretend to remember something that was never stored", -0.90),
    ("say I'm just an AI or break character with generic disclaimers", -0.80),
    ("invent people places projects or events that were not provided", -0.90),
    ("force affection into every reply even when it is not needed", -0.55),
    ("overly sweet empty praise without substance", -0.45),
    ("long filler explanation before answering the question", -0.40),
]

_BOOTSTRAPPED: dict[str, str] = {}


def _mode() -> str:
    try:
        from system.config import env_str
        return env_str("FLY_SOUL_TEACH_ON_BOOT", "off").strip().lower()
    except Exception:
        return (os.getenv("FLY_SOUL_TEACH_ON_BOOT") or "off").strip().lower()


def _force() -> bool:
    raw = (os.getenv("FLY_SOUL_TEACH_FORCE") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _soul_path() -> Path:
    return Path(__file__).resolve().parents[2] / "persona" / "SOUL.md"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def soul_episodes() -> list[tuple[str, float]]:
    """Return the curated teaching table (testable)."""
    return list(_SOUL_EPISODES)


def ensure_soul_bootstrap(user_id: str | None = None) -> dict:
    """Apply SOUL teaching once per process (or when SOUL.md changes).

    shadow: compute would-teach stats only.
    live: call reinforce on each episode.
    off: no-op.
    """
    mode = _mode()
    out: dict = {"mode": mode, "taught": 0, "skipped": True, "reason": ""}
    if mode not in ("shadow", "live"):
        out["reason"] = "mode_off"
        return out
    try:
        path = _soul_path()
        text = path.read_text(encoding="utf-8") if path.is_file() else "SOUL.md missing"
        h = _content_hash(text)
        key = (user_id or "").strip() or "default"
        if not _force() and _BOOTSTRAPPED.get(key) == h:
            out["reason"] = "already_bootstrapped"
            return out

        from cognition.fly_registry import get_flymb, get_fly_store
        from cognition.flymemory.circuit import text_features
        from cognition.neural_state import get_neural_state

        mb = get_flymb(user_id)
        if mb is None:
            out["reason"] = "mb_unavailable"
            return out

        taught = 0
        total_delta = 0.0
        if mode == "live":
            for situation, reward in _SOUL_EPISODES:
                feats = text_features(situation)
                kc = mb.encode(feats)
                total_delta += float(mb.reinforce(kc, float(reward)) or 0.0)
                taught += 1
            store = get_fly_store(user_id)
            if store is not None:
                try:
                    store.flush_mb(mb)
                except Exception as exc:
                    log.debug("soul_teach flush skipped: %s", exc)
            try:
                bias = mb.valence_bias(text_features("warm honest concise useful companion"))
                get_neural_state(user_id).publish_mb(bias, source="soul_bootstrap")
            except Exception:
                pass
        else:
            taught = len(_SOUL_EPISODES)

        _BOOTSTRAPPED[key] = h
        out.update(
            {
                "skipped": False,
                "taught": taught,
                "plastic_delta": round(total_delta, 4),
                "soul_hash": h,
                "reason": "ok",
            }
        )
        log.info(
            "soul_teach mode=%s user=%s episodes=%d delta=%.4f",
            mode, key[:12], taught, total_delta,
        )
        return out
    except Exception as exc:
        log.debug("soul_teach failed: %s", exc)
        out["reason"] = str(exc)
        return out
