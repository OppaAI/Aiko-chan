"""Stage 2: MB approach/avoid → memory retrieval score adjustment.

Closes the causal gap:
  memory candidate text → MB valence_bias → score delta → influence log

Used by memorize ranking paths. Never raises.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("aiko.memory.fly_rank")


def _mode() -> str:
    try:
        return (os.getenv("MEMORY_FLYMB_MODE", "off") or "off").strip().lower()
    except Exception:
        return "off"


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def mb_bias_for_text(text: str, *, user_id: str | None = None) -> float | None:
    try:
        from cognition.fly_registry import get_flymb
        from cognition.flymemory.circuit import text_features

        mb = get_flymb(user_id)
        if mb is None:
            return None
        return float(mb.valence_bias(text_features(text or "")))
    except Exception as exc:
        log.debug("mb_bias_for_text failed: %s", exc)
        return None


def adjust_recall_score(
    score: float,
    text: str,
    *,
    user_id: str | None = None,
    memory_id: str | int | None = None,
    weight_env: str = "MEMORY_FLYMB_LTM_W",
    default_weight: float = 0.04,
    log_influence: bool = True,
) -> tuple[float, dict]:
    """Apply MB bias to a retrieval score.

    Negative bias (avoidance) reduces score more aggressively so taught
    "don't bring this up" topics drop out of the top-k more reliably.
    Positive bias gives a milder approach lift.

    Returns (new_score, meta).
    """
    meta: dict = {"mode": _mode(), "bias": None, "delta": 0.0, "weight": 0.0}
    mode = meta["mode"]
    if mode not in ("shadow", "live"):
        return float(score), meta
    bias = mb_bias_for_text(text, user_id=user_id)
    if bias is None:
        return float(score), meta
    meta["bias"] = round(bias, 4)
    w = _float_env(weight_env, default_weight)
    meta["weight"] = w
    if bias < 0:
        delta = w * bias * _float_env("MEMORY_FLYMB_AVOID_MULT", 1.8)
    else:
        delta = w * bias
    meta["delta"] = round(delta, 5)
    new_score = float(score)
    if mode == "live":
        new_score = float(score) + delta
    if log_influence and abs(delta) >= 1e-5:
        try:
            from cognition.neural_state import get_neural_state

            get_neural_state(user_id).record_influence(
                {
                    "kind": "recall_rank",
                    "mode": mode,
                    "memory_id": memory_id,
                    "bias": meta["bias"],
                    "delta": meta["delta"],
                    "weight": w,
                    "text_preview": (text or "")[:60],
                }
            )
        except Exception:
            pass
    return new_score, meta
