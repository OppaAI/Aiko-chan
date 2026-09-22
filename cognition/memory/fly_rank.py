"""Stage 2/3: MB approach/avoid → memory retrieval score adjustment.

Closes the causal gap:
  memory candidate text → MB valence_bias → score delta → influence log

Stage 3 also blends durable preference facts and a weak subliminal prior.
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
    """Apply MB bias to a retrieval score."""
    meta: dict = {"mode": _mode(), "bias": None, "delta": 0.0, "weight": 0.0}
    mode = meta["mode"]
    if mode not in ("shadow", "live"):
        return float(score), meta
    bias = mb_bias_for_text(text, user_id=user_id)
    delta = 0.0
    if bias is not None:
        meta["bias"] = round(bias, 4)
        w = _float_env(weight_env, default_weight)
        meta["weight"] = w
        delta = w * bias * _float_env("MEMORY_FLYMB_AVOID_MULT", 1.8) if bias < 0 else w * bias
    try:
        from cognition.memory.preference_store import preference_delta
        pdelta = preference_delta(text, user_id=user_id)
        meta["pref_delta"] = round(pdelta, 5)
        delta += pdelta
    except Exception:
        pass
    try:
        sw = _float_env("MEMORY_FLY_SUBLIMINAL_W", 0.2)
        if sw and user_id is not None:
            from cognition.neural_state import peek_neural_state
            st = peek_neural_state(user_id)
            if st is not None:
                sub = float(getattr(st, "valence", 0.0) or 0.0)
                delta += sw * 0.01 * sub
                meta["sub_blend"] = round(sw * 0.01 * sub, 5)
    except Exception:
        pass
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
                    "weight": meta["weight"],
                    "text_preview": (text or "")[:60],
                }
            )
        except Exception:
            pass
    return new_score, meta
