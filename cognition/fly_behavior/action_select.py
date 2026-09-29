"""Phase 5 — fly → action selection.

The LLM proposes; the fly disposes (biases / vetoes / ranks candidates).

Modes (AIKO_FLY_ACTION_MODE):
  off     — no scoring
  shadow  — score + log, LLM winner still used
  live    — fly ranking can change the winner (blend weight)

Weight: AIKO_FLY_ACTION_WEIGHT (default 0.25) blends fly final into ranking.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)


def _mode() -> str:
    return (os.environ.get("AIKO_FLY_ACTION_MODE") or "shadow").strip().lower()


def _weight() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("AIKO_FLY_ACTION_WEIGHT", "0.25") or 0.25)))
    except Exception:
        return 0.25


def score_candidates(
    candidates: list[str] | dict[str, Any],
    *,
    user_id: str | None = None,
    context: dict | None = None,
    llm_prior: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score action candidates with fly-derived signals.

    Returns a record dict suitable for logging / metrics / winner selection.
    Never raises.
    """
    record: dict[str, Any] = {
        "ts": time.time(),
        "mode": _mode(),
        "weight": _weight(),
        "candidates": {},
        "winner": None,
        "llm_winner": None,
        "applied": False,
        "ambiguous": False,
        "user_id": user_id,
    }
    try:
        if isinstance(candidates, dict):
            names = list(candidates.keys())
            priors = {k: float(v) if not isinstance(v, dict) else float(v.get("score", 0)) for k, v in candidates.items()}
        else:
            names = list(candidates or [])
            priors = {n: 1.0 / max(1, len(names)) for n in names}
        if llm_prior:
            for k, v in llm_prior.items():
                priors[k] = float(v)

        if not names:
            return record

        # NeuralState readouts (best-effort)
        valence = 0.0
        approach = 0.0
        avoidance = 0.0
        decisiveness = 0.5
        urgency = 0.0
        interrupt = False
        try:
            from cognition.neural_state import get_state
            ns = get_state(user_id)
            if ns:
                valence = float(getattr(ns, "valence", 0) or 0)
                approach = float(getattr(ns, "approach", 0) or 0)
                avoidance = float(getattr(ns, "avoidance", 0) or 0)
                decisiveness = float(getattr(ns, "decisiveness", 0.5) or 0.5)
                urgency = float(getattr(ns, "urgency", 0) or 0)
                interrupt = bool(getattr(ns, "interrupt", False))
        except Exception:
            pass

        w = _weight()
        scored: dict[str, dict] = {}
        for name in names:
            prior = float(priors.get(name, 0.0))
            # Simple fly bias: approach favors exploratory / social, avoidance favors safe / defer
            bias = 0.0
            nlow = name.lower()
            if any(k in nlow for k in ("approach", "engage", "ask", "explore", "help", "play")):
                bias += 0.3 * approach + 0.1 * valence
            if any(k in nlow for k in ("avoid", "defer", "refuse", "stop", "wait", "safe")):
                bias += 0.3 * avoidance - 0.1 * valence
            if any(k in nlow for k in ("interrupt", "barge", "urgent")):
                bias += 0.4 * urgency
            if interrupt and "interrupt" in nlow:
                bias += 0.5
            final = (1.0 - w) * prior + w * (prior + bias)
            veto = False
            if interrupt and any(k in nlow for k in ("continue", "ignore", "proceed")):
                veto = True
                final *= 0.1
            scored[name] = {
                "llm_prior": prior,
                "fly_bias": round(bias, 4),
                "final": round(final, 4),
                "veto": veto,
            }

        record["candidates"] = scored
        if scored:
            llm_win = max(scored.items(), key=lambda kv: kv[1]["llm_prior"])[0]
            fly_win = max(scored.items(), key=lambda kv: kv[1]["final"])[0]
            record["llm_winner"] = llm_win
            record["winner"] = fly_win
            # ambiguity: top two finals within 5%
            finals = sorted((v["final"] for v in scored.values()), reverse=True)
            if len(finals) >= 2 and finals[0] > 0 and (finals[0] - finals[1]) / finals[0] < 0.05:
                record["ambiguous"] = True

            mode = _mode()
            if mode == "live" and fly_win != llm_win and not record["ambiguous"]:
                record["applied"] = True
            elif mode == "live" and fly_win == llm_win:
                record["applied"] = False
            # shadow / off: applied stays False

        # Phase 12: aggregate metrics (never raises).
        try:
            from cognition.fly_behavior import metrics as _fly_metrics
            _fly_metrics.record_action(record, user_id=user_id)
        except Exception:
            pass
    except Exception as e:
        log.debug("score_candidates failed: %s", e)
    return record


def pick_winner(record: dict) -> str | None:
    """Return the winner name given mode."""
    if not record:
        return None
    mode = (record.get("mode") or _mode()).lower()
    if mode == "live" and record.get("winner"):
        return record["winner"]
    return record.get("llm_winner") or record.get("winner")


def note_feedback(
    *,
    action: str | None = None,
    feedback: str | None = None,
    taught: bool = False,
    user_id: str | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Record post-action feedback for plasticity / metrics."""
    out: dict[str, Any] = {
        "ts": time.time(),
        "action": action,
        "feedback": feedback,
        "taught": taught,
        "user_id": user_id,
    }
    if extra:
        out.update(extra)
    # Phase 12 metrics
    try:
        from cognition.fly_behavior import metrics as _fly_metrics
        _fly_metrics.record_feedback(out, user_id=user_id)
    except Exception:
        pass
    return out
