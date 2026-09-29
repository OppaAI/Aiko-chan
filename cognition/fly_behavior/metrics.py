"""Phase 12 — fly metrics + safe dial aggregates.

Lightweight in-process counters for action-selection trails and fullbrain
steps. Never raises. Safe to call from any hot path.

Studio: GET /studio/fly/api/metrics
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from typing import Any

_lock = threading.RLock()
_global: dict[str, float | int] = {
    "action_scores": 0,
    "action_applied": 0,       # live mode actually changed winner
    "action_agree_llm": 0,     # fly winner == LLM prior winner
    "action_disagree_llm": 0,
    "action_ambiguous": 0,
    "action_veto": 0,
    "feedback_praise": 0,
    "feedback_correction": 0,
    "feedback_taught": 0,
    "fullbrain_steps": 0,
    "fullbrain_skipped_throttle": 0,
    "fullbrain_ms_total": 0.0,
}
_per_user: dict[str, dict[str, float | int]] = {}
_started = time.time()


def _bucket(user_id: str | None) -> dict[str, float | int] | None:
    if not user_id:
        return None
    b = _per_user.get(user_id)
    if b is None:
        b = {k: 0 if not str(k).endswith("_ms_total") else 0.0 for k in _global}
        _per_user[user_id] = b
    return b


def _inc(key: str, user_id: str | None = None, n: float | int = 1) -> None:
    with _lock:
        _global[key] = _global.get(key, 0) + n  # type: ignore[operator]
        b = _bucket(user_id)
        if b is not None:
            b[key] = b.get(key, 0) + n  # type: ignore[operator]


def record_action(record: dict, user_id: str | None = None) -> None:
    """Record one score_candidates result. Never raises."""
    try:
        if not isinstance(record, dict):
            return
        _inc("action_scores", user_id)
        if record.get("applied"):
            _inc("action_applied", user_id)
        if record.get("ambiguous"):
            _inc("action_ambiguous", user_id)
        cands = record.get("candidates") or {}
        if not isinstance(cands, dict) or not cands:
            return
        # LLM prior winner vs fly-final ranking winner (before mode gate)
        prior_win = max(cands.items(), key=lambda kv: float(kv[1].get("llm_prior", 0)))[0]
        fly_win = max(cands.items(), key=lambda kv: float(kv[1].get("final", 0)))[0]
        if prior_win == fly_win:
            _inc("action_agree_llm", user_id)
        else:
            _inc("action_disagree_llm", user_id)
        if any(bool(v.get("veto")) for v in cands.values() if isinstance(v, dict)):
            _inc("action_veto", user_id)
    except Exception:
        pass


def record_feedback(out: dict, user_id: str | None = None) -> None:
    try:
        fb = (out or {}).get("feedback")
        if fb == "praise":
            _inc("feedback_praise", user_id)
        elif fb == "correction":
            _inc("feedback_correction", user_id)
        if (out or {}).get("taught"):
            _inc("feedback_taught", user_id)
    except Exception:
        pass


def record_fullbrain(*, ran: bool, ms: float = 0.0, user_id: str | None = None) -> None:
    try:
        if ran:
            _inc("fullbrain_steps", user_id)
            if ms > 0:
                _inc("fullbrain_ms_total", user_id, float(ms))
        else:
            _inc("fullbrain_skipped_throttle", user_id)
    except Exception:
        pass


def snapshot(user_id: str | None = None) -> dict[str, Any]:
    """Return aggregate metrics (process-wide + optional per-user)."""
    with _lock:
        g = dict(_global)
        u = dict(_per_user.get(user_id, {})) if user_id else {}
    scores = int(g.get("action_scores") or 0)
    applied = int(g.get("action_applied") or 0)
    agree = int(g.get("action_agree_llm") or 0)
    disagree = int(g.get("action_disagree_llm") or 0)
    fb_steps = int(g.get("fullbrain_steps") or 0)
    fb_ms = float(g.get("fullbrain_ms_total") or 0.0)
    rates = {
        "applied_rate": round(applied / scores, 4) if scores else 0.0,
        "agree_rate": round(agree / max(1, agree + disagree), 4),
        "fullbrain_avg_ms": round(fb_ms / fb_steps, 1) if fb_steps else 0.0,
    }
    modes = {}
    try:
        from system.config import env_str

        for k, default in (
            ("AIKO_FLY_ACTION_MODE", "shadow"),
            ("AIKO_FLY_ACTION_WEIGHT", "0.25"),
            ("AIKO_FULLBRAIN_EVERY_N", "1"),
            ("AIKO_FULLBRAIN_MIN_INTERVAL_S", "0"),
            ("MEMORY_FLYMB_MODE", "off"),
            ("MEMORY_FLYCX_MODE", "off"),
            ("MEMORY_FLYGF_MODE", "off"),
            ("MEMORY_FLYDN_MODE", "off"),
        ):
            modes[k] = env_str(k, default)
    except Exception:
        modes = {
            "AIKO_FLY_ACTION_MODE": os.getenv("AIKO_FLY_ACTION_MODE", "shadow"),
            "AIKO_FLY_ACTION_WEIGHT": os.getenv("AIKO_FLY_ACTION_WEIGHT", "0.25"),
            "AIKO_FULLBRAIN_EVERY_N": os.getenv("AIKO_FULLBRAIN_EVERY_N", "1"),
            "AIKO_FULLBRAIN_MIN_INTERVAL_S": os.getenv("AIKO_FULLBRAIN_MIN_INTERVAL_S", "0"),
        }
    return {
        "stage": "12",
        "uptime_s": round(time.time() - _started, 1),
        "process": g,
        "user": u,
        "rates": rates,
        "dial": modes,
        "notes": (
            "applied_rate: fraction of scored turns where live fly changed the winner; "
            "agree_rate: fly final rank matches LLM prior rank; "
            "throttle: AIKO_FULLBRAIN_EVERY_N / AIKO_FULLBRAIN_MIN_INTERVAL_S."
        ),
    }
