"""Dopamine layer around MB plasticity (Stage 4, rate-based since Phase 10A).

Phase 10A: the one-shot pulse model is superseded by the rate-based credit
engine in :mod:`cognition.flymemory.credit` — dopamine as a continuous
modulatory signal with rise/decay dynamics, prediction-error style
(reward minus a running baseline), and decaying eligibility traces per
pattern. ``pulse()`` keeps its signature and result shape as a thin shim:
a pulse is now one reward *event* injected into the rate model
(mark the pattern's trace, bump dopamine by the prediction error, credit
all live traces backward).

Separates *reward* (outcome signal) from *dopamine* (PAM / PPL1 drive):

    reward ∈ [-1, 1]
      ├── PAM  (appetitive)  when reward > 0
      └── PPL1 (aversive)    when reward < 0
            × eligibility trace value
            → KC→MBON reinforce, scaled by the dopamine signal

Mode AIKO_FLY_DOPAMINE_MODE=off|shadow|live (defaults from MB mode) gates the
whole path; live application additionally requires MEMORY_FLYMB_MODE=live.
Never raises.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("aiko.flymemory.dopamine")


def _mb_mode() -> str:
    try:
        from system.config import env_str
        return env_str("MEMORY_FLYMB_MODE", "off").strip().lower()
    except Exception:
        return (os.getenv("MEMORY_FLYMB_MODE") or "off").strip().lower()


def split_channels(reward: float) -> dict:
    """Map scalar reward → PAM / PPL1 magnitudes in [0, 1]."""
    r = max(-1.0, min(1.0, float(reward)))
    return {
        "reward": r,
        "pam": max(0.0, r),
        "ppl1": max(0.0, -r),
        "signed": r,
    }


def pulse(
    reward: float,
    *,
    user_id: str | None = None,
    text: str | None = None,
    kc=None,
    weight: float = 1.0,
    source: str = "dopamine",
) -> dict:
    """Inject one reward event into the rate-based credit engine.

    weight: initial eligibility-trace value for this pattern in [0, 1]
      (legacy: direct drive multiplier).
    Returns diagnostic dict; safe no-op when dopamine/MB off or unavailable.
    Legacy keys preserved: mode, applied, delta, pam, ppl1, source, reason,
    drive. New keys: pe, da, baseline, n_traces, n_applied_traces, scope.
    Does NOT flush to SQLite — callers persist once per teaching event.
    """
    out: dict = {
        "mode": _mb_mode(),
        "applied": False,
        "delta": 0.0,
        "pam": 0.0,
        "ppl1": 0.0,
        "source": source,
    }
    try:
        from cognition.flymemory import credit as _credit

        if kc is None and not text:
            out["reason"] = "no_kc"
            return out
        replay = source in ("replay", "flyworld-replay")
        scope = "flyworld-sim" if source == "flyworld-replay" else "replay" if replay else "real"
        res = _credit.credit_event(
            user_id,
            reward,
            kc=kc,
            mark_value=weight,
            text=text,
            source=source,
            scope=scope,
            broadcast=not replay,
        )
    except Exception as exc:
        log.debug("dopamine pulse skipped: %s", exc)
        out["reason"] = str(exc)
        return out

    # Legacy-shaped result, enriched with the rate-model diagnostics.
    out.update({
        "mode": res.get("mode", "shadow"),
        "applied": bool(res.get("applied")),
        "delta": float(res.get("delta") or 0.0),
        "pam": float(res.get("pam") or 0.0),
        "ppl1": float(res.get("ppl1") or 0.0),
        "drive": float(res.get("drive") or 0.0),
        "reason": str(res.get("reason") or ""),
        "pe": float(res.get("pe") or 0.0),
        "da": float(res.get("da") or 0.0),
        "baseline": float(res.get("baseline") or 0.0),
        "n_traces": int(res.get("n_traces") or 0),
        "n_applied_traces": int(res.get("n_applied_traces") or 0),
        "scope": str(res.get("scope") or "real"),
        "reversal": bool(res.get("reversal")),
    })
    if "prior_bias" in res:
        out["prior_bias"] = res["prior_bias"]
    return out
