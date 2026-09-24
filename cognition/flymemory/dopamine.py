"""Dopamine layer around MB plasticity (Stage 4).

Separates *reward* (outcome signal) from *dopamine* (PAM / PPL1 drive):

    reward ∈ [-1, 1]
      ├── PAM  (appetitive)  when reward > 0
      └── PPL1 (aversive)    when reward < 0
            × eligibility weight
            → KC→MBON reinforce

Connectome MBON signs already come from PAM vs PPL1 innervation; this
module only times and scales the teaching pulse. Never raises.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger("aiko.flymemory.dopamine")


def _mb_mode() -> str:
    try:
        from system.config import env_str
        return env_str("MEMORY_FLYMB_MODE", "off").strip().lower()
    except Exception:
        return (os.getenv("MEMORY_FLYMB_MODE") or "off").strip().lower()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _bool_env(name: str, default: bool) -> bool:
    try:
        raw = os.getenv(name, "1" if default else "0").strip().lower()
        return raw not in ("0", "false", "no", "off")
    except Exception:
        return default


# ── duplicate-pulse debounce (Phase 2 "stop the waste") ──────────────────
# Near-identical consecutive teaching signals ("thanks" / "thank you") each
# used to trigger a full reinforce + SQLite table rewrite. KC patterns are
# top-k sparse over 4,064 dims at ~5% sparsity: identical text → cosine 1.0,
# paraphrases → >0.9, different topics → ~0.05. So cosine ≥ 0.9 cleanly
# separates "said the same thing again" from new content.
# Records only the last APPLIED pulse per user; skipped duplicates still flow
# through the eligibility trail upstream (callers record eligibility when
# applied=False).
_LAST_PULSE: dict[str, tuple[np.ndarray, float, float]] = {}
_LAST_PULSE_LOCK = threading.RLock()
_PULSE_LOCKS: dict[str, threading.RLock] = {}


def _pulse_lock(user_id: str | None) -> threading.RLock:
    with _LAST_PULSE_LOCK:
        return _PULSE_LOCKS.setdefault(user_id or "guest", threading.RLock())


def _is_duplicate_pulse(user_id: str | None, kc, drive: float) -> bool:
    if not _bool_env("FLY_DOPAMINE_DEDUP", True):
        return False
    cos_thr = _float_env("FLY_DOPAMINE_DEDUP_COS", 0.9)
    tol = _float_env("FLY_DOPAMINE_DEDUP_REWARD_TOL", 0.15)
    try:
        kc = np.asarray(kc, dtype=np.float64).reshape(-1)
    except Exception:
        return False
    n = float(np.linalg.norm(kc))
    if n <= 0:
        return False
    with _LAST_PULSE_LOCK:
        last = _LAST_PULSE.get(user_id or "guest")
    if last is None:
        return False
    last_kc, last_drive, last_at = last
    if time.monotonic() - last_at > max(0.0, _float_env("FLY_DOPAMINE_DEDUP_SECONDS", 2.0)):
        return False
    if float(np.sign(drive)) != float(np.sign(last_drive)):
        return False
    if abs(drive - last_drive) > tol:
        return False
    denom = n * float(np.linalg.norm(last_kc))
    if denom <= 0:
        return False
    return float(np.dot(kc, last_kc) / denom) >= cos_thr


def _record_applied_pulse(user_id: str | None, kc, drive: float) -> None:
    try:
        kc = np.asarray(kc, dtype=np.float64).reshape(-1).copy()
    except Exception:
        return
    with _LAST_PULSE_LOCK:
        _LAST_PULSE[user_id or "guest"] = (kc, drive, time.monotonic())


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
    """Apply one dopamine-timed reinforce pulse to the MB.

    weight: eligibility / decay multiplier in [0, 1].
    Near-duplicate pulses (cosine-similar KC, same reward sign, close drive)
    are debounced: returns applied=False, reason="dedup" with no write.
    Does NOT flush to SQLite — callers persist once per teaching event.
    Returns diagnostic dict; safe no-op when MB off/unavailable.
    """
    out: dict = {
        "mode": _mb_mode(),
        "applied": False,
        "delta": 0.0,
        "pam": 0.0,
        "ppl1": 0.0,
        "source": source,
    }
    mode = out["mode"]
    if mode not in ("shadow", "live"):
        out["reason"] = "mode_off"
        return out
    ch = split_channels(reward)
    out["pam"] = ch["pam"]
    out["ppl1"] = ch["ppl1"]
    w = max(0.0, min(1.0, float(weight)))
    signed = ch["signed"] * w
    if abs(signed) < 1e-9:
        out["reason"] = "zero_drive"
        return out
    try:
        from cognition.fly_registry import get_flymb
        from cognition.flymemory.circuit import text_features

        mb = get_flymb(user_id)
        if mb is None:
            out["reason"] = "mb_unavailable"
            return out
        if kc is None:
            if not text:
                out["reason"] = "no_kc"
                return out
            kc = mb.encode(text_features(text))

        reversal_mult = 1.0
        if text:
            try:
                bias = float(mb.valence_bias(text_features(text)))
                out["prior_bias"] = round(bias, 4)
                if bias * signed < -0.05:
                    reversal_mult = max(1.0, _float_env("FLY_REVERSAL_MULT", 1.6))
                    out["reversal"] = True
            except Exception:
                pass

        drive = signed * reversal_mult
        out["drive"] = round(drive, 4)
        if mode == "live":
            # Debounce: a near-identical pulse to the last applied one teaches
            # nothing new — skip the reinforce AND the table rewrite. The
            # eligibility trail upstream stays intact (callers record it when
            # applied=False).
            with _pulse_lock(user_id):
                if _is_duplicate_pulse(user_id, kc, drive):
                    out["applied"] = False
                    out["reason"] = "dedup"
                    return out
                delta = float(mb.reinforce(kc, drive) or 0.0)
                out["delta"] = round(delta, 4)
                out["applied"] = abs(delta) > 0.0
                if out["applied"]:
                    _record_applied_pulse(user_id, kc, drive)
            # NOTE: no flush_mb here — the teaching event persists once:
            # online_teach()/teach_interrupt_honored() delegate to
            # eligibility.assign_credit(), which flushes once at the end of
            # the trail, covering the direct pulse too. (Before Phase 2,
            # pulse() flushed internally AND callers flushed, causing ~10
            # full DELETE + re-INSERT cycles of mb_plastic per event.)
            out["reason"] = "ok"
        else:
            out["reason"] = "shadow"
        try:
            from cognition.neural_state import get_neural_state
            get_neural_state(user_id).record_influence(
                {
                    "kind": "dopamine",
                    "mode": mode,
                    "source": source,
                    "pam": round(ch["pam"] * w, 4),
                    "ppl1": round(ch["ppl1"] * w, 4),
                    "drive": out.get("drive", 0.0),
                    "delta": out["delta"],
                    "reversal": bool(out.get("reversal")),
                    "preview": (text or "")[:40],
                }
            )
        except Exception:
            pass
    except Exception as exc:
        log.debug("dopamine pulse skipped: %s", exc)
        out["reason"] = str(exc)
    return out
