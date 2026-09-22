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
            delta = float(mb.reinforce(kc, drive) or 0.0)
            out["delta"] = round(delta, 4)
            out["applied"] = abs(delta) > 0.0
            try:
                from cognition.fly_registry import get_fly_store
                store = get_fly_store(user_id)
                if store is not None:
                    store.flush_mb(mb)
            except Exception:
                pass
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
