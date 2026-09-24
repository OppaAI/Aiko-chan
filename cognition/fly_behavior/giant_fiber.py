"""Giant-fiber style interrupt prior (Stage 1 multi-source).

MaleCNS giant-fiber / escape pathway inspires a fast interrupt line for
Aiko: high-urgency events can bias the agent to abort or deprioritize the
current plan. This is NOT a full GF circuit simulation.

Stage 1: urgency is the max of keyword, safety, system_error, priority,
subliminal urgency cue, and motion salience — not regex alone.

Modes (MEMORY_FLYGF_MODE): off | shadow | live
"""
from __future__ import annotations

import re
from system.config import env_str

_URGENCY_RE = re.compile(
    r"\b(stop|abort|cancel|emergency|urgent|help|now|wait|halt|freeze)\b",
    re.I,
)
_SAFETY_RE = re.compile(
    r"\b(don'?t|never|unsafe|danger|harm|kill|delete everything)\b",
    re.I,
)


def _mode() -> str:
    try:
        return env_str("MEMORY_FLYGF_MODE", "off").strip().lower()
    except Exception:
        return "off"


def assess_interrupt(
    text: str,
    *,
    system_error: bool = False,
    priority: float = 0.0,
    subliminal_urgency: float = 0.0,
    motion_salience: float = 0.0,
    voice_urgency: float = 0.0,
) -> dict:
    """Return urgency in [0,1] and whether interrupt should fire.

    Shadow: compute + log via caller. Live: caller should honor interrupt.

    Phase 6: voice_urgency carries the loudness→urgency vote from the
    sensory pathways (0 when no voice prosody was available).
    """
    mode = _mode()
    if mode not in ("shadow", "live"):
        return {
            "mode": mode,
            "urgency": 0.0,
            "interrupt": False,
            "would_interrupt": False,
            "sources": {},
        }
    t = text or ""
    sources: dict[str, float] = {}
    u = 0.0
    if system_error:
        sources["system_error"] = 0.85
        u = max(u, 0.85)
    if _URGENCY_RE.search(t):
        sources["keyword"] = 0.7
        u = max(u, 0.7)
    if _SAFETY_RE.search(t):
        sources["safety"] = 0.55
        u = max(u, 0.55)
    pr = max(0.0, min(1.0, float(priority or 0.0)))
    if pr > 0:
        sources["priority"] = pr
        u = max(u, pr)
    su = max(0.0, min(1.0, float(subliminal_urgency or 0.0)))
    if su > 0:
        su_u = 0.35 + 0.35 * su
        sources["subliminal"] = round(su_u, 4)
        u = max(u, su_u)
    ms = max(0.0, min(1.0, float(motion_salience or 0.0)))
    if ms >= 0.55:
        ms_u = 0.35 + 0.25 * ms
        sources["motion"] = round(ms_u, 4)
        u = max(u, ms_u)
    vu = max(0.0, min(1.0, float(voice_urgency or 0.0)))
    if vu > 0:
        # Loud voice is a nudge, never an interrupt by itself (capped at
        # 0.45 upstream — below the 0.65 interrupt line).
        sources["voice"] = round(vu, 4)
        u = max(u, vu)

    interrupt = u >= 0.65
    return {
        "mode": mode,
        "urgency": round(u, 4),
        "interrupt": interrupt and mode == "live",
        "would_interrupt": interrupt,
        "sources": sources,
    }


def should_abort_plan(user_id: str | None = None) -> bool:
    """True when live GF interrupt is set on NeuralState (agent loops call this)."""
    if _mode() != "live":
        return False
    try:
        from cognition.neural_state import get_neural_state

        st = get_neural_state(user_id)
        return bool(st.interrupt) or float(st.urgency or 0.0) >= 0.65
    except Exception:
        return False
