"""Giant-fiber style interrupt prior (functional abstraction).

MaleCNS giant-fiber / escape pathway inspires a *fast interrupt line* for
Aiko: high-urgency events can bias the agent to abort or deprioritize the
current plan. This is NOT a full GF circuit simulation.

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


def assess_interrupt(text: str, *, system_error: bool = False, priority: float = 0.0) -> dict:
    """Return urgency in [0,1] and whether interrupt should fire.

    Shadow: compute + log via caller. Live: caller should honor interrupt.
    """
    mode = _mode()
    if mode not in ("shadow", "live"):
        return {"mode": mode, "urgency": 0.0, "interrupt": False}
    t = text or ""
    u = 0.0
    if system_error:
        u = max(u, 0.85)
    if _URGENCY_RE.search(t):
        u = max(u, 0.7)
    if _SAFETY_RE.search(t):
        u = max(u, 0.55)
    u = max(u, max(0.0, min(1.0, float(priority or 0.0))))
    interrupt = u >= 0.65
    return {
        "mode": mode,
        "urgency": round(u, 4),
        "interrupt": interrupt and mode == "live",
        "would_interrupt": interrupt,
    }
