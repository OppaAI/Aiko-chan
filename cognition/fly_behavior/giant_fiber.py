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
from cognition.neural_state import _finite
from system.config import env_str, env_float

_URGENCY_RE = re.compile(
    r"\b(stop|abort|cancel|emergency|urgent|help|now|wait|halt|freeze)\b",
    re.I,
)
_SAFETY_RE = re.compile(
    r"\b(don'?t|never|unsafe|danger|harm|kill|delete everything)\b",
    re.I,
)

# A request ABOUT stopping something is not a request to stop Aiko. Keep
# attention/urgency cues separate from commands that cancel ongoing work.
_STOP_COMMAND_RE = re.compile(
    r"^\s*(?:aiko[,! ]+)?(?:please\s+)?(?:"
    r"(?:stop|abort|cancel|halt|freeze|wait)"
    r"(?:\s+(?:talking|speaking|generating|working|responding|that|it|"
    r"(?:this|the current)\s+(?:task|response|reply|plan|operation|generation)))?"
    r"(?:\s+(?:now|please|a moment|a second))?"
    r"|やめて|止まって|待って|中止して|그만|멈춰|잠깐)\s*[.!！。]*\s*$",
    re.I,
)


def cancel_threshold() -> float:
    return max(0.75, min(1.0, _finite(env_float("MEMORY_FLYGF_CANCEL_THRESHOLD", 0.85), 0.85)))


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
    motion_sudden: bool = False,
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
    stop_command = bool(_STOP_COMMAND_RE.fullmatch(t))
    if stop_command:
        sources["stop_command"] = 1.0
        u = 1.0
    if system_error:
        sources["system_error"] = 0.9
        u = max(u, 0.9)
    if _URGENCY_RE.search(t):
        sources["keyword"] = 0.7
        u = max(u, 0.7)
    if _SAFETY_RE.search(t):
        sources["safety"] = 0.55
        u = max(u, 0.55)
    pr = max(0.0, min(1.0, _finite(priority, 0.0)))
    if pr > 0:
        sources["priority"] = pr
        u = max(u, pr)
    su = max(0.0, min(1.0, _finite(subliminal_urgency, 0.0)))
    if su > 0:
        su_u = 0.35 + 0.35 * su
        sources["subliminal"] = round(su_u, 4)
        u = max(u, su_u)
    ms = max(0.0, min(1.0, _finite(motion_salience, 0.0)))
    if ms >= 0.55 or motion_sudden:
        ms_u = min(1.0, 0.35 + 0.25 * ms + (0.20 if motion_sudden else 0.0))
        sources["motion"] = round(ms_u, 4)
        u = max(u, ms_u)
    vu = max(0.0, min(1.0, _finite(voice_urgency, 0.0)))
    if vu > 0:
        # Loud voice is a nudge, never an interrupt by itself (capped at
        # 0.45 upstream — advisory rather than a cancellation command).
        sources["voice"] = round(vu, 4)
        u = max(u, vu)

    # Motion, loudness, semantic urgency and its decaying CX trace may guide
    # attention, but must not independently suppress replies or tool calls.
    interrupt = stop_command or max(0.9 if system_error else 0.0, pr) >= cancel_threshold()
    return {
        "mode": mode,
        "urgency": round(u, 4),
        "interrupt": interrupt and mode == "live",
        "would_interrupt": interrupt,
        "sources": sources,
        "cancel_threshold": cancel_threshold(),
    }


def should_abort_plan(user_id: str | None = None) -> bool:
    """True when live GF interrupt is set on NeuralState (agent loops call this)."""
    if _mode() != "live":
        return False
    try:
        from cognition.neural_state import peek_neural_state

        st = peek_neural_state(user_id)
        if st is None:
            return False
        return bool(st.interrupt)
    except Exception:
        return False
