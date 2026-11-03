"""Phase 11 — persona-conditioned fly circuits.

The persona changes how the fly's circuits respond to situations; it
never tells the fly what to do. Traits modulate circuit GAINS (CX
gain/decay/thresholds, MB plasticity, GF urgency sensitivity, DN vigor),
never action scores. There is no code path from a trait to a vote.

Submodules:
  state       NeuralPersonalityState — per-identity persistent traits,
              JSON on disk, resettable, inspectable. Separate from
              persona/SOUL.md, which this package never modifies.
  modulators  trait → gain mapping; persona_gains() computes,
              applied_gains() gates on AIKO_FLY_PERSONA_MODE.
  plasticity  the ONLY trait writer, via Phase-10A dopamine events.
  trace       explainability record (future Studio panel data).

Mode AIKO_FLY_PERSONA_MODE=off|shadow|live (default shadow):
  off    — no persona work at all.
  shadow — gains computed + logged, never applied.
  live   — gains applied; trait plasticity active (scope="real" only).

Conscience vetoes remain absolute: persona touches circuit gains
upstream of voting and can never clear or weaken a veto flag.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("aiko.fly.persona")


def persona_mode() -> str:
    """AIKO_FLY_PERSONA_MODE: off | shadow | live (default shadow)."""
    try:
        m = (os.getenv("AIKO_FLY_PERSONA_MODE", "shadow") or "shadow").strip().lower()
    except Exception:
        m = "shadow"
    return m if m in ("off", "shadow", "live") else "shadow"


from cognition.fly_persona.state import (  # noqa: E402
    TRAITS,
    NeuralPersonalityState,
    clear_personality,
    get_personality,
    persona_defaults,
)
from cognition.fly_persona.modulators import (  # noqa: E402
    applied_gains,
    persona_gains,
)
from cognition.fly_persona.plasticity import on_credit_outcome  # noqa: E402
from cognition.fly_persona.trace import (  # noqa: E402
    explain_turn,
    record_persona_trace,
)


def reset_personality(user_id: str | None = None) -> dict:
    """Restore an identity's traits to persona defaults."""
    try:
        return get_personality(user_id).reset()
    except Exception as exc:
        log.debug("reset_personality skipped: %s", exc)
        return {"error": str(exc)[:120]}


__all__ = [
    "TRAITS",
    "NeuralPersonalityState",
    "applied_gains",
    "clear_personality",
    "explain_turn",
    "get_personality",
    "on_credit_outcome",
    "persona_defaults",
    "persona_gains",
    "persona_mode",
    "record_persona_trace",
    "reset_personality",
]
