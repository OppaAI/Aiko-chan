"""Global giant-fiber cancel (Stage 6).

Extends should_abort_plan so TTS, tools, scheduler, and body output can all
honor the same interrupt flag without each path re-implementing thresholds.
"""
from __future__ import annotations

import logging

log = logging.getLogger("aiko.fly.gf_global")


def should_cancel_output(user_id: str | None = None) -> bool:
    """True when live GF says stop speech / motion / non-critical work."""
    try:
        from cognition.fly_behavior.giant_fiber import should_abort_plan
        return bool(should_abort_plan(user_id))
    except Exception:
        return False


def should_cancel_tts(user_id: str | None = None) -> bool:
    return should_cancel_output(user_id)


def should_cancel_tools(user_id: str | None = None) -> bool:
    return should_cancel_output(user_id)


def should_cancel_scheduler(user_id: str | None = None) -> bool:
    """Background posts / autonomous jobs should yield to interrupt."""
    return should_cancel_output(user_id)


def clear_interrupt(user_id: str | None = None) -> bool:
    """Clear NeuralState interrupt after the user regains control."""
    try:
        from cognition.neural_state import get_neural_state
        st = get_neural_state(user_id)
        st.publish_gf(0.0, False, source="cleared")
        st.record_influence({"kind": "gf_clear", "interrupt": False})
        # Phase 7: drop the decaying urgency trace too, or it would
        # re-warm urgency after the user cleared the interrupt.
        try:
            from cognition.centralcomplex.temporal import reset_urgency
            reset_urgency(user_id)
        except Exception as exp:
            log.debug("clear_interrupt trace reset skipped: %s", exp)
        return True
    except Exception as exp:
        log.debug("clear_interrupt skipped: %s", exp)
        return False
