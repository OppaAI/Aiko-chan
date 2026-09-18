"""Per-turn fly behavioral priors: GF interrupt, LH context, circadian proxy.

Called once at the start of a user turn. Publishes into NeuralState.
Modes stay off/shadow until explicitly enabled in config.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger("aiko.fly_behavior.turn")


def _circadian_phase_now() -> float:
    """0..1 fraction of local day (wall-clock until a real clock circuit exists)."""
    try:
        now = datetime.now().astimezone()
    except Exception:
        now = datetime.now(timezone.utc)
    secs = now.hour * 3600 + now.minute * 60 + now.second
    return secs / 86400.0


def apply_turn_priors(
    text: str,
    *,
    user_id: str | None = None,
    system_error: bool = False,
    priority: float = 0.0,
) -> dict:
    """Run GF + LH + circadian; return a compact summary for the turn.

    Never raises. Safe to call on every turn.
    """
    out: dict = {
        "interrupt": False,
        "urgency": 0.0,
        "familiarity": 0.5,
        "novel": False,
        "sleep_pressure": None,
        "tone_bits": [],
    }
    try:
        from cognition.neural_state import get_neural_state
        from cognition.fly_behavior.giant_fiber import assess_interrupt
        from cognition.fly_behavior.lateral_horn import context_prior

        st = get_neural_state(user_id)
        st.publish_circadian(_circadian_phase_now(), source="wallclock")

        gf = assess_interrupt(text or "", system_error=system_error, priority=priority)
        st.publish_gf(float(gf.get("urgency") or 0.0), bool(gf.get("interrupt")), source="turn")
        out["urgency"] = gf.get("urgency", 0.0)
        out["interrupt"] = bool(gf.get("interrupt"))
        if gf.get("mode") in ("shadow", "live"):
            log.debug(
                "flygf mode=%s urgency=%.2f interrupt=%s would=%s",
                gf.get("mode"), gf.get("urgency"), gf.get("interrupt"), gf.get("would_interrupt"),
            )

        lh = context_prior(text or "", user_id=user_id)
        if "familiarity" in lh:
            st.publish_lh(float(lh["familiarity"]), source="turn")
            out["familiarity"] = lh.get("familiarity", 0.5)
            out["novel"] = bool(lh.get("novel"))
            if lh.get("mode") in ("shadow", "live"):
                log.debug(
                    "flylh mode=%s familiarity=%.2f novel=%s bias=%s",
                    lh.get("mode"), lh.get("familiarity"), lh.get("novel"), lh.get("bias"),
                )

        out["sleep_pressure"] = st.sleep_pressure

        bits: list[str] = []
        if st.interrupt or st.urgency >= 0.65:
            bits.append("user signaled urgency — answer briefly, acknowledge stop/wait")
        if st.avoidance > 0.3:
            bits.append("slightly cautious tone")
        if st.approach > 0.3 and st.context_familiarity > 0.6:
            bits.append("warm, familiar tone")
        if st.sleep_pressure > 0.6:
            bits.append("keep the reply shorter; low energy")
        phase = st.circadian_phase
        if phase < 0.25 or phase > 0.9:
            bits.append("late/early hours — softer volume of commitment")
        out["tone_bits"] = bits
    except Exception as exc:
        log.debug("fly turn priors skipped: %s", exp if False else exc)
    return out


def maintenance_level(user_id: str | None = None) -> str:
    """Map sleep_pressure → normal | reduced | maintenance.

    Used by agent cadence / background work. Does not itself run dreams.
    """
    try:
        from cognition.neural_state import get_neural_state
        from system.config import env_str
        mode = env_str("MEMORY_FLYSLEEP_MODE", "off").strip().lower()
        if mode not in ("shadow", "live"):
            return "normal"
        sp = float(get_neural_state(user_id).sleep_pressure or 0.0)
        if sp >= 0.75:
            level = "maintenance"
        elif sp >= 0.45:
            level = "reduced"
        else:
            level = "normal"
        if mode == "shadow":
            log.debug("flysleep mode=shadow sleep=%.2f would_level=%s", sp, level)
            return "normal"
        return level
    except Exception:
        return "normal"
