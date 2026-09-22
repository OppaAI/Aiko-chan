"""Per-turn fly behavioral priors (Stage 1 closed loops).

Called once at the start of a user turn. Publishes into NeuralState:
  GF interrupt (multi-source), LH familiarity, circadian, MB valence,
  DN vigor, optional SOUL bootstrap + online teach.

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
    prior_assistant: str | None = None,
    subliminal_urgency: float = 0.0,
) -> dict:
    """Run Stage-1 fly priors; return a compact summary for the turn.

    Never raises. Safe to call on every turn.
    """
    out: dict = {
        "interrupt": False,
        "urgency": 0.0,
        "familiarity": 0.5,
        "novel": False,
        "sleep_pressure": None,
        "valence": 0.0,
        "motor_vigor": 1.0,
        "tone_bits": [],
        "gf_sources": {},
        "soul_teach": None,
        "online_teach": None,
    }
    try:
        from cognition.neural_state import get_neural_state
        from cognition.fly_behavior.giant_fiber import assess_interrupt
        from cognition.fly_behavior.lateral_horn import context_prior

        st = get_neural_state(user_id)
        st.publish_circadian(_circadian_phase_now(), source="wallclock")

        try:
            from cognition.flymemory.soul_teach import ensure_soul_bootstrap
            out["soul_teach"] = ensure_soul_bootstrap(user_id)
        except Exception as exc:
            log.debug("soul_teach skipped: %s", exc)

        try:
            from cognition.fly_behavior.cx_topic import apply_topic_drive
            out["cx_topic"] = apply_topic_drive(text or "", user_id=user_id)
            from cognition.flymemory.online_teach import teach_from_user_text
            out["online_teach"] = teach_from_user_text(
                text or "", user_id=user_id, prior_assistant=prior_assistant
            )
        except Exception as exc:
            log.debug("online_teach skipped: %s", exc)

        try:
            from system.config import env_str
            mb_mode = env_str("MEMORY_FLYMB_MODE", "off").strip().lower()
        except Exception:
            mb_mode = "off"
        if mb_mode in ("shadow", "live"):
            try:
                from cognition.fly_registry import get_flymb
                from cognition.flymemory.circuit import text_features

                mb = get_flymb(user_id)
                if mb is not None:
                    bias = float(mb.valence_bias(text_features(text or "")))
                    out["valence"] = bias
                    if mb_mode == "live":
                        st.publish_mb(bias, source="turn")
                    st.record_influence(
                        {"kind": "mb_valence", "mode": mb_mode, "bias": round(bias, 4)}
                    )
            except Exception as exc:
                log.debug("mb valence skipped: %s", exc)

        motion = float(st.motion_salience or 0.0)
        gf = assess_interrupt(
            text or "",
            system_error=system_error,
            priority=priority,
            subliminal_urgency=float(subliminal_urgency or 0.0),
            motion_salience=motion,
        )
        st.publish_gf(float(gf.get("urgency") or 0.0), bool(gf.get("interrupt")), source="turn")
        out["urgency"] = gf.get("urgency", 0.0)
        out["interrupt"] = bool(gf.get("interrupt"))
        out["gf_sources"] = gf.get("sources") or {}
        if gf.get("mode") in ("shadow", "live"):
            log.debug(
                "flygf mode=%s urgency=%.2f interrupt=%s would=%s sources=%s",
                gf.get("mode"), gf.get("urgency"), gf.get("interrupt"),
                gf.get("would_interrupt"), gf.get("sources"),
            )
            st.record_influence(
                {
                    "kind": "gf",
                    "mode": gf.get("mode"),
                    "urgency": gf.get("urgency"),
                    "interrupt": gf.get("interrupt"),
                    "would_interrupt": gf.get("would_interrupt"),
                    "sources": gf.get("sources"),
                }
            )
            if gf.get("interrupt"):
                try:
                    from cognition.flymemory.online_teach import teach_interrupt_honored
                    teach_interrupt_honored(user_id, text or "stop abort cancel")
                except Exception:
                    pass

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
                st.record_influence(
                    {
                        "kind": "lh",
                        "mode": lh.get("mode"),
                        "familiarity": lh.get("familiarity"),
                        "novel": lh.get("novel"),
                        "bias": lh.get("bias"),
                    }
                )

        try:
            from system.config import env_str
            dn_mode = env_str("MEMORY_FLYDN_MODE", "off").strip().lower()
        except Exception:
            dn_mode = "off"
        if dn_mode in ("shadow", "live"):
            try:
                from cognition.flysense.dn import FlyDN

                energy = max(0.0, min(1.0, 1.0 - float(st.sleep_pressure or 0.0)))
                drv = FlyDN().drive(
                    energy=energy,
                    decisiveness=float(st.decisiveness or 0.5),
                    affect=float(st.valence or 0.0),
                )
                out["motor_vigor"] = drv.get("rate_mult", 1.0)
                if dn_mode == "live":
                    st.publish_dn(
                        arousal=float(drv.get("arousal") or 0.5),
                        rate_mult=float(drv.get("rate_mult") or 1.0),
                        source="turn",
                    )
                st.record_influence(
                    {
                        "kind": "dn",
                        "mode": dn_mode,
                        "arousal": drv.get("arousal"),
                        "rate_mult": drv.get("rate_mult"),
                        "vol_mult": drv.get("vol_mult"),
                    }
                )
            except Exception as exc:
                log.debug("dn drive skipped: %s", exc)

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
        log.debug("fly turn priors skipped: %s", exc)
    return out


def _maintenance_level_from_pressure(sleep_pressure: float) -> str:
    if sleep_pressure >= 0.75:
        return "maintenance"
    if sleep_pressure >= 0.45:
        return "reduced"
    return "normal"


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
        level = _maintenance_level_from_pressure(sp)
        if mode == "shadow":
            log.debug("flysleep mode=shadow sleep=%.2f would_level=%s", sp, level)
            return "normal"
        return level
    except Exception:
        return "normal"
