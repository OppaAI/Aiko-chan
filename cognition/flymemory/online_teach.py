"""Online DAN-like teaching from live conversation outcomes (Stage 1).

Maps user feedback / stop-honored / sticky-topic rejection into reinforce().
Never raises to callers. Safe on every turn.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("aiko.flymemory.online_teach")

_PRAISE_RE = re.compile(
    r"\b(thanks|thank you|perfect|great job|love that|nice|well done|good girl)\b",
    re.I,
)
_CORRECT_RE = re.compile(
    r"\b(no[,.]?\s+that'?s wrong|incorrect|stop saying|don'?t (say|talk about)|not true)\b",
    re.I,
)
_STOP_TOPIC_RE = re.compile(
    r"\b(stop (talking about|bringing up)|enough about|quit mentioning)\b",
    re.I,
)


def _mode() -> str:
    try:
        from system.config import env_str
        return env_str("MEMORY_FLYMB_MODE", "off").strip().lower()
    except Exception:
        return "off"


def _record_eligibility(out: dict, user_id: str | None, text: str) -> None:
    if out.get("eligibility_recorded", False):
        return
    try:
        from cognition.flymemory.eligibility import record_step

        out["eligibility_recorded"] = record_step(user_id, text)
    except Exception:
        pass


def teach_from_user_text(
    text: str,
    *,
    user_id: str | None = None,
    prior_assistant: str | None = None,
) -> dict:
    """Infer a small reward from the user's message and reinforce MB if live.

    Returns {mode, reward, taught, reason, eligibility_recorded}.
    """
    mode = _mode()
    out = {
        "mode": mode,
        "reward": 0.0,
        "taught": False,
        "reason": "",
        "eligibility_recorded": False,
    }
    if mode not in ("shadow", "live"):
        out["reason"] = "mode_off"
        return out
    t = text or ""
    reward = 0.0
    reason = ""
    try:
        from cognition.flymemory.teach_api import _TOPIC_FROM_AVOID, extract_topic, teach_preference
        topic = extract_topic(t)
        if topic and (_STOP_TOPIC_RE.search(t) or _TOPIC_FROM_AVOID.search(t)):
            pref = teach_preference(topic, direction="avoid", user_id=user_id)
            out.update({"reward": pref.get("reward", -0.55), "reason": "avoid_topic", "taught": pref.get("taught", False), "topic": topic})
            if pref.get("mode") in ("shadow", "live"):
                if mode == "live" and pref.get("taught"):
                    try:
                        from cognition.flymemory.eligibility import assign_credit
                        assign_credit(user_id, pref.get("reward", -0.55))
                        _record_eligibility(out, user_id, t)
                    except Exception:
                        pass
                return out
        if topic and re.search(r"\b(prefer|always|do more of|i like when)\b", t, re.I):
            pref = teach_preference(topic, direction="prefer", user_id=user_id)
            out.update({"reward": pref.get("reward", 0.5), "reason": "prefer_topic", "taught": pref.get("taught", False), "topic": topic})
            if pref.get("mode") in ("shadow", "live"):
                if mode == "live" and pref.get("taught"):
                    try:
                        from cognition.flymemory.eligibility import assign_credit
                        assign_credit(user_id, pref.get("reward", 0.5))
                        _record_eligibility(out, user_id, t)
                    except Exception:
                        pass
                return out
    except Exception as exc:
        log.debug("teach_api path skipped: %s", exc)

    if _STOP_TOPIC_RE.search(t):
        reward = -0.55
        reason = "stop_topic"
        teach_text = prior_assistant or t
    elif _CORRECT_RE.search(t):
        reward = -0.45
        reason = "correction"
        teach_text = prior_assistant or t
    elif _PRAISE_RE.search(t):
        reward = 0.40
        reason = "praise"
        teach_text = prior_assistant or t
    else:
        out["reason"] = "no_signal"
        if mode == "live":
            _record_eligibility(out, user_id, t)
        return out

    out["reward"] = reward
    out["reason"] = reason
    if mode == "shadow":
        log.debug("online_teach shadow reason=%s reward=%+.2f", reason, reward)
        return out
    mb = None
    direct_applied = False
    direct_flushed = False
    try:
        from cognition.fly_registry import get_flymb, get_fly_store
        from cognition.flymemory.circuit import text_features
        from cognition.neural_state import get_neural_state

        mb = get_flymb(user_id)
        if mb is None:
            out["reason"] = "mb_unavailable"
            return out
        kc = mb.encode(text_features(teach_text))
        try:
            # Phase 10A: one rate-based credit event via the dopamine shim —
            # mark this turn's trace, bump dopamine by the prediction error,
            # credit all live traces backward. Replaces the legacy
            # pulse-then-assign_credit pair (which double-bumped dopamine).
            from cognition.flymemory.dopamine import pulse as _da_pulse
            _cr = _da_pulse(
                reward, user_id=user_id, kc=kc, text=teach_text,
                source="online_teach")
            if _cr.get("reason") != "ok" or not _cr.get("applied"):
                _record_eligibility(out, user_id, teach_text)
                return out
            direct_applied = True
            delta = float(_cr.get("delta") or 0.0)
        except Exception:
            delta = float(mb.reinforce(kc, reward) or 0.0)
            if not delta:
                _record_eligibility(out, user_id, teach_text)
                return out
            direct_applied = True
        bias = mb.valence_bias(text_features(teach_text))
        st = get_neural_state(user_id)
        st.publish_mb(bias, source=f"online:{reason}")
        try:
            st.record_influence(
                {
                    "kind": "online_teach",
                    "reason": reason,
                    "reward": reward,
                    "delta": round(delta, 4),
                    "bias": round(bias, 4),
                }
            )
        except Exception:
            pass
        out["taught"] = True
        out["delta"] = round(delta, 4)
        try:
            _record_eligibility(out, user_id, teach_text)
            out["credit_steps"] = int(_cr.get("n_applied_traces", 0))
        except Exception:
            pass
        log.debug("online_teach live reason=%s reward=%+.2f delta=%.4f", reason, reward, delta)
    except Exception as exc:
        log.debug("online_teach failed: %s", exc)
        out["reason"] = str(exc)
    finally:
        if direct_applied and not direct_flushed:
            try:
                store = get_fly_store(user_id)
                if store is not None:
                    store.flush_mb(mb)
            except Exception:
                log.debug("online_teach direct pulse flush failed", exc_info=True)
    return out


def teach_interrupt_honored(
    user_id: str | None = None,
    text: str = "user stop abort cancel",
    *,
    eligibility_recorded: bool = False,
) -> dict:
    """Small positive teaching when GF interrupt was honored."""
    mode = _mode()
    out = {"mode": mode, "taught": False, "eligibility_recorded": eligibility_recorded}
    if mode != "live":
        return out
    mb = None
    direct_applied = False
    direct_flushed = False
    try:
        from cognition.fly_registry import get_flymb, get_fly_store
        from cognition.flymemory.circuit import text_features

        mb = get_flymb(user_id)
        if mb is None:
            return out
        try:
            # Phase 10A: single rate-based credit event via the dopamine
            # shim (mark + dopamine + backward credit), replacing
            # pulse-then-assign_credit.
            from cognition.flymemory.dopamine import pulse as _da_pulse
            _cr = _da_pulse(
                0.35, user_id=user_id, text=text, source="interrupt_honored")
            if _cr.get("reason") != "ok" or not _cr.get("applied"):
                _record_eligibility(out, user_id, text)
                return out
            direct_applied = True
        except Exception:
            delta = float(mb.reinforce(mb.encode(text_features(text)), 0.35) or 0.0)
            if not delta:
                _record_eligibility(out, user_id, text)
                return out
            direct_applied = True
        try:
            _record_eligibility(out, user_id, text)
        except Exception:
            pass
        out.update({"taught": True, "reward": 0.35})
        return out
    except Exception as exc:
        log.debug("teach_interrupt_honored failed: %s", exc)
        out["error"] = str(exc)
        return out
    finally:
        if direct_applied and not direct_flushed:
            try:
                store = get_fly_store(user_id)
                if store is not None:
                    store.flush_mb(mb)
            except Exception:
                log.debug("interrupt direct pulse flush failed", exc_info=True)
