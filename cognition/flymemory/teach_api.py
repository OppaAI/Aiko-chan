"""Explicit prefer / avoid teaching (Stage 2).

Gives operators a clear mechanism to teach Aiko to approach or avoid a
topic without relying only on free-form praise/stop cues.

  teach_preference("fruit tarts", direction="avoid", user_id=...)
  teach_preference("short answers", direction="prefer", user_id=...)

Effects:
  * MB reinforce on the topic text (associative bias)
  * NeuralState valence publish + influence event
  * Optional durable preference fact via memory write hook (best-effort)

Never raises to callers.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("aiko.flymemory.teach_api")

_TOPIC_FROM_STOP = re.compile(
    r"(?:stop (?:talking about|bringing up)|enough about|quit mentioning)\s+(.+?)(?:[.!?]|$)",
    re.I,
)
_TOPIC_FROM_PREFER = re.compile(
    r"(?:prefer|always|do more of|i like when you)\s+(.+?)(?:[.!?]|$)",
    re.I,
)
_TOPIC_FROM_AVOID = re.compile(
    r"(?:avoid|never (?:talk about|bring up)|don'?t (?:talk about|bring up))\s+(.+?)(?:[.!?]|$)",
    re.I,
)


def _mb_mode() -> str:
    try:
        from system.config import env_str
        return env_str("MEMORY_FLYMB_MODE", "off").strip().lower()
    except Exception:
        return "off"


def extract_topic(text: str) -> str | None:
    """Best-effort topic phrase from a user teaching utterance."""
    t = (text or "").strip()
    if not t:
        return None
    for rx in (_TOPIC_FROM_STOP, _TOPIC_FROM_AVOID, _TOPIC_FROM_PREFER):
        m = rx.search(t)
        if m:
            topic = m.group(1).strip(" \t\"'`.,;:")
            if 2 <= len(topic) <= 80:
                return topic
    return None


def teach_preference(
    topic: str,
    *,
    direction: str = "avoid",
    user_id: str | None = None,
    strength: float | None = None,
    write_memory_fact: bool = True,
) -> dict:
    """Teach approach (prefer) or avoidance for a topic string.

    direction: prefer | approach | avoid | suppress
    strength: optional |reward| in (0, 1]; defaults 0.5 prefer / 0.55 avoid.
    """
    mode = _mb_mode()
    topic = (topic or "").strip()
    out: dict = {
        "mode": mode,
        "topic": topic,
        "direction": direction,
        "taught": False,
        "reward": 0.0,
        "reason": "",
    }
    if not topic:
        out["reason"] = "empty_topic"
        return out
    if mode not in ("shadow", "live"):
        out["reason"] = "mode_off"
        return out

    d = (direction or "avoid").strip().lower()
    if d in ("prefer", "approach", "like", "want"):
        sign = 1.0
        d = "prefer"
        default_mag = 0.50
    else:
        sign = -1.0
        d = "avoid"
        default_mag = 0.55
    mag = float(strength) if strength is not None else default_mag
    mag = max(0.05, min(1.0, abs(mag)))
    reward = sign * mag
    out["reward"] = reward
    out["direction"] = d

    if mode == "shadow":
        out["reason"] = "shadow"
        log.debug("teach_preference shadow topic=%r dir=%s reward=%+.2f", topic, d, reward)
        return out

    try:
        from cognition.fly_registry import get_flymb, get_fly_store
        from cognition.flymemory.circuit import text_features
        from cognition.neural_state import get_neural_state

        mb = get_flymb(user_id)
        if mb is None:
            out["reason"] = "mb_unavailable"
            return out
        feats = text_features(topic)
        kc = mb.encode(feats)
        delta = float(mb.reinforce(kc, reward) or 0.0)
        bias = float(mb.valence_bias(feats))
        st = get_neural_state(user_id)
        st.publish_mb(bias, source=f"teach:{d}")
        st.record_influence(
            {
                "kind": "teach_preference",
                "topic": topic[:80],
                "direction": d,
                "reward": round(reward, 4),
                "delta": round(delta, 4),
                "bias": round(bias, 4),
            }
        )
        store = get_fly_store(user_id)
        if store is not None:
            try:
                store.flush_mb(mb)
            except Exception:
                pass
        out["taught"] = True
        out["delta"] = round(delta, 4)
        out["bias"] = round(bias, 4)
        out["reason"] = "ok"

        if write_memory_fact:
            try:
                _write_preference_fact(topic, d, user_id=user_id)
            except Exception as exc:
                log.debug("preference fact write skipped: %s", exc)
    except Exception as exc:
        log.debug("teach_preference failed: %s", exc)
        out["reason"] = str(exc)
    return out


def _write_preference_fact(topic: str, direction: str, *, user_id: str | None) -> None:
    """Best-effort durable semantic preference in the memory store."""
    try:
        from cognition.memory.memorize import MemoryStore  # noqa: F401
    except Exception:
        return
    verb = "prefers discussing" if direction == "prefer" else "should not unsolicitedly bring up"
    fact = f"User preference: Aiko {verb} {topic}."
    try:
        store = None
        try:
            from cognition.memory import get_memory_store
            store = get_memory_store(user_id)
        except Exception:
            store = None
        if store is None:
            return
        if hasattr(store, "add_fact"):
            store.add_fact(fact, kind="preference", user_id=user_id)
        elif hasattr(store, "remember"):
            store.remember(fact, user_id=user_id)
    except Exception as exc:
        log.debug("preference fact failed: %s", exc)
