"""Motion director — maps conversation events to VRM avatar gestures.

The body answers faster than the LLM. A lean-in, clap, or bow lands in
under 100 ms, so the turn *feels* instant even while tokens are still
generating — the single biggest perceived-latency win in the stack, and
exactly what a human does (we signal attention with the body long before
we find words).

Design
-------
* Pure function of (user text, emotion, intensity, impulse) — no LLM, no
  I/O, deterministic given inputs except cooldown state.
* Per-gesture cooldowns (25 s) plus a global reactive cooldown (8 s) so
  Aiko doesn't flail on every message.
* Gesture names must exist in the frontend gesture engine
  (``interface/webui/static/vrm.js``); unknown names are never emitted.
* Delivery is via ``webui_bridge().play_gesture(name)`` which the
  frontend handles as ``{"type": "gesture", "name": ...}``.
"""

from __future__ import annotations

import re
import time

# Every gesture the frontend engine can play (vrm.js gesture switch).
KNOWN_GESTURES = frozenset({
    "lookAround", "lookAtHand", "hairBrush", "fingerPlay", "meetGaze",
    "curiousTilt", "shiftWeight", "stretchNeck", "raiseHand", "chinTouch",
    "shoulderRoll", "sway", "headNod", "wristFlick", "adjustSleeve",
    "handOnHip", "crossArms", "touchCollar", "brushShoulder", "stretchArm",
    "leanIn", "openPalm", "speakingNod", "earTuck", "gentleStretch",
    "handsClasp", "thoughtfulLook", "emphasizePoint", "bothHandsExplain",
    "chinThink", "handNearMouth", "armsFoldThink", "lookUpThink",
    "contemplativeNod", "tapFinger",
    # Added by the presence upgrade: young, lively motions.
    "clap", "dance", "wave", "giggle", "bow",
})

_GESTURE_COOLDOWN_S = 25.0
_GLOBAL_COOLDOWN_S = 8.0

_THANKS = ("thank you", "thanks", "thx", "arigato", "doumo")
_PRAISE = ("amazing", "great job", "well done", "you're the best",
           "youre the best", "so smart", "impressive", "love you",
           "sugoi", "kakkoii")
_APOLOGY = ("sorry", "apologize", "my bad", "forgive me", "gomen")
_GREETING = ("hello", "hey aiko", "hi aiko", "ohayo", "konnichiwa",
             "good morning", "good evening", "konbanwa")
_FAREWELL = ("goodbye", "good night", "goodnight", "bye aiko", "see you",
             "oyasumi", "mata ne")
_CELEBRATION = ("congratulations", "congrats", "yay", "we did it", "birthday",
                "party", "let's dance", "lets dance", "dance", "music",
                "omedetou")


def _contains(low: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", low) for w in words)


class MotionDirector:
    """Stateful (cooldowns only) conversation → gesture mapper."""

    __slots__ = ("_last_gesture_t", "_last_any_t")

    def __init__(self) -> None:
        self._last_gesture_t: dict[str, float] = {}
        self._last_any_t: float = 0.0

    # ── public API ────────────────────────────────────────────────────

    def on_listening_start(self) -> str:
        """Call the moment the user starts speaking / a turn begins.

        Always returns ``leanIn`` — the universal human "I'm listening".
        No cooldown: attention must be instant.
        """
        return "leanIn"

    def suggest(self, user_text: str, *, emotion: str = "neutral",
                intensity: float = 0.4,
                impulse: str = "respond_normally") -> str | None:
        """Pick one reactive gesture for this turn, or None.

        Priority: explicit social cues first, then affect-driven motion.
        """
        now = time.monotonic()
        if now - self._last_any_t < _GLOBAL_COOLDOWN_S:
            return None
        low = (user_text or "").lower()

        candidate: str | None = None
        if _contains(low, _PRAISE):
            candidate = "clap"          # delighted applause for praise
        elif _contains(low, _THANKS):
            candidate = "bow"           # a grateful little bow
        elif _contains(low, _APOLOGY):
            candidate = "handsClasp"    # reassuring, "it's okay"
        elif _contains(low, _GREETING):
            candidate = "wave"
        elif _contains(low, _FAREWELL):
            candidate = "wave"
        elif _contains(low, _CELEBRATION):
            candidate = "dance"         # pure joy — she dances
        elif emotion == "playful" and intensity >= 0.55:
            candidate = "giggle"
        elif emotion in ("excited",) and intensity >= 0.7:
            candidate = "clap"
        elif emotion in ("warm", "affectionate", "tender") and intensity >= 0.6:
            candidate = "handsClasp"    # touched, hands together
        elif impulse == "soften_and_lean_in":
            candidate = "leanIn"
        elif impulse == "ask_followup_question" or emotion == "curious":
            candidate = "curiousTilt"
        elif impulse == "act_with_confidence":
            candidate = "emphasizePoint"

        if candidate is None or candidate not in KNOWN_GESTURES:
            return None
        if now - self._last_gesture_t.get(candidate, 0.0) < _GESTURE_COOLDOWN_S:
            return None
        self._last_gesture_t[candidate] = now
        self._last_any_t = now
        return candidate

    def reset_cooldowns(self) -> None:
        self._last_gesture_t.clear()
        self._last_any_t = 0.0


__all__ = ["MotionDirector", "KNOWN_GESTURES"]
