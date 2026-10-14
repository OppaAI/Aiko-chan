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
    "contemplativeNod", "tapFinger", "headShake", "blowKiss",
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
             "oyasumi", "mata ne", "i am leaving", "im leaving", "leaving now",
             "got to go", "gotta go", "gotta run", "see you later",
             "take care", "bye bye", "bye-bye", "bye")
_CELEBRATION = ("congratulations", "congrats", "yay", "we did it", "birthday",
                "party", "let's dance", "lets dance", "dance", "music",
                "omedetou")

# Outfit requests: user asks Aiko to change clothes. Keys must match the
# wardrobe manifest in the combined wardrobe VRM (vrm.js window.aikoSetOutfit).
# Order matters: check specific outfits before the generic change verbs.
_OUTFIT_PATTERNS = (
    ("pajama",    ("pajama", "pajamas", "pyjama", "寝間着", "パジャマ")),
    ("swimsuit",  ("swimsuit", "bikini", "水着", "みずぎ")),
    ("sailor",    ("sailor", "セーラー",)),
    ("gym",       ("gym clothes", "gym shirt", "体操服", "たいそうふく")),
    ("beret",     ("beret", "ベレー",)),
    ("seifuku",   ("seifuku", "school uniform", "制服", "せいふく")),
    ("cardigan",  ("pink cardigan", "カーディガン")),
    ("loungewear",("loungewear", "room wear")),
    ("yukata",    ("yukata", "浴衣", "ゆかた")),
)
_OUTFIT_VERBS = ("change into", "wear the", "wear your", "put on",
                 "着替えて", "きがえて", "着せて")
# Negations that cancel an outfit request ("don't change", "not the pajamas").
# Kept narrow: only clear "don't do it" signals, so "wear the pajamas,
# not the seifuku" still matches pajama.
_OUTFIT_NEGATIONS = ("don't", "do not", "doesn't", "やめて", "いやだ")

_AGREE = (
    "you're right", "youre right", "you are right", "exactly", "i agree",
    "good point", "great point", "that's right", "thats right", "absolutely",
    "of course", "definitely", "couldn't agree more", "couldnt agree more",
    "well said", "fair enough", "you've got a point", "youve got a point",
    "そうだね", "その通り", "確かに", "たしかに", "おっしゃる通り",
)
_DISAGREE = (
    "i disagree", "not quite", "not exactly", "actually,", "however,",
    "on the other hand", "i see it differently", "i don't think so",
    "i dont think so", "that's not right", "thats not right",
    "hmm, no", "no,", "nope", "i'm afraid not", "im afraid not",
    "でも", "いや", "ちがう", "違う", "そうじゃない",
)



def _contains(low: str, words: tuple[str, ...]) -> bool:
    # ASCII \b doesn't exist inside CJK text (all chars are non-word), and a
    # trailing \b after punctuation can never match before a space — so:
    #   pure ASCII word -> strict \b..\b
    #   ASCII with punctuation ("actually,") -> leading \b only
    #   CJK / non-ASCII -> bare substring (no false-boundary risk)
    def _pat(w: str) -> str:
        esc = re.escape(w)
        if re.fullmatch(r"[\w]+", w, flags=re.ASCII):
            return rf"\b{esc}\b"
        if re.fullmatch(r"[\x00-\x7F]+", w):
            return rf"\b{esc}"
        return esc
    # re.ASCII: \b uses ASCII word boundaries, so English cues still match
    # when adjacent to non-ASCII text (e.g. "byeまたね"). CJK patterns are
    # bare substrings, unaffected by the flag.
    return any(re.search(_pat(w), low, flags=re.ASCII) for w in words)


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
            candidate = "blowKiss"    # warm farewell: fingertips to lips, blown outward
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

    def suggest_reply(self, assistant_text: str) -> str | None:
        """Pick a reactive head gesture for AIKO'S reply, or None.

        Nod when she agrees with the user, shake when she disagrees —
        the same deterministic, no-LLM design as suggest(). Shares the
        cooldown state so she never nods and claps on the same turn.
        """
        now = time.monotonic()
        if now - self._last_any_t < _GLOBAL_COOLDOWN_S:
            return None
        low = (assistant_text or "").lower()
        candidate: str | None = None
        # Disagreement wins ties: a "yes, but…" reply reads as disagreeing.
        if _contains(low, _DISAGREE):
            candidate = "headShake"
        elif _contains(low, _AGREE):
            candidate = "headNod"
        if candidate is None or candidate not in KNOWN_GESTURES:
            return None
        if now - self._last_gesture_t.get(candidate, 0.0) < _GESTURE_COOLDOWN_S:
            return None
        self._last_gesture_t[candidate] = now
        self._last_any_t = now
        return candidate

    def suggest_outfit(self, user_text: str) -> str | None:
        """Map an explicit outfit request to a wardrobe key, or None.

        Requires BOTH an outfit noun and an affirmative intent verb —
        a bare noun ("pajamas?") or a negated request ("don't wear the
        pajamas") is ignored. No cooldowns — changing clothes is always
        user-initiated.
        """
        low = (user_text or "").lower()
        if _contains(low, _OUTFIT_NEGATIONS):
            return None
        if not _contains(low, _OUTFIT_VERBS):
            return None
        for key, nouns in _OUTFIT_PATTERNS:
            if _contains(low, nouns):
                return key
        return None

    def reset_cooldowns(self) -> None:
        self._last_gesture_t.clear()
        self._last_any_t = 0.0


__all__ = ["MotionDirector", "KNOWN_GESTURES"]
