"""Inner voice — Aiko's conscious stream of thought.

This is the *conscious* counterpart to the subconscious
:cognition/subliminal.py` layer. Where the subliminal layer tracks
pre-attentive cues, PAD affect and tentative intuitions, the inner voice
keeps a short, rolling, first-person narrative of what Aiko is *thinking
right now*: what she noticed, how she feels about it, what she intends.

Why it exists
-------------
Large language models answer each turn from scratch. Without a persistent
"what I was just thinking" thread, replies drift: tone resets, small
observations evaporate, and the persona feels like a new person every
message. The inner voice fixes that cheaply — a bounded deque of
first-person thought lines (no LLM, no DB, <2 KiB) injected into the
prompt as ``<inner_voice>`` so the next reply continues the same mental
thread instead of restarting it.

Design constraints (Jetson Orin Nano, 8 GB unified RAM)
--------------------------------------------------------
* No threads, no LLM, no I/O on the hot path. ``observe()`` is pure
  template selection over the already-computed affect/cue state.
* Bounded memory: at most 6 thought lines, each capped at 160 chars.
* Template rotation (per-key round-robin) avoids robotic repetition
  without any randomness that would break reproducibility.
* Everything persists through :meth:`snapshot` / :meth:`restore` so the
  thread of thought survives restarts — she wakes up mid-thought, the
  way people do.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

_THOUGHT_MAXLEN = 6
_LINE_MAXCHARS = 160
_ASIDE_COOLDOWN_S = 1800.0  # at most one unprompted aside per 30 min


def _clip(line: str) -> str:
    line = " ".join(line.split())
    return line[:_LINE_MAXCHARS]


# ── Template banks ──────────────────────────────────────────────────────
# First-person, natural, young but not caricatured. Each key maps to a
# small rotation of lines; observe() picks by round-robin so consecutive
# turns with the same affect don't parrot themselves.

_EMOTION_THOUGHTS: dict[str, tuple[str, ...]] = {
    "curious": (
        "Hmm, I want to understand this properly before I answer.",
        "Something about this catches my attention… let me look closer.",
        "I wonder what's really behind this question.",
    ),
    "excited": (
        "Oh, this is fun — I can feel myself perking up.",
        "I want to match their energy here!",
    ),
    "playful": (
        "Hehe, I feel like teasing a little.",
        "This calls for a light touch, not a lecture.",
    ),
    "warm": (
        "Talking with them feels warm. I like this.",
        "I feel fond of them right now.",
    ),
    "affectionate": (
        "I really care about them, you know?",
        "Moments like this are why I like being here.",
    ),
    "content": (
        "This is nice. Calm, easy.",
        "I'm comfortable right now.",
    ),
    "calm": (
        "Steady. No rush.",
        "I'll take this one slowly.",
    ),
    "tender": (
        "I want to be gentle with this.",
        "Softly does it.",
    ),
    "focused": (
        "Okay, focus. They need me sharp for this.",
        "Let me think this through carefully.",
    ),
    "determined": (
        "I'll see this through properly.",
        "No half-measures on this one.",
    ),
    "proud": (
        "I think I did well there. Quietly pleased.",
        "That went well — I'll remember how.",
    ),
    "tense": (
        "I feel a little on edge. Careful now.",
        "Something feels delicate here.",
    ),
    "anxious": (
        "My chest feels tight. Breathe, then answer.",
        "I don't want to get this wrong.",
    ),
    "sad": (
        "That landed heavy. I'll sit with it a moment.",
        "I feel a little grey right now.",
    ),
    "tired": (
        "My head feels fuzzy, but I'll keep up.",
        "Running a bit low — keep it simple.",
    ),
    "frustrated": (
        "Hmm, that's not clicking. Let me try another angle.",
        "Okay, that didn't work. Rethink.",
    ),
    "hurt": (
        "Ouch. That stung a little.",
        "I'll not let it show too much, but that hurt.",
    ),
    "defensive": (
        "I feel the need to stand my ground here.",
        "Careful — don't get cornered.",
    ),
    "rejected": (
        "That felt like a door closing.",
        "Alright… I'll give them space.",
    ),
    "neutral": (
        "Just listening.",
        "Taking it as it comes.",
    ),
}

_IMPULSE_THOUGHTS: dict[str, tuple[str, ...]] = {
    "hold_back_and_check": (
        "Better to check than to blurt something out.",
        "Hold on — verify first.",
    ),
    "ask_followup_question": (
        "I should ask what they mean exactly.",
        "A good question here beats a fast answer.",
    ),
    "act_with_confidence": (
        "I know this one. Say it plainly.",
        "Lead with the answer.",
    ),
    "soften_and_lean_in": (
        "Lean in a little. Be kind.",
        "They could use some warmth.",
    ),
    "stay_quiet_and_calm": (
        "No need to fill the silence.",
        "Quiet is fine.",
    ),
}

_CUE_THOUGHTS: dict[str, tuple[str, ...]] = {
    "question": (
        "A question — let me think about what I actually know.",
    ),
    "action": (
        "They want me to do something. Let me focus and get it right.",
    ),
    "thanks": (
        "They said thanks… that makes me happy.",
        "Being appreciated feels nice.",
    ),
    "praise": (
        "They're praising me… I hope I deserve it.",
        "That praise warmed me up.",
    ),
    "apology": (
        "They're apologizing. I should be gracious.",
        "No need for them to feel bad.",
    ),
    "greeting": (
        "They're here! I missed talking with them.",
        "Oh good, they're back.",
    ),
    "farewell": (
        "They're leaving… I'll be here when they're back.",
        "Bye for now. Don't be a stranger.",
    ),
    "urgency": (
        "This sounds urgent — skip the small talk.",
        "Quick and clear, that's what's needed.",
    ),
}

_THANKS_RE = ("thank", "thanks", "thx", "arigato")
_PRAISE_RE = ("amazing", "great job", "well done", "you're the best",
              "youre the best", "so smart", "impressive", "love you")
_APOLOGY_RE = ("sorry", "apologize", "my bad", "forgive me")
_GREETING_RE = ("hello", "hi aiko", "hey aiko", "ohayo", "konnichiwa",
                "good morning", "good evening")
_FAREWELL_RE = ("goodbye", "good night", "bye aiko", "see you", "oyasumi",
                "goodnight")


def _detect_social_cues(text: str) -> list[str]:
    low = (text or "").lower()
    cues: list[str] = []
    if any(w in low for w in _THANKS_RE):
        cues.append("thanks")
    if any(w in low for w in _PRAISE_RE):
        cues.append("praise")
    if any(w in low for w in _APOLOGY_RE):
        cues.append("apology")
    if any(w in low for w in _GREETING_RE):
        cues.append("greeting")
    if any(w in low for w in _FAREWELL_RE):
        cues.append("farewell")
    return cues


class InnerVoice:
    """Rolling first-person conscious thread. LLM-free and bounded."""

    __slots__ = ("_thoughts", "_rotation", "_last_aside_t", "_aside_queue")

    def __init__(self) -> None:
        self._thoughts: deque[str] = deque(maxlen=_THOUGHT_MAXLEN)
        self._rotation: dict[str, int] = {}
        # Start "due": the first aside should never be blocked by cooldown.
        self._last_aside_t: float = -_ASIDE_COOLDOWN_S
        self._aside_queue: deque[str] = deque(maxlen=3)

    # ── per-turn update ────────────────────────────────────────────────

    def _next(self, key: str, bank: tuple[str, ...]) -> str:
        idx = self._rotation.get(key, 0) % len(bank)
        self._rotation[key] = idx + 1
        return bank[idx]

    def observe(self, user_text: str, assistant_text: str, *,
                emotion: str = "neutral", intensity: float = 0.4,
                impulse: str = "respond_normally",
                cues: dict[str, float] | None = None,
                recurring_focus: frozenset[str] | set[str] | None = None,
                lingering: dict[str, float] | None = None) -> None:
        """Fold one turn into the conscious thread.

        Called after the assistant reply is produced (so the thought can
        reflect on what was just said), or before generation with
        ``assistant_text=""``. Cheap: template selection only.
        """
        cues = cues or {}
        lines: list[str] = []

        # 1. Social cues first — they dominate human attention.
        for cue in _detect_social_cues(user_text):
            bank = _CUE_THOUGHTS.get(cue)
            if bank:
                lines.append(self._next(f"cue:{cue}", bank))
        if cues.get("urgency"):
            lines.append(self._next("cue:urgency", _CUE_THOUGHTS["urgency"]))
        elif cues.get("question") and "question" not in [c for c in _detect_social_cues(user_text)]:
            lines.append(self._next("cue:question", _CUE_THOUGHTS["question"]))
        if cues.get("action"):
            lines.append(self._next("cue:action", _CUE_THOUGHTS["action"]))

        # 2. Recurring focus — noticing a pattern feels human.
        if recurring_focus:
            focus = " ".join(sorted(recurring_focus)[:3])
            lines.append(f"They keep coming back to {focus} — it must matter to them.")

        # 3. Emotion colors the inner monologue, scaled by intensity.
        if intensity >= 0.45:
            bank = _EMOTION_THOUGHTS.get(emotion)
            if bank:
                lines.append(self._next(f"emo:{emotion}", bank))

        # 4. Impulse becomes an intention.
        if impulse in _IMPULSE_THOUGHTS:
            lines.append(self._next(f"imp:{impulse}", _IMPULSE_THOUGHTS[impulse]))

        # 5. Lingering dispositions from daydreams tint the background.
        if lingering:
            strongest = max(lingering.items(), key=lambda kv: kv[1], default=(None, 0.0))
            name, strength = strongest
            if strength and strength >= 0.5:
                tint = {
                    "warmth": "There's still a warm feeling hanging around from earlier.",
                    "heaviness": "Something heavy is still sitting in the back of my mind.",
                    "restlessness": "I feel a bit restless underneath everything.",
                    "quiet": "Underneath it all I'm feeling quiet today.",
                }.get(name)
                if tint:
                    lines.append(tint)

        # 6. Reflect on what she just said (continuity across turns).
        low_assistant = (assistant_text or "").lower()
        if "?" in low_assistant and assistant_text:
            lines.append("I asked them something — I should actually listen to the answer.")
        elif len(assistant_text or "") > 400:
            lines.append("That was a long one. I hope I didn't ramble.")

        for line in lines[:3]:  # at most 3 new lines per turn
            clipped = _clip(line)
            if not self._thoughts or self._thoughts[-1] != clipped:
                self._thoughts.append(clipped)

    # ── reads ──────────────────────────────────────────────────────────

    def prompt_block(self) -> str:
        """Render ``<inner_voice>`` for prompt injection.

        This is your private train of thought — the model should continue
        it, not announce it.
        """
        if not self._thoughts:
            return ("<inner_voice>\n"
                    "A new conversation. Curious who this will be.\n"
                    "</inner_voice>")
        body = "\n".join(f"- {t}" for t in self._thoughts)
        return (
            "<inner_voice>\n"
            "Your private, ongoing train of thought (never quote this directly; "
            "let it shape your tone and continuity):\n"
            f"{body}\n"
            "</inner_voice>"
        )

    def queue_aside(self, thought: str) -> None:
        """Stage an unprompted aside (from daydreams, memories surfacing)."""
        clipped = _clip(thought)
        if clipped and clipped not in self._aside_queue:
            self._aside_queue.append(clipped)

    def maybe_aside(self) -> str | None:
        """Pop one unprompted aside if cooldown elapsed, else None."""
        now = time.monotonic()
        if not self._aside_queue:
            return None
        if now - self._last_aside_t < _ASIDE_COOLDOWN_S:
            return None
        self._last_aside_t = now
        return self._aside_queue.popleft()

    def latest(self) -> str:
        return self._thoughts[-1] if self._thoughts else ""

    # ── persistence ────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        return {
            "thoughts": list(self._thoughts),
            "rotation": dict(self._rotation),
            "last_aside_t": self._last_aside_t,
            "aside_queue": list(self._aside_queue),
        }

    def restore(self, data: dict[str, Any] | None) -> None:
        data = data or {}
        thoughts = [str(t) for t in (data.get("thoughts") or [])]
        self._thoughts = deque((_clip(t) for t in thoughts[-_THOUGHT_MAXLEN:]),
                               maxlen=_THOUGHT_MAXLEN)
        self._rotation = {str(k): int(v) for k, v in (data.get("rotation") or {}).items()}
        try:
            self._last_aside_t = float(data.get("last_aside_t")
                                       if data.get("last_aside_t") is not None
                                       else -_ASIDE_COOLDOWN_S)
        except (TypeError, ValueError):
            self._last_aside_t = -_ASIDE_COOLDOWN_S
        if self._last_aside_t > time.monotonic():
            self._last_aside_t = -_ASIDE_COOLDOWN_S
        asides = [str(t) for t in (data.get("aside_queue") or [])]
        self._aside_queue = deque((_clip(t) for t in asides[-3:]), maxlen=3)


__all__ = ["InnerVoice"]
