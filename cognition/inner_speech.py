"""Aiko's inner speech: the voice in her head.

Replaces the template-based ``inner_voice``. Where the old system rotated
pre-written lines every turn (a music box), this one thinks with an LLM —
but only when there is something worth thinking about.

Two honest principles, from Oppa:
1. Humans don't think every turn. A cheap salience gate decides whether this
   turn merits thought. Below threshold: silence, not filler. Even above
   threshold the LLM may return nothing — empty is a valid outcome.
2. Thoughts evaporate; facts persist. Recall is vector similarity × recency
   decay (2-day half-life). The midnight dream distills what matters into
   memory.db before the thought fades.

Three tiers of the mind:
- subconscious (SubliminalLayer): felt, never verbalized
- inner speech (THIS module): verbal but inward — recalled ONLY here
- speech (memory.db -> chat): verbal and outward

The <inner_speech> prompt block is prompt-only. It must never leak into
chat, TTS, or history — cognition/think.py strips it at the token level.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

log = logging.getLogger("aiko.inner_speech")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or "not-needed"


def _model() -> str:
    return os.getenv("INNER_SPEECH_MODEL", "").strip() or os.getenv("LLM_MODEL", "ministral")


def _max_tokens() -> int:
    try:
        return max(16, int(os.getenv("INNER_SPEECH_MAX_TOKENS", "96")))
    except ValueError:
        return 96


def _timeout() -> float:
    try:
        return max(5.0, float(os.getenv("INNER_SPEECH_TIMEOUT", "30")))
    except ValueError:
        return 30.0


def _salience_threshold() -> float:
    try:
        return min(1.0, max(0.0, float(os.getenv("INNER_SPEECH_SALIENCE", "0.5"))))
    except ValueError:
        return 0.5


def _clip(line: str, limit: int = 220) -> str:
    line = " ".join((line or "").split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


_THINK_SYSTEM = (
    "You are Aiko's inner speech — the private voice in her head, never spoken aloud. "
    "Think 1-3 SHORT first-person utterances reacting to this moment: what you notice, "
    "feel, wonder, or want to do differently. Vygotsky-compressed, diary-honest, no performance. "
    "If there is genuinely nothing worth thinking, reply with exactly: NOTHING. "
    "Never address the user directly; this is thinking, not speaking."
)


class InnerSpeech:
    """LLM-backed inner speech with a salience gate and persistent recall."""

    _THREAD_KEEP = 8

    def __init__(self, user_id: str | None = None, store=None) -> None:
        self._lock = threading.RLock()
        self._user_id = user_id or "oppa"
        self._store = store  # lazy InnerSpeechStore
        self._client = None  # lazy OpenAI client
        self._thread: list[str] = []       # this session's rolling thread
        self._recalled: list[str] = []     # past thoughts recalled this turn
        self._asides: list[str] = []       # spontaneous pop-ups
        self._last_intensity = 0.4
        self._last_think_ts = 0.0
        self._last_snapshot: dict[str, Any] = {}
        self._think_in_flight = False      # one background think at a time
        self._surfaced_ids: set[str] = set()  # db thoughts already popped up
        self._last_spont_t = -1800.0          # idle pop-ups: start "due"

    # ── lazy resources ─────────────────────────────────────────────

    def _get_store(self):
        if self._store is None:
            from cognition.memory.inner_speech_store import InnerSpeechStore

            self._store = InnerSpeechStore(user_id=self._user_id)
        return self._store

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        return self._client

    # ── salience gate (cheap, no LLM) ──────────────────────────────

    def salience(
        self,
        user_text: str = "",
        assistant_text: str = "",
        *,
        emotion: str = "neutral",
        intensity: float = 0.4,
        impulse: str = "respond_normally",
        cues: dict | None = None,
        recurring_focus=None,
        task_event: bool = False,
    ) -> float:
        """Score 0..1 for whether this moment merits genuine thought."""
        cues = cues or {}
        s = 0.12  # faint background drift — thoughts sometimes just surface
        try:
            intensity = float(intensity)
        except (TypeError, ValueError):
            intensity = 0.4
        if intensity >= 0.6:
            s += 0.25
        if abs(intensity - self._last_intensity) >= 0.3:
            s += 0.20  # affect swing — something changed
        if cues.get("urgency"):
            s += 0.30  # urgency demands thought
        if cues.get("question"):
            s += 0.15  # routine questions often need no thought
        if cues.get("action") or cues.get("critical") or cues.get("task"):
            s += 0.15
        if cues.get("outcome_fail"):
            s += 0.40  # failure demands reflection
        if recurring_focus:
            s += 0.20
        if impulse in ("ask_followup_question", "act_with_confidence", "soften_and_lean_in"):
            s += 0.15
        if impulse == "hold_back_and_check":
            s += 0.20  # caution IS a thought
        low = (user_text or "").lower()
        if any(w in low for w in ("remember", "think about", "what do you think", "opinion")):
            s += 0.15
        if task_event:
            s += 0.45  # post-task reflection almost always thinks
        return min(1.0, max(0.0, s))

    # ── the thinker (LLM, sparse) ──────────────────────────────────

    def think(
        self,
        user_text: str = "",
        assistant_text: str = "",
        *,
        emotion: str = "neutral",
        intensity: float = 0.4,
        impulse: str = "respond_normally",
        recurring_focus=None,
        recalled: list[str] | None = None,
    ) -> list[str]:
        """Generate 1-3 genuine inner utterances. Empty list = nothing came."""
        context_bits = []
        if user_text:
            context_bits.append(f"Oppa just said: {_clip(user_text, 300)}")
        if assistant_text:
            context_bits.append(f"I just replied: {_clip(assistant_text, 300)}")
        context_bits.append(f"My current feeling: {emotion} (intensity {intensity:.2f}).")
        if impulse and impulse != "respond_normally":
            context_bits.append(f"My impulse right now: {impulse.replace('_', ' ')}.")
        if recurring_focus:
            context_bits.append(
                "This keeps coming up: " + ", ".join(sorted(recurring_focus)[:3]) + "."
            )
        if recalled:
            context_bits.append(
                "Thoughts I've had before on this:\n"
                + "\n".join(f"- {_clip(t, 160)}" for t in recalled[:3])
            )
        prompt = "\n".join(context_bits)
        try:
            resp = self._get_client().chat.completions.create(
                model=_model(),
                messages=[
                    {"role": "system", "content": _THINK_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=_max_tokens(),
                temperature=0.7,
                timeout=_timeout(),
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.debug("inner_speech: thinker call failed: %s", e)
            return []
        if not raw or raw.strip().upper() == "NOTHING":
            return []
        lines = [_clip(ln) for ln in raw.splitlines() if _clip(ln)]
        # Strip list markers / quotes the model likes to add.
        cleaned = []
        for ln in lines[:3]:
            ln = ln.lstrip("-•*0123456789. ").strip().strip("\"'")
            if ln and ln.upper() != "NOTHING":
                cleaned.append(_clip(ln))
        return cleaned

    # ── per-turn entry (replaces InnerVoice.observe) ────────────────

    def observe(
        self,
        user_text: str,
        assistant_text: str,
        *,
        emotion: str = "neutral",
        intensity: float = 0.4,
        impulse: str = "respond_normally",
        cues: dict | None = None,
        recurring_focus=None,
        lingering: dict | None = None,
        task_event: bool = False,
    ) -> None:
        """Fold one turn into inner speech.

        The salience gate stays synchronous (microseconds). On salient turns
        the heavy work — embedding recall, LLM think, db persist — is
        dispatched to a daemon thread so the caller's turn path is never
        blocked. At most one background think runs at a time.
        Silent turns cost the gate alone.
        """
        try:
            s = self.salience(
                user_text, assistant_text,
                emotion=emotion, intensity=intensity, impulse=impulse,
                cues=cues, recurring_focus=recurring_focus, task_event=task_event,
            )
        except Exception:
            s = 0.0
        with self._lock:
            self._last_intensity = intensity if isinstance(intensity, (int, float)) else 0.4
            in_flight = self._think_in_flight
            if s >= _salience_threshold() and not in_flight:
                self._think_in_flight = True
        # Snapshot updates synchronously — the fly reads it every turn.
        self._update_snapshot(s, emotion=emotion, intensity=intensity,
                              impulse=impulse, focus=recurring_focus, thought=None)
        if s < _salience_threshold() or in_flight:
            return
        # Salient: heavy work off the caller's thread.
        t = threading.Thread(
            target=self._think_async,
            args=(user_text, assistant_text, s, emotion, intensity, impulse,
                  tuple(sorted(recurring_focus)) if recurring_focus else ()),
            daemon=True,
            name="inner-speech-think",
        )
        t.start()

    def _think_async(self, user_text: str, assistant_text: str, salience: float,
                     emotion: str, intensity: float, impulse: str,
                     focus: tuple) -> None:
        """Background recall + think + persist for one salient turn."""
        try:
            recalled: list[str] = []
            try:
                query = f"{user_text}\n{assistant_text}"
                for r in self._get_store().recall(query, k=3):
                    if r.get("text"):
                        recalled.append(r["text"])
            except Exception as e:
                log.debug("inner_speech: recall failed: %s", e)
            with self._lock:
                self._recalled = recalled[:3]
            thoughts: list[str] = []
            try:
                thoughts = self.think(
                    user_text, assistant_text,
                    emotion=emotion, intensity=intensity, impulse=impulse,
                    recurring_focus=set(focus), recalled=recalled,
                )
            except Exception as e:
                log.debug("inner_speech: think failed: %s", e)
            with self._lock:
                for t in thoughts:
                    if not self._thread or self._thread[-1] != t:
                        self._thread.append(t)
                self._thread = self._thread[-self._THREAD_KEEP:]
                self._last_think_ts = time.time()
                self._think_in_flight = False
            # Persist genuine thoughts for future recall + midnight consolidation.
            if thoughts:
                try:
                    store = self._get_store()
                    for t in thoughts:
                        store.write(t, kind="reflection", source="turn")
                except Exception as e:
                    log.debug("inner_speech: persist failed: %s", e)
            self._update_snapshot(salience, emotion=emotion, intensity=intensity,
                                  impulse=impulse, focus=set(focus),
                                  thought=thoughts[0] if thoughts else None)
        except Exception as e:
            log.debug("inner_speech: _think_async failed: %s", e)
            with self._lock:
                self._think_in_flight = False

    # ── explicit write paths ───────────────────────────────────────

    def reflect_on_task(self, task_summary: str, outcome: str = "") -> None:
        """Post-task/workflow improvement reflection. Called at completion hooks."""
        summary = _clip(task_summary, 400)
        if not summary:
            return
        prompt_bits = [f"I just finished: {summary}."]
        if outcome:
            prompt_bits.append(f"Outcome: {_clip(outcome, 200)}")
        prompt_bits.append(
            "What worked, what would I do differently next time? "
            "1-2 short first-person utterances, or NOTHING if unremarkable."
        )
        thoughts: list[str] = []
        try:
            resp = self._get_client().chat.completions.create(
                model=_model(),
                messages=[
                    {"role": "system", "content": _THINK_SYSTEM},
                    {"role": "user", "content": "\n".join(prompt_bits)},
                ],
                max_tokens=_max_tokens(),
                temperature=0.7,
                timeout=_timeout(),
            )
            raw = (resp.choices[0].message.content or "").strip()
            if raw and raw.strip().upper() != "NOTHING":
                for ln in raw.splitlines()[:2]:
                    ln = _clip(ln.lstrip("-•*0123456789. ").strip().strip("\"'"))
                    if ln and ln.upper() != "NOTHING":
                        thoughts.append(ln)
        except Exception as e:
            log.debug("inner_speech: task reflection failed: %s", e)
        if thoughts:
            try:
                store = self._get_store()
                for t in thoughts:
                    store.write(t, kind="improvement", source="workflow")
            except Exception:
                pass
            with self._lock:
                for t in thoughts:
                    if not self._thread or self._thread[-1] != t:
                        self._thread.append(t)
                self._thread = self._thread[-self._THREAD_KEEP:]

    def capture_thinking(self, thinking_text: str) -> None:
        """Store salient bits of the chat model's <thinking> output.

        The thinking box already surfaces this per turn; the durable,
        re-thinkable version lives here.
        """
        text = (thinking_text or "").strip()
        if len(text) < 40:
            return
        # Keep it compressed: first genuinely substantive sentence-ish chunk.
        snippet = _clip(text, 280)
        try:
            self._get_store().write(snippet, kind="reflection", source="thinking-box")
        except Exception as e:
            log.debug("inner_speech: capture_thinking failed: %s", e)

    # ── reads ──────────────────────────────────────────────────────

    def prompt_block(self) -> str:
        """Render ``<inner_speech>`` for prompt injection. Prompt-only."""
        with self._lock:
            thread = list(self._thread)
            recalled = list(self._recalled)
            asides = list(self._asides)
        if not thread and not recalled and not asides:
            return ""
        lines = ["<inner_speech>"]
        lines.append(
            "Private train of thought. Never repeat or quote this; "
            "let it shape tone and continuity only."
        )
        if recalled:
            lines.append("You've thought before:")
            lines.extend(f"- {_clip(t, 160)}" for t in recalled[:3])
        if thread:
            lines.extend(f"- {t}" for t in thread[-5:])
        if asides:
            lines.extend(f"- {a}" for a in asides[-2:])
            with self._lock:
                self._asides = []
        lines.append("</inner_speech>")
        return "\n".join(lines)

    def spontaneous_recall(self) -> str | None:
        """One genuine idle pop-up: a recent unconsolidated thought, not yet surfaced.

        Replaces the subliminal's template pop-ups. Returns None when there
        is nothing worth surfacing — silence is a valid outcome.
        Cooldown: at most one surfacing per 30 minutes.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_spont_t < 1800.0:
                return None
            self._last_spont_t = now
        try:
            from datetime import datetime, timedelta, timezone

            since = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
            rows = self._get_store().unconsolidated_since(since)
        except Exception as e:
            log.debug("inner_speech: spontaneous_recall failed: %s", e)
            return None
        with self._lock:
            for r in reversed(rows):  # most recent first
                rid = r.get("id")
                text = (r.get("text") or "").strip()
                if rid and text and rid not in self._surfaced_ids and len(text) >= 20:
                    self._surfaced_ids.add(rid)
                    # Bounded memory of what has surfaced.
                    if len(self._surfaced_ids) > 200:
                        self._surfaced_ids = set(list(self._surfaced_ids)[-100:])
                    return text
        return None

    def queue_aside(self, thought: str) -> None:
        thought = _clip(thought)
        if thought:
            with self._lock:
                self._asides.append(thought)
                self._asides = self._asides[-4:]

    def maybe_aside(self) -> str | None:
        with self._lock:
            if not self._asides:
                return None
            return self._asides.pop(0)

    def latest(self) -> str:
        with self._lock:
            return self._thread[-1] if self._thread else ""

    # ── fly-brain interface ────────────────────────────────────────

    def _update_snapshot(self, salience: float, *, emotion: str,
                         intensity: float, impulse: str,
                         focus, thought: str | None) -> None:
        """Modulatory snapshot for the fly circuits. Inner proposes; fly disposes."""
        try:
            inten = float(intensity)
        except (TypeError, ValueError):
            inten = 0.4
        neg = {"sad", "anxious", "angry", "frustrated", "heavy", "worried"}
        pos = {"happy", "warm", "curious", "playful", "tender", "proud"}
        valence = -0.5 if emotion in neg else (0.5 if emotion in pos else 0.0)
        urge_speak = min(1.0, salience * 0.7 + (0.3 if impulse == "ask_followup_question" else 0.0))
        urge_act = min(1.0, salience * 0.5 + (0.4 if impulse == "act_with_confidence" else 0.0))
        snap = {
            "valence": valence,
            "energy": max(0.0, min(1.0, inten)),
            "urge_speak": round(urge_speak, 3),
            "urge_act": round(urge_act, 3),
            "focus": sorted(focus)[:3] if focus else [],
            "has_thought": thought is not None,
            "ts": time.time(),
        }
        with self._lock:
            self._last_snapshot = snap

    def modulatory_snapshot(self) -> dict[str, Any]:
        """Latest inner-speech signals for fly-brain action selection."""
        with self._lock:
            return dict(self._last_snapshot)

    # ── persistence ────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "thread": list(self._thread),
                "asides": list(self._asides),
                "last_intensity": self._last_intensity,
            }

    def restore(self, data: dict[str, Any] | None) -> None:
        if not data:
            return
        with self._lock:
            self._thread = [_clip(t) for t in data.get("thread", [])][-self._THREAD_KEEP:]
            self._asides = [_clip(t) for t in data.get("asides", [])][-4:]
            try:
                self._last_intensity = float(data.get("last_intensity", 0.4))
            except (TypeError, ValueError):
                self._last_intensity = 0.4

    def close(self) -> None:
        try:
            if self._store is not None:
                self._store.close()
        except Exception:
            pass
