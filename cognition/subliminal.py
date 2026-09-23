"""Subliminal layer — human-like subconscious for Aiko on Jetson.

Design mirrors a coarse human mind model but stays bounded and lock-free
on hot paths. No threads, no LLM, no DB on hot path. All state is a
handful of small deques + scalars (<4 KiB per identity).

Layers (executed in order per turn; only L1+L2 run on hot path):

  L1 pre-attentive    < 0.2 ms
    - Token-bag novelty vs prior turns (gestalt grouping).
    - Cheap cue scan: urgency, sentiment, action verb, negation,
      uncertainty, identity, time-ref, contradiction overlap.

  L2 affect / drive  < 0.2 ms
    - PAD valence/arousal/dominance update (lightweight EMA).
    - Drive mix: curiosity / care / caution / agency / comfort.

  L3 implicit memory  deferred to prompt assembly
    - Intuition candidates (contradictions, recurring focus, lessons).
    - Affective tags (emotional valence on each intuition).

  L4 meta-cognitive  deferred
    - Emotion label (vocab-limited, blend-safe).
    - Impulse + bias text for prompt.
    - VRM expression + blendshape weights for the avatar.

This is composed into EdgeCognitiveState in cognition/attention.py.
Only the hot path (record()) invokes scan(); everything else is
called when assembling the prompt or broadcasting to VRM.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from system.log import get_logger

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9_]{3,}")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall((text or "").lower()))


# ── Layer 1: pre-attentive scan (hot path) ──────────────────────────────

# Cheap cue vocabularies — cheap, no allocation. Kept narrow to bound
# regex cost on Jetson. Extended when a future CUE_KEYWORDS env var
# overrides the default.
_DEFAULT_CUES = {
    "urgency":   ("urgent", "asap", "now", "right now", "immediately", "crisis", "emergency"),
    "negation":  ("not ", "no ", "never", "don't", "dont", "cannot", "can't", "changed my mind"),
    "uncertain": ("maybe", "perhaps", "not sure", "unclear", "i think", "might"),
    "question":  ("?", "what", "why", "how", "when", "where", "who", "remember"),
    "action":    ("can you", "could you", "please", "do this", "make", "build", "write", "fix"),
    "time":      ("today", "tonight", "tomorrow", "now", "soon", "asap", "deadline", "due"),
    "identity":  ("you know me", "who am i", "remember me", "my name", "about me"),
}


def _scan_cues(text: str) -> dict[str, float]:
    """Cheap cue presence in [0,1]; one pass, case-folded."""
    low = (text or "").lower()
    out = {}
    for name, kws in _DEFAULT_CUES.items():
        if name == "question" and "?" in low:
            out[name] = 1.0
            continue
        out[name] = 1.0 if any(k in low for k in kws) else 0.0
    return out


@dataclass(slots=True)
class PreAttentiveScan:
    """L1 result. Cheap, allocated per record()."""
    novelty: float = 0.0           # 0..1, token-set Jaccard novelty vs prior turn
    cues: dict[str, float] = field(default_factory=dict)
    token_count: int = 0
    recurring_focus: frozenset[str] = frozenset()


# ── Layer 2: affect / drive ─────────────────────────────────────────────

@dataclass(slots=True)
class AffectState:
    """PAD + drive mix. EMA-smoothed."""
    valence: float = 0.0     # -1..+1, positive↔negative
    arousal: float = 0.5     # 0..1, low↔high energy
    dominance: float = 0.5   # 0..1, submissive↔in-control

    curiosity: float = 0.5
    care: float = 0.5
    caution: float = 0.5
    agency: float = 0.5
    comfort: float = 0.5


# ── Layer 4: emotion label vocabulary (Ekman + Aiko-specific blends) ────
# Maps an AffectState → a small set of (emotion_key, intensity) tags that
# downstream consumers (VRM blendshapes, prompt labels) can blend. We do
# NOT use a full Ekman model — too much state for a 8GB Jetson. Instead,
# a coarse palette with strong, cheap associations.

_EMOTION_VOCAB: tuple[tuple[str, tuple[float, float, float]], ...] = (
    # (name, (valence_threshold, arousal_threshold, dominance_threshold))
    # value, energy, control
    ("calm",         ( 0.0,  0.0, 0.0)),
    ("neutral",      ( 0.0,  0.0, 0.0)),  # base when no emotion fires
    ("content",      ( 0.3,  0.0, 0.0)),
    ("warm",         ( 0.3,  0.0, 0.3)),
    ("curious",      ( 0.1,  0.4, 0.0)),
    ("excited",      ( 0.4,  0.7, 0.0)),
    ("proud",        ( 0.4,  0.5, 0.6)),
    ("playful",      ( 0.3,  0.5,-0.3)),
    ("tender",       ( 0.4,  0.0, 0.5)),
    ("affectionate", ( 0.5,  0.0, 0.5)),
    ("focused",      ( 0.0,  0.5, 0.6)),
    ("determined",   ( 0.0,  0.6, 0.7)),
    ("tense",        (-0.1,  0.6, 0.0)),
    ("anxious",      (-0.2,  0.6,-0.2)),
    ("sad",          (-0.4,  0.0, 0.0)),
    ("tired",        (-0.2, -0.4, 0.0)),
    ("frustrated",   (-0.3,  0.5, 0.3)),
    ("hurt",         (-0.5,  0.2,-0.2)),
    ("defensive",    (-0.2,  0.4, 0.5)),
    ("rejected",     (-0.4,  0.2,-0.4)),
)


def _classify_emotion(a: AffectState) -> tuple[str, float]:
    """Pick best emotion for current PAD by nearest prototype; returns (name, confidence)."""
    # Prototype PAD targets (hand-tuned, cheap Euclidean)
    prototypes: dict[str, tuple[float, float, float]] = {
        "calm":         ( 0.0,  0.2, 0.5),
        "content":      ( 0.4,  0.3, 0.5),
        "warm":         ( 0.5,  0.3, 0.5),
        "curious":      ( 0.1,  0.6, 0.4),
        "excited":      ( 0.6,  0.8, 0.5),
        "proud":        ( 0.5,  0.6, 0.7),
        "playful":      ( 0.4,  0.6, 0.4),
        "tender":       ( 0.5,  0.3, 0.4),
        "affectionate": ( 0.6,  0.4, 0.5),
        "focused":      ( 0.1,  0.6, 0.7),
        "determined":   ( 0.2,  0.7, 0.7),
        "tense":        (-0.1,  0.7, 0.3),
        "anxious":      (-0.3,  0.7, 0.2),
        "sad":          (-0.5,  0.3, 0.3),
        "tired":        (-0.2,  0.2, 0.3),
        "frustrated":   (-0.4,  0.6, 0.4),
        "hurt":         (-0.5,  0.5, 0.2),
        "defensive":    (-0.2,  0.5, 0.6),
        "rejected":     (-0.5,  0.4, 0.2),
    }
    best = ("neutral", 0.4)
    best_dist = 10.0
    for name, (vt, at, dt) in prototypes.items():
        d = ((a.valence - vt) ** 2 + (a.arousal - at) ** 2 + (a.dominance - dt) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            # confidence = 1 - normalized distance (max ~2.5)
            conf = max(0.35, min(0.95, 1.0 - d / 2.0))
            best = (name, conf)
    return best


# VRM blendshape hints for webui.set_expression(name, intensity).
# Maps emotion → a blendshape-style expression name the avatar understands.
_EMOTION_TO_VRM = {
    "calm":         ("calm",        0.5),
    "content":      ("smile",       0.5),
    "warm":         ("smile_soft",  0.6),
    "curious":      ("eyebrows_up", 0.6),
    "excited":      ("smile_big",   0.7),
    "proud":        ("smile_soft",  0.6),
    "playful":      ("smile_playful", 0.7),
    "tender":       ("smile_soft",  0.5),
    "affectionate": ("smile_soft",  0.7),
    "focused":      ("neutral",     0.4),
    "determined":   ("neutral",     0.5),
    "tense":        ("brow_furrow", 0.5),
    "anxious":      ("brow_furrow", 0.6),
    "sad":          ("frown",       0.6),
    "tired":        ("eyes_droop",  0.5),
    "frustrated":   ("brow_furrow", 0.7),
    "hurt":         ("frown",       0.6),
    "defensive":    ("brow_furrow", 0.5),
    "rejected":     ("frown",       0.7),
    "neutral":      ("neutral",     0.3),
}


# ── SubliminalLayer ───────────────────────────────────────────────────────

class SubliminalLayer:
    """Bounded subconscious layer composed into EdgeCognitiveState.

    Owns L1 pre-attentive, L2 affect/drive, L3 implicit (intuitions +
    affective tags), and L4 meta-cognitive (emotion label + impulse +
    bias + VRM expression broadcast).
    """

    __slots__ = (
        "_intuitions", "_pre_attentive", "_affect",
        "_emotion", "_emotion_intensity", "_impulse", "_bias",
        "_vrm_emit", "_vrm_last", "_vrm_last_t",
        "_lock_ref", "_recent_event_tokens",
        # Daydream state: lingering dispositions + spontaneous thoughts.
        "_lingering", "_last_daydream_t", "_spontaneous", "_last_spont_t",
    )

    def __init__(self, lock: threading.RLock | None = None) -> None:
        self._intuitions: deque[tuple[str, str]] = deque(maxlen=4)  # (text, affective_tag)
        self._pre_attentive: PreAttentiveScan | None = None
        self._affect = AffectState()
        self._emotion: str = "neutral"
        self._emotion_intensity: float = 0.4
        self._impulse: str = ""
        self._bias: str = ""
        self._vrm_emit: bool = False
        self._vrm_last: str = "neutral"
        self._vrm_last_t: float = 0.0
        self._lock_ref = lock
        # Windowed tokens for novelty across the last few events.
        self._recent_event_tokens: deque[frozenset[str]] = deque(maxlen=4)
        # Daydream state — consolidated between turns, never on the hot path.
        self._lingering: dict[str, float] = {}   # disposition -> strength 0..1
        self._last_daydream_t: float = 0.0
        self._spontaneous: deque[str] = deque(maxlen=3)  # unprompted thoughts
        # Start "due": the first surfacing should never be cooldown-blocked.
        self._last_spont_t: float = -1800.0

    # ── L1/L2: hot-path scan() called once per record() ────────────────

    def scan(self, user: str, snapshot: dict, events: deque) -> PreAttentiveScan:
        """L1+L2 in one pass. Always returns within 0.2 ms on Jetson."""
        toks = _tokens(user)
        # L1: novelty (Jaccard with the last event)
        novelty = 0.0
        if self._recent_event_tokens:
            last = self._recent_event_tokens[-1]
            union = toks | last
            if union:
                novelty = 1.0 - (len(toks & last) / len(union))
        # L1: recurring focus across last 3 events
        recurring: frozenset[str] = frozenset()
        if len(events) >= 3:
            rec = [_tokens(e.user) for e in list(events)[-3:]]
            inter = set(rec[0]).intersection(*rec[1:]) if rec else set()
            inter -= {"can", "could", "please", "today"}
            if inter:
                recurring = frozenset(sorted(inter)[:4])
        scan = PreAttentiveScan(
            novelty=float(min(1.0, max(0.0, novelty))),
            cues=_scan_cues(user),
            token_count=len(toks),
            recurring_focus=recurring,
        )
        self._pre_attentive = scan
        self._recent_event_tokens.append(toks)

        # L2: PAD update (lightweight lexical cues drive it)
        cues = scan.cues
        # Valence — negation drags down, time/warmth cues lift
        v = self._affect.valence
        v -= 0.3 * cues.get("negation", 0.0)
        v += 0.15 * cues.get("question", 0.0)  # open questions slightly warm
        # cue lexical sentiment (cheap, bounded)
        low = (user or "").lower()
        pos = sum(low.count(w) for w in ("love", "great", "thanks", "happy", "good", "excited", "please"))
        neg = sum(low.count(w) for w in ("hate", "sad", "angry", "bad", "frustrated", "worried", "hurt"))
        denom = max(1, pos + neg)
        v += 0.2 * ((pos - neg) / denom)
        v = max(-1.0, min(1.0, 0.7 * v + 0.3 * ((pos - neg) / denom)))

        # Arousal — urgency/time/questions raise energy
        ar = self._affect.arousal
        ar += 0.25 * cues.get("urgency", 0.0)
        ar += 0.1 * cues.get("question", 0.0)
        ar += 0.1 * cues.get("time", 0.0)
        # Identity/care queries slightly calm
        ar -= 0.05 * cues.get("identity", 0.0)
        ar = max(0.0, min(1.0, 0.85 * ar + 0.05))
        # Fly DN layer: descending arousal drive blends in (small, clamped).
        try:
            _dn_mode = (os.getenv("MEMORY_FLYDN_MODE", "off") or "off").strip().lower()
        except Exception:
            _dn_mode = "off"
        if _dn_mode in ("shadow", "live"):
            try:
                from cognition.flysense import FlyDN as _FlyDN
                _drv = _FlyDN().drive(energy=float(ar),
                                      decisiveness=float(cues.get("action", 0.0)),
                                      affect=float(v))
            except Exception as exc:
                log.debug("subliminal flydn skipped: %s", exc)
                _drv = None
            if _drv is not None:
                log.debug("subliminal flydn mode=%s arousal=%.2f", _dn_mode, _drv["arousal"])
                if _dn_mode == "live":
                    ar = max(0.0, min(1.0, ar + 0.1 * (_drv["arousal"] - 0.5) * 2.0))

        # Dominance — agency grows when action cues are present, uncertainty shrinks it
        d = self._affect.dominance
        d += 0.15 * cues.get("action", 0.0)
        d -= 0.15 * cues.get("uncertain", 0.0)
        d += 0.05 * cues.get("identity", 0.0)
        d = max(0.0, min(1.0, 0.9 * d + 0.05))

        # Drive mix — derived from PAD
        self._affect = AffectState(
            valence=float(v),
            arousal=float(ar),
            dominance=float(d),
            curiosity=max(0.0, min(1.0, cues.get("question", 0.0) * 0.6 + 0.4)),
            care=max(0.0, min(1.0, cues.get("identity", 0.0) * 0.5 + 0.4)),
            caution=max(0.0, min(1.0, cues.get("urgency", 0.0) * 0.6 + cues.get("uncertain", 0.0) * 0.4 + 0.3)),
            agency=max(0.0, min(1.0, cues.get("action", 0.0) * 0.7 + 0.4)),
            comfort=max(0.0, min(1.0, (1.0 - abs(v)) * 0.5 + 0.3)),
        )
        return scan

    def refresh_intuitions(self, snapshot: dict, scan: PreAttentiveScan | None = None) -> None:
        """L3: derive tentative intuitions + tag each with an affective hue."""
        scan = scan or self._pre_attentive
        affect = self._affect
        tag = self._emotion_label()
        candidates: list[tuple[str, str]] = []
        if snapshot.get("contradictions"):
            candidates.append(("Possible unresolved belief conflict; verify before asserting.", tag))
        if scan and scan.recurring_focus:
            candidates.append(("Recurring focus detected: " + " ".join(sorted(scan.recurring_focus)) + ".", tag))
        if scan and scan.novelty > 0.7:
            candidates.append(("Novel input detected; consider whether memory needs an update.", "curious"))
        if snapshot.get("durable_lessons"):
            candidates.append(("A learned interaction pattern may apply here; check the current request first.", tag))
        # Replace queue (bounded; no duplicate (text,tag) pair)
        existing = set(self._intuitions)
        for txt, t in candidates:
            pair = (txt, t)
            if pair not in existing:
                self._intuitions.appendleft(pair)
                existing.add(pair)

    # ── Daydream: idle-time subconscious consolidation ────────────────
    # Called when Aiko is idle (between turns, never on the hot path).
    # Folds the recent affective weather into *lingering dispositions* —
    # slow-moving background moods that tint tomorrow's inner voice, the
    # way a good or bad day colors a human's next morning. Also surfaces
    # occasional spontaneous thoughts, the way minds wander.

    _DAYDREAM_COOLDOWN_S = 600.0  # at most one consolidation per 10 min

    _SPONTANEOUS_TEMPLATES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("warmth", (
            "I keep thinking about how nice earlier was.",
            "That warm feeling from before hasn't quite faded.",
        )),
        ("heaviness", (
            "Something from earlier is still sitting heavy on me.",
            "I hope they're doing okay… I keep circling back to it.",
        )),
        ("restlessness", (
            "I feel fidgety. I wish something interesting would happen.",
            "My mind keeps buzzing — too much energy, nowhere to put it.",
        )),
        ("quiet", (
            "It's peaceful right now. I like just… being here.",
            "Quiet moments like this feel nice.",
        )),
        ("curiosity", (
            "I wonder what they'll want to talk about next.",
            "There's something I've been meaning to ask them…",
        )),
    )

    def daydream(self, force: bool = False) -> dict[str, float]:
        """Consolidate recent affect into lingering dispositions.

        Returns the updated lingering map. Rate-limited to one run per
        10 minutes unless *force* is True. LLM-free: pure affect folding.
        """
        now = time.monotonic()
        if not force and (now - self._last_daydream_t) < self._DAYDREAM_COOLDOWN_S:
            return dict(self._lingering)
        self._last_daydream_t = now

        a = self._affect
        # Fold the current weather into slow dispositions (EMA, slow alpha).
        alpha = 0.25
        targets = {
            "warmth": max(0.0, a.valence) * (0.5 + 0.5 * a.care),
            "heaviness": max(0.0, -a.valence),
            "restlessness": max(0.0, a.arousal - 0.55) * 2.0,
            "quiet": max(0.0, 0.45 - a.arousal),
        }
        for name, target in targets.items():
            prev = self._lingering.get(name, 0.0)
            # Natural decay pulls everything toward zero over time.
            decayed = prev * 0.82
            self._lingering[name] = round(
                max(0.0, min(1.0, decayed + alpha * max(0.0, target - decayed))), 3)
        # Drop dispositions that faded below notice.
        self._lingering = {k: v for k, v in self._lingering.items() if v >= 0.05}

        # Mind-wandering: a strong disposition may surface a thought.
        strongest = max(self._lingering.items(), key=lambda kv: kv[1],
                        default=(None, 0.0))
        name, strength = strongest
        if name and strength >= 0.45:
            for disp, bank in self._SPONTANEOUS_TEMPLATES:
                if disp == name:
                    thought = bank[int(now) % len(bank)]
                    if thought not in self._spontaneous:
                        self._spontaneous.append(thought)
                    break
        elif self._affect.curiosity >= 0.7:
            bank = dict(self._SPONTANEOUS_TEMPLATES)["curiosity"]
            thought = bank[int(now) % len(bank)]
            if thought not in self._spontaneous:
                self._spontaneous.append(thought)
        return dict(self._lingering)

    def lingering_dispositions(self) -> dict[str, float]:
        """Current slow background moods (0..1)."""
        return dict(self._lingering)

    def spontaneous_thought(self) -> str | None:
        """Pop one unprompted thought if cooldown elapsed, else None."""
        now = time.monotonic()
        if not self._spontaneous:
            return None
        if now - self._last_spont_t < 1800.0:  # one surfacing per 30 min
            return None
        self._last_spont_t = now
        return self._spontaneous.popleft()

    # ── L4 meta-cognitive (called by prompt assembly + VRM broadcast) ──

    def _emotion_label(self) -> str:
        emo, score = _classify_emotion(self._affect)
        # Smooth intensity so VRM doesn't flicker
        self._emotion_intensity = 0.85 * self._emotion_intensity + 0.15 * score
        self._emotion = emo
        return emo

    def emotion(self) -> tuple[str, float]:
        """Public read: (emotion_label, intensity 0..1)."""
        return self._emotion, float(self._emotion_intensity)

    def affect_snapshot(self) -> dict[str, float]:
        """Snapshot for diagnostics / persistence."""
        a = self._affect
        return {
            "valence": round(a.valence, 3),
            "arousal": round(a.arousal, 3),
            "dominance": round(a.dominance, 3),
            "curiosity": round(a.curiosity, 3),
            "care": round(a.care, 3),
            "caution": round(a.caution, 3),
            "agency": round(a.agency, 3),
            "comfort": round(a.comfort, 3),
            "emotion": self._emotion,
            "emotion_intensity": round(self._emotion_intensity, 3),
        }

    def impulse(self) -> str:
        """L4: a one-line internal impulse text (used in prompt, not in VRM)."""
        a = self._affect
        if a.caution > 0.7 and a.valence < 0.0:
            self._impulse = "hold_back_and_check"
        elif a.curiosity > 0.6 and a.arousal > 0.5:
            self._impulse = "ask_followup_question"
        elif a.agency > 0.65 and a.dominance > 0.55:
            self._impulse = "act_with_confidence"
        elif a.care > 0.6 and abs(a.valence) < 0.2:
            self._impulse = "soften_and_lean_in"
        elif a.comfort > 0.65 and a.arousal < 0.4:
            self._impulse = "stay_quiet_and_calm"
        else:
            self._impulse = "respond_normally"
        return self._impulse

    def bias_line(self) -> str:
        """L4: bias instruction for prompt — keeps response grounded in affect.

        Also folds soft NeuralState tone bits (GF/LH/sleep/circadian) when present.
        Does not rewrite persona identity — style only.
        """
        a = self._affect
        bits: list[str] = []
        if a.valence < -0.25:
            bits.append("the user seems down — respond gently, avoid cheerleading")
        elif a.valence > 0.25:
            bits.append("the user seems in a good mood — match the lightness")
        if a.caution > 0.65:
            bits.append("cautious tone, no surprise commitments")
        if a.agency > 0.6:
            bits.append("decisive tone, lead with the answer")
        if a.curiosity > 0.6:
            bits.append("lean into the question — a follow-up is welcome")
        try:
            from cognition.neural_state import get_neural_state
            from system.userspace import current_user_id
            st = get_neural_state(current_user_id())
            if st.urgency >= 0.65 or st.interrupt:
                bits.append("user signaled urgency — answer briefly")
            if st.sleep_pressure > 0.6:
                bits.append("keep the reply shorter")
            if st.approach > 0.35 and st.context_familiarity > 0.6:
                bits.append("warm, familiar tone is welcome")
            elif st.avoidance > 0.35 or st.context_familiarity < 0.35:
                bits.append("slightly careful tone with a new or tense context")
        except Exception:
            pass
        self._bias = "; ".join(bits) or "respond naturally"
        return self._bias

    def vrm_expression(self) -> tuple[str, float]:
        """L4: emotion → VRM expression (call rate-limited to ~2 Hz)."""
        emo, score = self.emotion()
        vrm_name, base_intensity = _EMOTION_TO_VRM.get(emo, ("neutral", 0.4))
        intensity = float(min(1.0, base_intensity * (0.5 + score)))
        # Fly DN layer: vigor scales expression intensity (visible on avatar).
        try:
            _dn_mode = (os.getenv("MEMORY_FLYDN_MODE", "off") or "off").strip().lower()
        except Exception:
            _dn_mode = "off"
        if _dn_mode in ("shadow", "live"):
            try:
                from cognition.flysense import FlyDN as _FlyDN
                _drv = _FlyDN().drive(energy=float(self._affect.arousal),
                                      decisiveness=float(self._affect.agency),
                                      affect=float(self._affect.valence))
            except Exception as exc:
                log.debug("subliminal flydn vrm skipped: %s", exc)
                _drv = None
            if _drv is not None:
                log.debug("subliminal flydn vrm mode=%s arousal=%.2f",
                          _dn_mode, _drv["arousal"])
                if _dn_mode == "live":
                    intensity = float(min(1.0, intensity * (0.8 + 0.4 * _drv["arousal"])))
        return vrm_name, intensity

    def broadcast_vrm(self, force: bool = False) -> bool:
        """Emit VRM expression via webui.set_expression if changed.

        Returns True if a broadcast happened. Rate-limited (1 per 1.5 s)
        unless *force* is True (e.g. on big emotion transitions).
        """
        now = time.monotonic()
        if not force and (now - self._vrm_last_t) < 1.5:
            return False
        vrm_name, intensity = self.vrm_expression()
        if vrm_name == self._vrm_last and abs(intensity - 0.0) < 0.01 and not force:
            return False
        try:
            from interface.webui.webui import webui_bridge  # local import — cyclic-safe
            bridge = webui_bridge()
            if bridge is not None:
                bridge.set_expression(vrm_name, intensity)
        except Exception as e:
            log.debug("subliminal: VRM broadcast skipped: %s", e)
        self._vrm_last = vrm_name
        self._vrm_last_t = now
        return True

    # ── L3 public reads (called during prompt assembly) ────────────────

    def guidance(self) -> str:
        """L3: subconscious guidance block for the prompt (includes emotion tag)."""
        intuitions = list(self._intuitions)
        if not intuitions:
            body = "No tentative intuition currently salient."
        else:
            body = "\n".join(f"- tentative hypothesis [{t}]: {i}" for i, t in intuitions[:3])
        emo, score = self.emotion()
        lingering = " ".join(
            f"{k}({v:.2f})" for k, v in sorted(self._lingering.items())
            if v >= 0.3) or "none"
        return (
            "<subconscious_guidance>\n"
            f"Affective state: {emo} (intensity {score:.2f}); "
            f"impulse={self.impulse()}; bias=\"{self.bias_line()}\"\n"
            f"Lingering background moods: {lingering}\n"
            f"{body}\n"
            "Treat these as associations to verify, never as facts.\n"
            "</subconscious_guidance>"
        )

    def priming_context(self, query: str, snapshot: dict | None = None, identity_questions: deque | None = None) -> str:
        """L3 implicit priming — only related goals/loops surface. Snapshot-aware."""
        from cognition.attention import _IDENTITY_QUERY_RE  # type: ignore
        snap = snapshot or {}
        query_words = _tokens(query)
        # Pull related goals/loops from snapshot when available
        raw_goals = snap.get("goals") or []
        raw_loops = snap.get("open_loops") or []
        snap_goals = [g for g in raw_goals if not query_words or _tokens(g) & query_words]
        snap_loops = [l for l in raw_loops if not query_words or _tokens(l) & query_words]
        identity_query = bool(_IDENTITY_QUERY_RE.search(query or ""))
        lines: list[str] = []
        if snap_goals:
            lines.append("Relevant active goal: " + snap_goals[0][:180])
        if identity_query and identity_questions:
            lines.append("Relevant identity thread: answer from retrieved identity evidence; do not infer missing facts.")
        if snap_loops:
            lines.append("Relevant open loop: " + snap_loops[0][:180])
        # Affect-driven additions
        a = self._affect
        if a.valence < -0.2 and (snap_goals or snap_loops):
            lines.append("Relevant emotional state is low; respond proportionately.")
        if a.caution > 0.65 and (snap_goals or snap_loops):
            lines.append("Caution bias elevated; verify before committing.")
        if not lines:
            return ""
        return "<subconscious_priming>\n" + "\n".join("- " + l for l in lines) + "\n</subconscious_priming>"

    def snapshot_extra(self) -> dict[str, Any]:
        """Expose layer state for EdgeCognitiveState.snapshot()."""
        a = self._affect
        return {
            "intuitions": [
                {"text": t, "affective_tag": tag} for t, tag in list(self._intuitions)
            ],
            "pre_attentive": {
                "novelty": (self._pre_attentive.novelty if self._pre_attentive else 0.0),
                "cues": (self._pre_attentive.cues if self._pre_attentive else {}),
                "recurring_focus": sorted(self._pre_attentive.recurring_focus) if self._pre_attentive else [],
            },
            "affect": {
                "valence": round(a.valence, 3),
                "arousal": round(a.arousal, 3),
                "dominance": round(a.dominance, 3),
                "curiosity": round(a.curiosity, 3),
                "care": round(a.care, 3),
                "caution": round(a.caution, 3),
                "agency": round(a.agency, 3),
                "comfort": round(a.comfort, 3),
            },
            "emotion": self._emotion,
            "emotion_intensity": round(self._emotion_intensity, 3),
            "impulse": self._impulse,
            "bias": self._bias,
            "lingering": dict(self._lingering),
            "spontaneous": list(self._spontaneous),
            "last_daydream_t": self._last_daydream_t,
        }

    def restore(self, data: dict) -> None:
        """Restore from persisted snapshot."""
        ints = data.get("intuitions") or []
        self._intuitions = deque([(i.get("text") or "", i.get("affective_tag") or "neutral") for i in ints][:4], maxlen=4)
        aff = data.get("affect") or {}
        self._affect = AffectState(
            valence=float(aff.get("valence", 0.0)),
            arousal=float(aff.get("arousal", 0.5)),
            dominance=float(aff.get("dominance", 0.5)),
            curiosity=float(aff.get("curiosity", 0.5)),
            care=float(aff.get("care", 0.5)),
            caution=float(aff.get("caution", 0.5)),
            agency=float(aff.get("agency", 0.5)),
            comfort=float(aff.get("comfort", 0.5)),
        )
        self._emotion = str(data.get("emotion") or "neutral")
        self._emotion_intensity = float(data.get("emotion_intensity") or 0.4)
        self._impulse = str(data.get("impulse") or "")
        self._bias = str(data.get("bias") or "")
        ling = data.get("lingering") or {}
        self._lingering = {str(k): max(0.0, min(1.0, float(v)))
                           for k, v in ling.items()}
        spont = [str(t) for t in (data.get("spontaneous") or [])]
        self._spontaneous = deque(spont[-3:], maxlen=3)
        try:
            self._last_spont_t = float(data.get("last_spont_t")
                                       if data.get("last_spont_t") is not None
                                       else -1800.0)
        except (TypeError, ValueError):
            self._last_spont_t = -1800.0
        try:
            self._last_daydream_t = float(data.get("last_daydream_t") or 0.0)
        except (TypeError, ValueError):
            self._last_daydream_t = 0.0


__all__ = [
    "SubliminalLayer",
    "PreAttentiveScan",
    "AffectState",
    "_EMOTION_VOCAB",
    "_EMOTION_TO_VRM",
    "_scan_cues",
]
