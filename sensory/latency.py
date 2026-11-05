"""Per-turn latency instrumentation: typed input → first MioTTS audio.

A tiny first-write-wins timestamp tracker shared by the chat pipeline
(cognition.think), the TTS client (sensory.speak), and the boot path
(system.wakeup). Stages, in turn order:

    input_received     typed text arrived at route()/chat()
    intent_embedded    intent embedding done (route_vec)
    recall_completed   chained recall done (memory → entity-link → knowledge)
    conscience_inbound gate_respond done (route, pre-LLM)
    first_llm_token    first content token from the LLM stream
    first_sentence     first complete sentence parsed from the stream
    final_token        LLM stream complete
    local_review       _review_response done
    conscience_outbound gate_speak done (finalize, pre-emit)
    first_tts_synth    first successful MioTTS synthesis this turn
    first_audio        first audio playback started

`mark()` is first-write-wins within a turn: callers mark unconditionally and
only the first call sticks, so hot paths (e.g. _synthesize per chunk) stay
branch-free. `reset()` starts a new turn. `report()` renders the stage →
stage breakdown plus the headline input→first-audio number; it is logged
once, when first audio fires.
"""
from __future__ import annotations

import time
import threading
import contextvars
from functools import wraps

from system.log import get_logger

log = get_logger(__name__)

STAGES = (
    "input_received",
    "intent_embedded",
    "recall_completed",
    "conscience_inbound",
    "first_llm_token",
    "first_sentence",
    "final_token",
    "local_review",
    "conscience_outbound",
    "first_tts_synth",
    "first_audio",
)


class TurnLatency:
    """First-write-wins per-turn stage timestamps."""

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self.marks.clear()

    def mark(self, stage: str) -> None:
        with self._lock:
            if stage not in self.marks:
                self.marks[stage] = time.perf_counter()

    def elapsed(self, start: str, end: str) -> float | None:
        with self._lock:
            a, b = self.marks.get(start), self.marks.get(end)
        if a is None or b is None:
            return None
        return max(0.0, b - a)

    def report(self) -> str:
        """One-line stage→stage breakdown, e.g. for the log.

        Intervals are computed in timestamp order, not STAGES order:
        sentence streaming can start audio before final_token, so
        declaration order would produce negative intervals.
        """
        with self._lock:
            marks = dict(self.marks)
        if not marks:
            return "[latency] no marks this turn"
        ordered = sorted(marks.items(), key=lambda kv: kv[1])
        parts = []
        prev = None
        for stage, t in ordered:
            if prev is not None:
                parts.append(f"{prev[0]}→{stage}={t - prev[1]:.3f}s")
            prev = (stage, t)
        a, b = marks.get("input_received"), marks.get("first_audio")
        total = max(0.0, b - a) if a is not None and b is not None else None
        headline = f"input→first_audio={total:.3f}s" if total is not None else "input→first_audio=n/a"
        return "[latency] " + headline + (" | " + " ".join(parts) if parts else "")


_tracker = contextvars.ContextVar("aiko_turn_latency", default=None)
_route_depth = contextvars.ContextVar("aiko_latency_route_depth", default=0)
_chat_depth = contextvars.ContextVar("aiko_latency_chat_depth", default=0)


def current() -> TurnLatency:
    tracker = _tracker.get()
    if tracker is None:
        tracker = TurnLatency()
        _tracker.set(tracker)
    return tracker


def reset() -> None:
    # Allocate rather than clearing: copied playback contexts keep their own
    # previous-turn tracker even after the caller starts its next response.
    _tracker.set(TurnLatency())


def routed_call(method):
    @wraps(method)
    def run(*args, **kwargs):
        reset()
        token = _route_depth.set(_route_depth.get() + 1)
        try:
            return method(*args, **kwargs)
        finally:
            _route_depth.reset(token)
    return run


def chat_call(method):
    @wraps(method)
    def run(*args, **kwargs):
        if not _route_depth.get() and not _chat_depth.get():
            reset()
        token = _chat_depth.set(_chat_depth.get() + 1)
        try:
            return method(*args, **kwargs)
        finally:
            _chat_depth.reset(token)
    return run


def mark(stage: str) -> None:
    current().mark(stage)


def report() -> str:
    return current().report()


def elapsed(start: str, end: str) -> float | None:
    return current().elapsed(start, end)
