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

    def reset(self) -> None:
        self.marks.clear()

    def mark(self, stage: str) -> None:
        if stage not in self.marks:
            self.marks[stage] = time.perf_counter()

    def elapsed(self, start: str, end: str) -> float | None:
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
        if not self.marks:
            return "[latency] no marks this turn"
        ordered = sorted(self.marks.items(), key=lambda kv: kv[1])
        parts = []
        prev = None
        for stage, t in ordered:
            if prev is not None:
                parts.append(f"{prev[0]}→{stage}={t - prev[1]:.3f}s")
            prev = (stage, t)
        total = self.elapsed("input_received", "first_audio")
        headline = f"input→first_audio={total:.3f}s" if total is not None else "input→first_audio=n/a"
        return "[latency] " + headline + (" | " + " ".join(parts) if parts else "")


_current = TurnLatency()


def reset() -> None:
    _current.reset()


def mark(stage: str) -> None:
    _current.mark(stage)


def report() -> str:
    return _current.report()


def elapsed(start: str, end: str) -> float | None:
    return _current.elapsed(start, end)
