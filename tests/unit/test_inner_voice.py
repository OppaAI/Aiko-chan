"""Unit tests for the inner voice (conscious stream). Offline, no LLM."""
import time

import pytest

from cognition.inner_voice import InnerVoice


def _voice():
    return InnerVoice()


def test_observe_thanks_adds_thought():
    v = _voice()
    v.observe("thank you so much!", "You're welcome!",
              emotion="warm", intensity=0.7, impulse="respond_normally")
    block = v.prompt_block()
    assert "<inner_voice>" in block
    assert "thanks" in block.lower()


def test_observe_question_and_action_cues():
    v = _voice()
    v.observe("can you please build me a small website?",
              "Sure, here's what I'll do.",
              emotion="focused", intensity=0.6, impulse="act_with_confidence",
              cues={"question": 1.0, "action": 1.0})
    block = v.prompt_block().lower()
    assert "question" in block or "do something" in block


def test_recurring_focus_noticed():
    v = _voice()
    v.observe("tell me about ramen", "Ramen is great.",
              emotion="neutral", intensity=0.3,
              recurring_focus=frozenset({"ramen"}))
    assert "ramen" in v.prompt_block().lower()


def test_lingering_tint():
    v = _voice()
    v.observe("hi", "hello!", emotion="calm", intensity=0.3,
              lingering={"warmth": 0.8})
    assert "warm" in v.prompt_block().lower()


def test_bounded_memory():
    v = _voice()
    for i in range(30):
        v.observe(f"thank you {i}", "yw", emotion="warm", intensity=0.8)
    snap = v.snapshot()
    assert len(snap["thoughts"]) <= 6
    for t in snap["thoughts"]:
        assert len(t) <= 160


def test_template_rotation_avoids_parroting():
    v = _voice()
    v.observe("thanks", "yw", emotion="warm", intensity=0.8)
    first = v.latest()
    v.observe("thanks again", "yw", emotion="warm", intensity=0.8)
    second = v.latest()
    # rotation should give a different line for the repeated cue
    assert first != second


def test_snapshot_restore_roundtrip():
    v = _voice()
    v.observe("thank you!", "You're welcome!", emotion="warm", intensity=0.8)
    v.queue_aside("I keep thinking about how nice earlier was.")
    snap = v.snapshot()
    v2 = _voice()
    v2.restore(snap)
    assert v2.latest() == v.latest()
    assert "<inner_voice>" in v2.prompt_block()


def test_aside_cooldown():
    v = _voice()
    v.queue_aside("a wandering thought")
    assert v.maybe_aside() == "a wandering thought"
    v.queue_aside("another thought")
    # cooldown: second aside not due yet
    assert v.maybe_aside() is None


def test_restore_empty_is_safe():
    v = _voice()
    v.restore(None)
    v.restore({})
    assert "<inner_voice>" in v.prompt_block()


def test_restore_resets_future_aside_timestamp():
    v = _voice()
    v.restore({
        "last_aside_t": time.monotonic() + 60.0,
        "aside_queue": ["a restored thought"],
    })
    assert v.maybe_aside() == "a restored thought"
