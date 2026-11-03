"""Unit tests for subliminal daydream consolidation + spontaneous thoughts."""
import time

from cognition.subliminal import SubliminalLayer


def _layer():
    return SubliminalLayer()


def test_daydream_builds_warmth():
    s = _layer()
    s.scan("I love spending time talking with you, this is wonderful",
           {}, __import__("collections").deque())
    out = s.daydream(force=True)
    assert out.get("warmth", 0.0) > 0.0


def test_daydream_builds_heaviness():
    s = _layer()
    s.scan("I hate this, everything is sad and terrible and I am hurt",
           {}, __import__("collections").deque())
    out = s.daydream(force=True)
    assert out.get("heaviness", 0.0) > 0.0


def test_daydream_rate_limited():
    s = _layer()
    first = s.daydream(force=True)
    second = s.daydream(force=False)  # within cooldown -> same map
    assert first == second


def test_lingering_decays():
    from collections import deque
    s = _layer()
    s.scan("I love this wonderful day", {}, deque())
    s.daydream(force=True)
    before = s.lingering_dispositions().get("warmth", 0.0)
    assert before > 0.0
    # Many neutral-idle cycles: valence itself fades, then warmth follows.
    for _ in range(15):
        s.scan("ok", {}, deque())
        s.daydream(force=True)
    after = s.lingering_dispositions().get("warmth", 0.0)
    assert after < before


def test_daydream_emits_no_template_thoughts():
    # The subliminal is pure signal now: daydream folds dispositions only.
    # Idle pop-ups come from genuine inner-speech recall, not templates.
    s = _layer()
    assert not hasattr(s, "spontaneous_thought")
    s.scan("I love spending time talking with you, this is wonderful",
           {}, __import__("collections").deque())
    s.daydream(force=True)
    assert s.lingering_dispositions()  # dispositions still fold
    assert "spontaneous" not in s.snapshot_extra()


def test_guidance_mentions_lingering():
    s = _layer()
    s.scan("I love spending time talking with you, this is wonderful",
           {}, __import__("collections").deque())
    s.daydream(force=True)
    assert "Lingering background moods" in s.guidance()


def test_snapshot_restore_keeps_daydream_state():
    s = _layer()
    s.scan("I love spending time talking with you, this is wonderful",
           {}, __import__("collections").deque())
    s.daydream(force=True)
    snap = s.snapshot_extra()
    s2 = _layer()
    s2.restore(snap)
    assert s2.lingering_dispositions() == s.lingering_dispositions()


def test_restore_resets_future_monotonic_timestamps():
    future = time.monotonic() + 60.0
    s = _layer()
    s.restore({"last_daydream_t": future, "last_spont_t": future})
    assert s._last_daydream_t == 0.0
    assert not hasattr(s, "_last_spont_t")


def test_snapshot_has_no_spontaneous_state():
    s = _layer()
    s.restore({"spontaneous": ["A thought"], "last_spont_t": 123.0})
    snap = s.snapshot_extra()
    assert "spontaneous" not in snap
    assert "last_spont_t" not in snap
