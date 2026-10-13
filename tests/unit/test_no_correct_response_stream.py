"""Skip _correct_response when sentence streaming already spoke the draft
(latency #10): a second LLM call after the stream only diverges the text
record from the audio the user already heard. The review flags are still
logged to the reviews ring; only the LLM rewrite is skipped.

Locked here:
  * sentences_gated=True -> _correct_response NOT called (draft accepted)
  * sentences_gated=False -> _correct_response called (existing behavior)
"""
from __future__ import annotations

import threading

import pytest

from cognition.think import AikoThink


@pytest.fixture()
def think():
    t = object.__new__(AikoThink)
    t._speak_lock = threading.Lock()
    # Review yields >=2 flags so _correct_response would normally fire.
    t._review_response = lambda *a, **kw: {"flags": ["f1", "f2"]}
    t._get_speak = lambda: None
    t._get_memorize = lambda: None
    return t


def test_sentence_streamed_path_skips_llm_correction(think):
    calls = []
    think._correct_response = lambda *a, **kw: calls.append(1) or "corrected"
    out = think._finalize_response("hi", "draft text", sentences_gated=True)
    assert calls == []
    assert out == "draft text"


def test_non_streamed_path_still_corrects(think):
    calls = []
    think._correct_response = lambda *a, **kw: calls.append(1) or "corrected"
    out = think._finalize_response("hi", "draft text", sentences_gated=False)
    assert calls == [1]
    assert out == "corrected"
