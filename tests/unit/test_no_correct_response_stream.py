"""No _correct_response LLM rewrite (latency #10): the review flags are
logged to the reviews ring for observability, but the LLM rewrite was the
same model fixing its own output — unproven value for ~1-3s TTFT. The draft
is accepted as-is.

Locked here:
  * _finalize_response never calls the LLM correction (method removed)
  * review flags are still logged via _review_response
"""
from __future__ import annotations

import threading

import pytest

from cognition.think import AikoThink


@pytest.fixture()
def think():
    t = object.__new__(AikoThink)
    t._speak_lock = threading.Lock()
    # Review yields >=2 flags; previously this would have fired the LLM rewrite.
    t._review_response = lambda *a, **kw: {"flags": ["f1", "f2"]}
    t._get_speak = lambda: None
    t._get_memorize = lambda: None
    return t


def test_finalize_accepts_draft_without_llm_rewrite(think):
    # The _correct_response method no longer exists; _finalize_response
    # must not attempt an LLM correction.
    assert not hasattr(think, "_correct_response")
    out = think._finalize_response("hi", "draft text", sentences_gated=True)
    assert out == "draft text"
    out = think._finalize_response("hi", "draft text", sentences_gated=False)
    assert out == "draft text"
