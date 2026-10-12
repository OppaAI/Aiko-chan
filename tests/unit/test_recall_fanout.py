"""Chained recall fan-out (latency #5): the entity-link hop and the
gated explicit knowledge search run concurrently once memories are
known, and the explicit search reuses a speculatively pre-computed KB
vector instead of embedding in-leg.

Locked here:
  * gate open → both legs run; explicit receives the pre-computed vector
  * gate closed → explicit never submitted (speculative vector discarded)
  * speculative future failure → falls back to in-leg embed (vector=None)
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from cognition import think as think_module
from cognition.think import AikoThink


def _fanout_think() -> AikoThink:
    think = object.__new__(AikoThink)
    think._history = []
    think._history_lock = threading.RLock()
    think._get_memorize = lambda: SimpleNamespace(_mem=SimpleNamespace(_embedder=object()))
    think._fetch_linked_knowledge = (
        lambda memories, raw, limit=3: "<knowledge_context>\n<knowledge_chunk>linked</knowledge_chunk>\n</knowledge_context>"
    )
    return think


def _ok_future(value):
    f = Future()
    f.set_result(value)
    return f


def _weak_memories():
    # No usable memories → _memory_cannot_answer opens the gate.
    return []


def _strong_memories():
    return [{
        "text": "Oppa likes ramen",
        "_recall_score": 5.0,
        "_reconstruction_confidence": "high",
    }]


def test_explicit_search_reuses_speculative_vector(monkeypatch):
    think = _fanout_think()
    seen = {}

    def fake_kb(query, limit=None, max_chars=None, embedder=None, query_vector=None):
        seen["query_vector"] = query_vector
        return "<knowledge_context>\n<knowledge_chunk>explicit</knowledge_chunk>\n</knowledge_context>"

    monkeypatch.setattr(think_module, "knowledge_context_for", fake_kb)
    sentinel = object()
    out = think._fetch_chained_knowledge(
        "what is my learned note about X",
        _weak_memories(), None, know_limit=3,
        kb_vec_future=_ok_future(sentinel),
    )
    # Both legs ran and their blocks concatenated in order.
    assert "linked" in out
    assert "explicit" in out
    # The speculative vector was passed through — no in-leg re-embed.
    assert seen["query_vector"] is sentinel


def test_gate_closed_skips_explicit_search(monkeypatch):
    think = _fanout_think()
    calls = {"explicit": 0}

    def fake_kb(*a, **kw):
        calls["explicit"] += 1
        return "<knowledge_context>\n<knowledge_chunk>explicit</knowledge_chunk>\n</knowledge_context>"

    monkeypatch.setattr(think_module, "knowledge_context_for", fake_kb)
    out = think._fetch_chained_knowledge(
        "what food do I like",
        _strong_memories(), None, know_limit=3,
        kb_vec_future=_ok_future(object()),
    )
    assert calls["explicit"] == 0
    assert "linked" in out
    assert "explicit" not in out


def test_speculative_failure_falls_back_to_in_leg_embed(monkeypatch):
    think = _fanout_think()
    seen = {}

    def fake_kb(query, limit=None, max_chars=None, embedder=None, query_vector=None):
        seen["query_vector"] = query_vector
        return "<knowledge_context>\n<knowledge_chunk>explicit</knowledge_chunk>\n</knowledge_context>"

    monkeypatch.setattr(think_module, "knowledge_context_for", fake_kb)
    bad = Future()
    bad.set_exception(RuntimeError("embedder down"))
    out = think._fetch_chained_knowledge(
        "what is my learned note about X",
        _weak_memories(), None, know_limit=3,
        kb_vec_future=bad,
    )
    # Falls back to embedding inside knowledge_context_for (vector=None).
    assert seen["query_vector"] is None
    assert "explicit" in out
