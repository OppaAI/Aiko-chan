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

# ─────────────────────────────────────────────────────────────────────────────
# _fetch_memory_only timeout (latency minor)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchMemoryOnlyTimeout:
    def _timeout_think(self, search_fn, timeout=0.2):
        import cognition.think as think_mod
        think = object.__new__(AikoThink)
        think._get_memorize = lambda: SimpleNamespace(search=search_fn)
        think._recall_query = lambda x: x
        # Patch the timeout to a small value for the test
        self._orig_timeout = think_mod.MEMORY_RECALL_TIMEOUT
        think_mod.MEMORY_RECALL_TIMEOUT = timeout
        return think, think_mod

    def _restore(self, think_mod):
        import cognition.think as m
        m.MEMORY_RECALL_TIMEOUT = self._orig_timeout

    def test_slow_search_times_out_returns_empty(self):
        import time
        import cognition.think as think_mod
        def slow_search(*a, **kw):
            time.sleep(5)
            return [{"memory": "too late"}]
        think, mod = self._timeout_think(slow_search, timeout=0.2)
        try:
            t0 = time.time()
            out = think._fetch_memory_only("hello")
            elapsed = time.time() - t0
        finally:
            self._restore(mod)
        assert out == []
        assert elapsed < 2.0, f"took {elapsed:.2f}s, timeout did not fire"

    def test_fast_search_returns_results(self):
        import cognition.think as think_mod
        def fast_search(*a, **kw):
            return [{"memory": "quick", "_recall_score": 1.0}]
        think, mod = self._timeout_think(fast_search, timeout=5.0)
        try:
            out = think._fetch_memory_only("hello")
        finally:
            self._restore(mod)
        assert len(out) == 1
        assert out[0]["memory"] == "quick"
