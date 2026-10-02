"""Unit tests for the last-resort websearch net in AikoThink.chat()."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from cognition import think as think_module
from cognition.think import AikoThink


class FakeMemorize:
    def format_for_context(self, *a, **kw):
        return ""

    def persona_context(self):
        return ""


class FakeMemoryInner:
    def __init__(self):
        self._embedder = None


def _net_think() -> AikoThink:
    think = object.__new__(AikoThink)
    think._memorize = FakeMemorize()
    think._memorize_lock = threading.Lock()
    think._active_user_ids = set()
    think._active_users_lock = threading.Lock()
    think._last_chat_time = 0.0
    think._proactive_lock = threading.Lock()
    think._proactive_resting = False
    think._history = []
    think._history_lock = threading.RLock()
    think._reasoning = False
    think._get_speak = lambda: None
    think._current_system_prompt = lambda *a, **kw: "SYS"
    think._current_system_prompt_parts = lambda *a, **kw: ("SYS", "")
    return think


def test_websearch_net_block_formats_results(monkeypatch):
    think = _net_think()
    captured = {}

    def fake_search(query, n):
        captured["query"] = query
        captured["n"] = n
        return ([{"title": "PNE", "url": "https://pne.example", "content": "The Fair is on."}], None)

    from agentic.toolkit import websearch
    monkeypatch.setattr(websearch, "web_search", fake_search)

    block = think._websearch_net_block("check internet what PNE is")
    assert captured == {"query": "check internet what PNE is", "n": 3}
    assert "https://pne.example" in block
    assert block.startswith("1. PNE")


def test_websearch_net_block_empty_on_error_or_no_results(monkeypatch):
    think = _net_think()
    from agentic.toolkit import websearch

    monkeypatch.setattr(websearch, "web_search", lambda q, n: ([], "searxng down"))
    assert think._websearch_net_block("anything") == ""

    monkeypatch.setattr(websearch, "web_search", lambda q, n: ([], None))
    assert think._websearch_net_block("anything") == ""


@pytest.fixture()
def chat_env(monkeypatch):
    """Stub out everything chat() touches besides the net under test.

    Chained recall: memory first, then the entity-link/explicit knowledge
    hop — no shared mem_kb future on this path.
    """
    think = _net_think()

    class EdgeState:
        def prioritize_memories(self, *a, **kw):
            return []

        def situation_context(self, *a, **kw):
            return ""

        def metacognitive_context(self, *a, **kw):
            return ""

    from cognition import attention
    monkeypatch.setattr(attention, "for_identity", lambda uid: EdgeState())
    monkeypatch.setattr(think_module.bioclock, "current_datetime_block", lambda: "")
    monkeypatch.setattr(think, "_fetch_memory_only", lambda *a, **kw: [])
    monkeypatch.setattr(think, "_fetch_chained_knowledge", lambda *a, **kw: "")
    monkeypatch.setattr(think, "_websearch_full_block", lambda *a, **kw: "")
    monkeypatch.setattr(think, "_sanitize_history", lambda history: history)
    monkeypatch.setattr(think, "_store_async", lambda *a, **kw: None)
    monkeypatch.setattr(think, "_finalize_response", lambda user_input, resp, cb=None, already_emitted=None, _spoken_prefix="": resp)
    return think


def _stream_capturing(seen_systems):
    def fake_stream(trimmed, system=None, system_tail=None, token_callback=None, emit=None):
        seen_systems.append((system or "") + ("\n\n" + system_tail if system_tail else ""))
        return "ok"

    return fake_stream


def test_chat_runs_websearch_net_for_internet_asks(chat_env, monkeypatch):
    think = chat_env
    net_calls = []

    def fake_net(query, token_callback=None):
        net_calls.append(query)
        return "1. PNE\n   https://pne.example\n   The Fair."

    monkeypatch.setattr(think, "_websearch_net_block", fake_net)
    seen_systems = []
    monkeypatch.setattr(think, "_stream_response", _stream_capturing(seen_systems))

    result = think.chat("check internet to see what PNE is")
    assert result == "ok"
    assert net_calls == ["check internet to see what PNE is"]
    assert "<search_results query='check internet to see what PNE is'>" in seen_systems[0]
    assert "https://pne.example" in seen_systems[0]


def test_chat_skips_websearch_net_without_hint_words(chat_env, monkeypatch):
    think = chat_env
    net_calls = []
    monkeypatch.setattr(think, "_websearch_net_block", lambda q, token_callback=None: net_calls.append(q) or "x")

    def fail_stream(*a, **kw):
        raise AssertionError("stream should still be reached")

    monkeypatch.setattr(think, "_stream_response", fail_stream)
    try:
        think.chat("what do you think about minimalism")
    except AssertionError as exc:
        assert "stream" in str(exc)
    assert net_calls == []


def test_chat_net_failure_degrades_to_plain_chat(chat_env, monkeypatch):
    think = chat_env
    monkeypatch.setattr(think, "_websearch_net_block", lambda q, token_callback=None: "")

    def fake_stream(trimmed, system=None, system_tail=None, token_callback=None, emit=None):
        full = (system or "") + (system_tail or "")
        assert "<search_results" not in full
        return "plain reply"

    monkeypatch.setattr(think, "_stream_response", fake_stream)
    assert think.chat("check internet to see what PNE is") == "plain reply"


# --- Piece 1: chained recall (memory → link hop → conditional knowledge) ---

def _unmock_chained(think, monkeypatch):
    """Restore the real _fetch_chained_knowledge over the fixture mock,
    and let prioritize_memories pass memories through (the fixture mock
    wipes them, which would force the explicit-knowledge branch)."""
    monkeypatch.delattr(think, "_fetch_chained_knowledge")

    class PassThrough:
        def prioritize_memories(self, raw_input, memories):
            return memories

        def situation_context(self, *a, **kw):
            return ""

        def metacognitive_context(self, *a, **kw):
            return ""

    from cognition import attention
    monkeypatch.setattr(attention, "for_identity", lambda uid: PassThrough())


def test_chained_recall_skips_explicit_knowledge_when_memory_answers(chat_env, monkeypatch):
    think = chat_env
    _unmock_chained(think, monkeypatch)
    calls = {"explicit": 0}

    strong = [{"text": "Oppa likes ramen", "_recall_score": 5.0, "_reconstruction_confidence": "high"}]
    monkeypatch.setattr(think, "_fetch_memory_only", lambda *a, **kw: list(strong))
    monkeypatch.setattr(
        think, "_fetch_linked_knowledge",
        lambda memories, raw, limit=3: "<knowledge_context>\n<knowledge_chunk>linked chunk</knowledge_chunk>\n</knowledge_context>",
    )

    def fake_kb(*a, **kw):
        calls["explicit"] += 1
        return "<knowledge_context>\n<knowledge_chunk>explicit chunk</knowledge_chunk>\n</knowledge_context>"

    monkeypatch.setattr(think_module, "knowledge_context_for", fake_kb)
    seen = []
    monkeypatch.setattr(think, "_stream_response", _stream_capturing(seen))

    assert think.chat("what food do I like") == "ok"
    # Memory answered: the explicit vector search must not run.
    assert calls["explicit"] == 0
    # The cheap one-link-down hop still provides associative context.
    assert "linked chunk" in seen[0]
    assert "explicit chunk" not in seen[0]


def test_chained_recall_runs_explicit_knowledge_when_memory_cannot_answer(chat_env, monkeypatch):
    think = chat_env
    _unmock_chained(think, monkeypatch)
    calls = {"explicit": 0}

    monkeypatch.setattr(think, "_fetch_memory_only", lambda *a, **kw: [])
    monkeypatch.setattr(think, "_fetch_linked_knowledge", lambda *a, **kw: "")

    def fake_kb(*a, **kw):
        calls["explicit"] += 1
        return "<knowledge_context>\n<knowledge_chunk>explicit chunk</knowledge_chunk>\n</knowledge_context>"

    monkeypatch.setattr(think_module, "knowledge_context_for", fake_kb)
    seen = []
    monkeypatch.setattr(think, "_stream_response", _stream_capturing(seen))

    assert think.chat("what food do I like") == "ok"
    assert calls["explicit"] == 1
    assert "explicit chunk" in seen[0]


def test_chained_recall_memory_first_order(chat_env, monkeypatch):
    # Memory must be fetched before the knowledge hop so the hop and the
    # gate see the final (prioritized, weak-set-dropped) memories.
    think = chat_env
    _unmock_chained(think, monkeypatch)
    order = []

    def fake_mem(*a, **kw):
        order.append("memory")
        return []

    def fake_link(memories, raw, limit=3):
        order.append("link-hop")
        assert memories == []  # sees memory's result
        return ""

    monkeypatch.setattr(think, "_fetch_memory_only", fake_mem)
    monkeypatch.setattr(think, "_fetch_linked_knowledge", fake_link)
    monkeypatch.setattr(think_module, "knowledge_context_for", lambda *a, **kw: "")
    seen = []
    monkeypatch.setattr(think, "_stream_response", _stream_capturing(seen))

    think.chat("hello")
    assert order == ["memory", "link-hop"]


# --- Piece 1: webchat flag merge ---

def test_web_search_flag_runs_full_block(chat_env, monkeypatch):
    think = chat_env
    web_calls = []
    net_calls = []

    def fake_full(query, token_callback=None):
        web_calls.append(query)
        return "<search_results query='q'>\nAnswer ONLY using these search results:\n\n1. x\n</search_results>"

    monkeypatch.setattr(think, "_websearch_full_block", fake_full)
    monkeypatch.setattr(think, "_websearch_net_block", lambda q, token_callback=None: net_calls.append(q) or "")
    seen = []
    monkeypatch.setattr(think, "_stream_response", _stream_capturing(seen))

    assert think.chat("mars mission update", web_search=True) == "ok"
    assert web_calls == ["mars mission update"]
    assert "Answer ONLY using these search results" in seen[0]


def test_local_chat_never_pays_web_latency(chat_env, monkeypatch):
    think = chat_env
    web_calls = []

    def fake_full(query, token_callback=None):
        web_calls.append(query)
        return "<search_results />"

    monkeypatch.setattr(think, "_websearch_full_block", fake_full)
    seen = []
    monkeypatch.setattr(think, "_stream_response", _stream_capturing(seen))

    assert think.chat("mars mission update") == "ok"
    assert web_calls == []
    assert "<search_results" not in seen[0]


def test_no_web_prefix_in_prompt_or_history(chat_env, monkeypatch):
    # web_search is a flag, never a literal "/web" prefix in prompt/history.
    think = chat_env
    monkeypatch.setattr(think, "_websearch_full_block", lambda q, token_callback=None: "")
    seen_prompts = []
    seen_trimmed = []

    def fake_stream(trimmed, system=None, system_tail=None, token_callback=None, emit=None):
        seen_trimmed.append([dict(m) for m in trimmed])
        seen_prompts.append((system or "") + (system_tail or ""))
        return "ok"

    monkeypatch.setattr(think, "_stream_response", fake_stream)

    think.chat("mars mission update", web_search=True)
    full_prompt = seen_prompts[0]
    assert "/web" not in full_prompt
    user_turns = [m for m in seen_trimmed[0] if m.get("role") == "user"]
    assert user_turns and user_turns[-1]["content"] == "mars mission update"


def test_webchat_wrapper_sets_web_search_flag(chat_env, monkeypatch):
    think = chat_env
    captured = {}

    def fake_chat(user_input, **kwargs):
        captured.update(kwargs)
        return "web ok"

    monkeypatch.setattr(think, "chat", fake_chat)
    monkeypatch.setattr(think_module, "_is_personal_sharing", lambda text: False)

    assert think.webchat("latest mars news") == "web ok"
    assert captured.get("web_search") is True


def test_webchat_personal_sharing_skips_web_results(chat_env, monkeypatch):
    # Narration about Oppa's own experiences must not be answered from web.
    think = chat_env
    captured = {}

    def fake_chat(user_input, **kwargs):
        captured.update(kwargs)
        return "plain ok"

    monkeypatch.setattr(think, "chat", fake_chat)
    monkeypatch.setattr(think_module, "_is_personal_sharing", lambda text: True)

    assert think.webchat("I went hiking yesterday and it was amazing") == "plain ok"
    assert captured.get("web_search", False) is False


# --- Piece 1: link-hop bound + super-node exclusion ---

def test_fetch_linked_knowledge_excludes_super_nodes(chat_env, monkeypatch):
    think = chat_env
    from cognition.memory import narrative
    from cognition import knowledge

    monkeypatch.setattr(
        narrative, "seed_entities_from_memories",
        lambda memories, query="": ["oppa", "ramen", "tokyo"],
    )
    monkeypatch.setattr(think, "_super_node_entities", lambda: {"oppa"})
    captured = {}

    def fake_linked(seeds, limit=3, max_chars=2000):
        captured["seeds"] = list(seeds)
        captured["limit"] = limit
        return ""

    monkeypatch.setattr(knowledge, "linked_knowledge_for_entities", fake_linked)

    assert think._fetch_linked_knowledge([{"text": "x"}], "raw input") == ""
    # The super-node "oppa" is filtered from the hop seeds; limit is bounded.
    assert captured["seeds"] == ["ramen", "tokyo"]
    assert captured["limit"] == 3


def test_fetch_linked_knowledge_empty_when_no_seeds(chat_env, monkeypatch):
    think = chat_env
    from cognition.memory import narrative
    from cognition import knowledge

    monkeypatch.setattr(narrative, "seed_entities_from_memories", lambda memories, query="": [])
    called = []

    def fake_linked(*a, **kw):
        called.append(True)
        return "x"

    monkeypatch.setattr(knowledge, "linked_knowledge_for_entities", fake_linked)

    assert think._fetch_linked_knowledge([{"text": "x"}], "raw") == ""
    assert called == []


# --- Piece 1: linked_knowledge_for_entities unit behavior ---

def test_linked_knowledge_hop_respects_overlap_min_and_limit(monkeypatch):
    from cognition.knowledge import search as ksearch

    rows = [
        {"id": "c1", "text": "ramen in tokyo notes", "chunk_index": 0, "created_at": "2026-09-01",
         "entities": '["ramen", "tokyo"]', "status": "active",
         "title": "t1", "source": "s", "kind": "note", "doc_id": "d1"},
        {"id": "c2", "text": "ramen only", "chunk_index": 0, "created_at": "2026-09-02",
         "entities": '["ramen"]', "status": "active",
         "title": "t2", "source": "s", "kind": "note", "doc_id": "d2"},
        {"id": "c3", "text": "ramen tokyo guide", "chunk_index": 0, "created_at": "2026-09-03",
         "entities": '["ramen", "tokyo", "guide"]', "status": "active",
         "title": "t3", "source": "s", "kind": "note", "doc_id": "d3"},
    ]

    class FakeConn:
        def execute(self, *a, **kw):
            class Cur:
                def fetchall(self):
                    return rows
            return Cur()

        def close(self):
            pass

    monkeypatch.setattr(ksearch, "connect", lambda uid: FakeConn())
    monkeypatch.setattr(ksearch, "current_user_id", lambda: "u1")
    monkeypatch.setenv("MEMORY_CROSS_STORE_MIN_ENTITY_OVERLAP", "2")

    block = ksearch.linked_knowledge_for_entities(["ramen", "tokyo"], limit=1)
    # c2 shares only 1 entity (< 2) and is excluded; limit=1 keeps the best.
    assert "ramen tokyo guide" in block
    assert "ramen in tokyo notes" not in block
    assert "ramen only" not in block

    assert ksearch.linked_knowledge_for_entities([], limit=3) == ""
    assert ksearch.linked_knowledge_for_entities(["ramen", "tokyo"], limit=0) == ""
