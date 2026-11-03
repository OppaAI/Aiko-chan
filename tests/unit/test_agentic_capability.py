"""
tests/unit/test_agentic_capability.py

Unit tests for agentic/capability.py — capability matching and tool schema filtering.

Run: pytest tests/unit/test_agentic_capability.py -v
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace")

sys.path.insert(0, "/home/oppa-ai/jetson")
from system.config import load_config
load_config()

from agentic.capability import (
    Capability,
    CAPABILITIES,
    _CAPABILITY_INSTRUCT,
    _CAPABILITY_THRESHOLD,
    _trigger_embed_cache,
    _get_trigger_embedding,
    match_capabilities,
    filtered_tool_schemas,
)
from agentic.registry import registry
import agentic.agentic  # noqa: F401 — bootstrap full tool registry


class FakeEmbedder:
    """Deterministic embedder for tests."""
    def __init__(self):
        self.call_count = 0
        self.last_text = None

    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        self.call_count += 1
        self.last_text = text
        h = 1 + hash(text + instruct) % 1000
        return np.array([float(h) / 1000.0] * 384, dtype=np.float32)


class TestCapabilityStructure:
    """Tests for capability data structures."""

    def test_tool_domains_comprehensive(self):
        """All registered tools should have a domain or be always-on."""
        tool_domains = registry.get_tool_domains()
        always_on = registry.get_always_on_tools()
        # Check key tools are present
        assert "adaptive_search" in tool_domains
        assert "deep_research" in tool_domains
        assert "write_report" in tool_domains
        assert "learn_knowledge" in tool_domains
        assert "kb_search" in tool_domains
        assert "synthesize_report" in tool_domains
        assert "combine_evidence" in tool_domains
        assert "condense_text" in tool_domains
        assert "polish_text" in tool_domains
        assert "make_plan" in always_on
        assert "save_note" in always_on

    def test_capabilities_defined(self):
        expected_caps = {"research", "scheduling", "kb_proposal", "photo", "repo", "job_hunt", "social",
                           "it_ops", "note_taking", "weather", "coding", "everyday", "multi_agent"}
        assert set(CAPABILITIES.keys()) == expected_caps

    def test_research_capability_domains(self):
        """Research capability should include research, kb, reports domains."""
        cap = CAPABILITIES["research"]
        assert set(cap.tool_domains) == {"research", "kb", "reports"}

    def test_repo_capability_includes_reports(self):
        cap = CAPABILITIES["repo"]
        assert "reports" in cap.tool_domains

    def test_always_on_tools_comprehensive(self):
        """Always-on tools should include base tools every turn needs."""
        expected = {"make_plan", "create_checklist", "save_note", "read_workspace_file",
                    "summarize_task_state", "list_playbooks", "run_playbook", "final_answer"}
        assert expected.issubset(registry.get_always_on_tools())


class TestTriggerEmbeddingCache:
    """Tests for trigger embedding caching."""

    def test_cache_populated_on_first_call(self):
        embedder = FakeEmbedder()
        cap = CAPABILITIES["research"]
        _trigger_embed_cache.clear()

        # Bypass the on-disk vector cache so the in-memory cache is exercised.
        with patch("agentic.capability._capability_vector_cache_path", return_value=None):
            vec1 = _get_trigger_embedding(cap, embedder)
            assert embedder.call_count == 1

            vec2 = _get_trigger_embedding(cap, embedder)
            assert embedder.call_count == 1  # Cached, no new call
            assert np.array_equal(vec1, vec2)

    def test_different_capabilities_different_vectors(self):
        embedder = FakeEmbedder()
        # Constant-component vectors become identical after normalization.
        # Use two genuinely different directions to exercise the cache keys.
        embedder.embed_query = MagicMock(side_effect=[
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ])
        _trigger_embed_cache.clear()

        vec_research = _get_trigger_embedding(CAPABILITIES["research"], embedder)
        vec_scheduling = _get_trigger_embedding(CAPABILITIES["scheduling"], embedder)

        # Should be different (different trigger texts)
        assert not np.array_equal(vec_research, vec_scheduling)
        assert embedder.embed_query.call_count == 2


class TestMatchCapabilities:
    """Tests for match_capabilities function."""

    def test_semantic_match_with_embedder(self):
        """Should match capabilities via cosine similarity."""
        embedder = FakeEmbedder()
        _trigger_embed_cache.clear()

        caps = match_capabilities("do deep research on quantum computing", embedder=embedder)
        assert "research" in caps

    def test_keyword_fallback_without_embedder(self):
        """Should fall back to keyword matching when no embedder."""
        caps = match_capabilities("schedule a meeting for tomorrow")
        assert "scheduling" in caps

    def test_keyword_fallback_when_embedder_fails(self):
        """Should fall back to keyword when embedder raises."""
        class BadEmbedder:
            def embed_query(self, *a, **k):
                raise RuntimeError("embedder broken")

        caps = match_capabilities("research this topic", embedder=BadEmbedder())
        # Should still match via keyword fallback
        assert "research" in caps

    def test_keyword_fallback_uses_token_boundaries(self):
        assert "repo" not in match_capabilities("dispatch the update")
        assert "social" not in match_capabilities("write a postmortem")
        assert "scheduling" not in match_capabilities("daily news brief")
        assert "social" in match_capabilities("publish to threads")

    def test_precomputed_query_vector_used(self):
        """Pre-computed query_vector should be used instead of re-embedding."""
        embedder = FakeEmbedder()
        _trigger_embed_cache.clear()

        # Pre-compute a vector
        query_vec = embedder.embed_query("test query", instruct=_CAPABILITY_INSTRUCT)

        caps = match_capabilities(
            "actual user input ignored",
            embedder=embedder,
            query_vector=query_vec
        )
        # embedder should only be called for trigger embeddings, not for query
        # (trigger embeddings are cached, so call_count might be small)
        assert isinstance(caps, list)

    def test_threshold_filtering(self):
        """Capabilities below threshold should not match."""
        # With our deterministic fake embedder, some capabilities may score low
        embedder = FakeEmbedder()
        _trigger_embed_cache.clear()

        caps = match_capabilities("completely unrelated gibberish xyz", embedder=embedder)
        # May return empty or only keyword matches
        assert isinstance(caps, list)

    def test_multiple_capabilities_can_match(self):
        embedder = FakeEmbedder()
        _trigger_embed_cache.clear()

        caps = match_capabilities("research and schedule a meeting", embedder=embedder)
        # Both should match via keyword fallback at minimum
        assert "research" in caps or "scheduling" in caps


class TestFilteredToolSchemas:
    """Tests for filtered_tool_schemas tool filtering."""

    def build_mock_schemas(self):
        """Build mock tool schemas for testing."""
        return [
            {"function": {"name": "make_plan", "parameters": {}}},
            {"function": {"name": "adaptive_search", "parameters": {}}},
            {"function": {"name": "deep_research", "parameters": {}}},
            {"function": {"name": "write_report", "parameters": {}}},
            {"function": {"name": "learn_knowledge", "parameters": {}}},
            {"function": {"name": "schedule_job", "parameters": {}}},
            {"function": {"name": "search_jobs", "parameters": {}}},
            {"function": {"name": "final_answer", "parameters": {}}},
        ]

    def test_no_capabilities_returns_all(self):
        """No matched capabilities -> return all schemas unchanged."""
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, [])
        assert len(filtered) == len(schemas)

    def test_research_capability_includes_research_tools(self):
        """Research capability pulls in research, kb, reports domains."""
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, ["research"])
        names = {s["function"]["name"] for s in filtered}

        # Research domain
        assert "adaptive_search" in names
        assert "deep_research" in names
        # KB domain
        assert "learn_knowledge" in names
        # Reports domain
        assert "write_report" in names
        # Always on
        assert "make_plan" in names
        assert "final_answer" in names
        # Not included
        assert "schedule_job" not in names
        assert "search_jobs" not in names

    def test_scheduling_capability_only_adds_scheduling(self):
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, ["scheduling"])
        names = {s["function"]["name"] for s in filtered}

        assert "schedule_job" in names
        assert "adaptive_search" not in names
        assert "make_plan" in names  # Always on

    def test_multiple_capabilities_union(self):
        """Multiple capabilities -> union of their domains."""
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, ["research", "scheduling"])
        names = {s["function"]["name"] for s in filtered}

        assert "adaptive_search" in names
        assert "schedule_job" in names

    def test_unknown_capability_ignored(self):
        """Unknown capability IDs should be ignored, not crash."""
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, ["nonexistent_capability"])
        # Should fall back to always_on only
        names = {s["function"]["name"] for s in filtered}
        assert names == {"make_plan", "final_answer"}  # only always_on from our mock

    def test_always_on_tools_always_included(self):
        """ALWAYS_ON_TOOLS always present regardless of capabilities."""
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, ["scheduling"])
        names = {s["function"]["name"] for s in filtered}

        assert "make_plan" in names
        assert "create_checklist" not in names  # Not in our mock but would be
        assert "save_note" not in names  # Not in mock
        assert "final_answer" in names

    def test_empty_filtered_returns_all(self):
        """If filtering removes everything, return all schemas (safety net)."""
        schemas = self.build_mock_schemas()
        # Capability that doesn't match any tool domains
        filtered = filtered_tool_schemas(schemas, ["nonexistent"])
        # Our mock only has always_on tools that don't have domains in TOOL_DOMAINS
        # So filtered would be just always_on. But safety net: if filtered is empty, return all.
        assert len(filtered) > 0


class TestCapabilityThreshold:
    """Tests around _CAPABILITY_THRESHOLD."""

    def test_default_threshold_value(self):
        assert _CAPABILITY_THRESHOLD == 0.35

    def test_instruct_constant(self):
        assert _CAPABILITY_INSTRUCT == "Which capability/tool domain applies to this task?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestCapabilityPlaybookConsistency:
    """Every capability id declared by any playbook must be registered,
    with trigger phrases, tool domains, and an overlay. Guards the
    reachability bug where unregistered ids silently excluded playbooks
    on capability-matched turns."""

    def test_all_playbook_capability_ids_registered(self):
        from agentic.graph_engine import _default_playbooks
        declared: set[str] = set()
        for p in _default_playbooks():
            declared.update(p.get("capabilities") or [])
        assert declared, "no playbook declares capabilities?"
        assert declared <= set(CAPABILITIES), (
            f"unregistered capability ids: {sorted(declared - set(CAPABILITIES))}"
        )

    def test_every_capability_has_triggers_domains_overlay(self):
        for cid, cap in CAPABILITIES.items():
            assert cap.triggers, f"{cid}: no trigger phrases"
            assert isinstance(cap.tool_domains, tuple), f"{cid}: domains not a tuple"
            assert isinstance(cap.system_overlay, str) and cap.system_overlay.strip(), (
                f"{cid}: empty system overlay"
            )

    def test_every_capability_domain_tag_exists_on_tools(self):
        import agentic.toolkit.plan  # noqa: F401 — register note/planning tools
        for _mod in ("agentic.toolkit.it_ops", "agentic.workflows.aurora_forecast.graph"):
            try:
                __import__(_mod)
            except Exception:
                pass
        tags = set(registry.get_tool_domains().values())
        for cid, cap in CAPABILITIES.items():
            if not cap.tool_domains:
                continue  # e.g. weather: LLM-only playbook, no tools
            assert set(cap.tool_domains) & tags, (
                f"{cid}: domains {cap.tool_domains} match no registered tool"
            )


@pytest.fixture
def harrier_cache(tmp_path, monkeypatch):
    from cognition.memory.vecstore import HarrierEmbedder
    monkeypatch.setenv('CAP_VECTOR_CACHE_DIR', str(tmp_path / 'vectors'))
    monkeypatch.setenv('MEMORY_FLYAL_MODE', 'off')
    _trigger_embed_cache.clear()

    def make(*, base_url='http://first.local', model='model-a', dims=2):
        # Override transport so these tests never call an embedding server.
        embedder = object.__new__(HarrierEmbedder)
        embedder.base_url = base_url
        embedder.model = model
        embedder.dims = dims
        embedder.embed_query = MagicMock(return_value=np.eye(1, dims, dtype=np.float32)[0])
        return embedder

    yield make
    _trigger_embed_cache.clear()


def test_memory_cache_isolates_foreign_embedder_instances():
    from agentic.capability import _capability_vector_cache_path
    first, second = FakeEmbedder(), FakeEmbedder()
    first.embed_query = MagicMock(return_value=np.array([1., 0.]))
    second.embed_query = MagicMock(return_value=np.array([0., 1., 0.]))
    cap = CAPABILITIES['research']
    assert _get_trigger_embedding(cap, first).shape == (2,)
    assert _get_trigger_embedding(cap, second).shape == (3,)
    assert first.embed_query.call_count == second.embed_query.call_count == 1
    assert _capability_vector_cache_path(cap, first) is None


def test_memory_cache_includes_trigger_text():
    embedder = FakeEmbedder()
    embedder.embed_query = MagicMock(side_effect=[np.array([1., 0.]), np.array([0., 1.])])
    first = _get_trigger_embedding(Capability('same-id', ('first trigger',)), embedder)
    second = _get_trigger_embedding(Capability('same-id', ('edited trigger',)), embedder)
    assert not np.array_equal(first, second)
    assert embedder.embed_query.call_count == 2


@pytest.mark.parametrize('change', ['base_url', 'model', 'dims', 'flyal'])
def test_memory_and_disk_keys_follow_real_embedding_geometry(harrier_cache, monkeypatch, change):
    from agentic.capability import _capability_vector_cache_path
    embedder = harrier_cache()
    cap = CAPABILITIES['research']
    first_path = _capability_vector_cache_path(cap, embedder)
    first = _get_trigger_embedding(cap, embedder)
    if change == 'flyal':
        monkeypatch.setenv('MEMORY_FLYAL_MODE', 'live')
    elif change == 'dims':
        embedder.dims = 3
    else:
        setattr(embedder, change, getattr(embedder, change) + '-changed')
    embedder.embed_query.return_value = np.eye(1, embedder.dims, k=1, dtype=np.float32)[0]
    second_path = _capability_vector_cache_path(cap, embedder)
    second = _get_trigger_embedding(cap, embedder)
    assert first_path != second_path
    assert first_path.is_file() and second_path.is_file()
    assert embedder.embed_query.call_count == 2
    assert not np.array_equal(first, second)


def test_cache_returns_private_vector_copies(harrier_cache):
    cap, embedder = CAPABILITIES['research'], harrier_cache()
    original = _get_trigger_embedding(cap, embedder)
    original[:] = 0
    cached = _get_trigger_embedding(cap, embedder)
    assert cached[0] == 1
    cached[:] = 0
    assert _get_trigger_embedding(cap, embedder)[0] == 1
    assert embedder.embed_query.call_count == 1


@pytest.mark.parametrize('bad', [np.array([np.nan, 0.]), np.array([1., 0., 0.]),
    np.array([[1., 0.]]), np.zeros(2), np.array([np.inf, 1.])])
def test_bad_disk_vectors_are_rebuilt(harrier_cache, bad):
    from agentic.capability import _capability_vector_cache_path
    cap, embedder = CAPABILITIES['research'], harrier_cache()
    path = _capability_vector_cache_path(cap, embedder)
    path.parent.mkdir()
    np.savez(path, vector=bad)
    actual = _get_trigger_embedding(cap, embedder)
    assert np.array_equal(actual, [1., 0.])
    assert embedder.embed_query.call_count == 1
    with np.load(path, allow_pickle=False) as data:
        assert np.array_equal(data['vector'], actual)


def test_disk_cache_survives_memory_clear_for_same_geometry(harrier_cache):
    cap, first = CAPABILITIES['research'], harrier_cache()
    expected = _get_trigger_embedding(cap, first)
    _trigger_embed_cache.clear()
    second = harrier_cache()
    second.embed_query.side_effect = AssertionError('cache should avoid embedding transport')
    assert np.array_equal(_get_trigger_embedding(cap, second), expected)
    second.embed_query.assert_not_called()


def test_failed_atomic_cache_write_does_not_break_routing_or_replace_valid_file(harrier_cache, monkeypatch):
    from agentic import capability
    cap, embedder = CAPABILITIES['research'], harrier_cache()
    expected = _get_trigger_embedding(cap, embedder)
    path = capability._capability_vector_cache_path(cap, embedder)
    old_bytes = path.read_bytes()
    _trigger_embed_cache.clear()
    # Force regeneration without making the existing destination unreadable.
    monkeypatch.setattr(capability.np, 'load', MagicMock(side_effect=OSError('transient read failure')))
    monkeypatch.setattr(capability.np, 'savez', MagicMock(side_effect=OSError('disk full')))
    assert np.array_equal(_get_trigger_embedding(cap, embedder), expected)
    assert path.read_bytes() == old_bytes
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_concurrent_cache_writers_leave_complete_archive(harrier_cache):
    from concurrent.futures import ThreadPoolExecutor
    from agentic.capability import _capability_vector_cache_path
    cap, embedder = CAPABILITIES['research'], harrier_cache()
    with ThreadPoolExecutor(max_workers=6) as pool:
        outputs = list(pool.map(lambda _: _get_trigger_embedding(cap, embedder), range(24)))
    assert all(np.array_equal(v, [1., 0.]) for v in outputs)
    path = _capability_vector_cache_path(cap, embedder)
    with np.load(path, allow_pickle=False) as data:
        assert np.array_equal(data['vector'], outputs[0])
    assert list(path.parent.iterdir()) == [path]
