"""Regression coverage for runtime failures found in the repository audit."""
from unittest.mock import Mock

import pytest


def test_spreading_activation_has_bounded_weighted_hops(monkeypatch):
    from cognition.memory import entity
    monkeypatch.setattr(entity, 'MEMORY_SPREADING_ENABLED', True)
    edges = [('a', 'b', 0.5), ('b', 'c', 1.0), ('c', 'd', 1.0), ('a', 'bad', float('nan'))]
    result = entity.spread_activation(['a'], edges, max_depth=2, decay=0.6, min_strength=0.1)
    assert result == pytest.approx({'a': 1.0, 'b': 0.3, 'c': 0.18})
    assert entity.spread_activation(['b'], [('a', 'b', 1.0)], max_depth=1)['a'] > 0


def test_attention_persist_replaces_shared_debounce_timer(monkeypatch):
    from cognition import attention
    monkeypatch.setattr(attention, 'EDGE_COGNITION_PERSIST', True)
    monkeypatch.setattr(attention, '_persist_timer', None)
    monkeypatch.setattr(attention, '_persist_pending', set())
    timer = Mock()
    monkeypatch.setattr(attention.threading, 'Timer', Mock(return_value=timer))
    state = attention.EdgeCognitiveState.__new__(attention.EdgeCognitiveState)
    state._identity = 'audit-owner'
    state.persist()
    state.persist()
    assert attention._persist_pending == {'audit-owner'}
    assert attention._persist_timer is timer
    assert timer.start.call_count == 2
    timer.cancel.assert_called_once()


def test_invalid_workflow_spec_falls_back_to_config(tmp_path, monkeypatch):
    from agentic.workflows.common import spec, config
    (tmp_path / 'spec.json').write_text('{}')
    monkeypatch.setattr(config, 'load_workflow_config', lambda _: {'name': 'fallback'})
    sentinel = object()
    fallback = Mock(return_value=sentinel)
    monkeypatch.setattr(spec, 'coerce_config_to_spec', fallback)
    assert spec.load_spec_for_workflow(tmp_path, graph_id='audit', name='Audit', goal='Check') is sentinel
    fallback.assert_called_once()
