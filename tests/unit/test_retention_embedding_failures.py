from types import SimpleNamespace

import pytest
from cognition.consolidate import retention


@pytest.mark.parametrize('batch', [[], [[1., 2.]], [[float('nan'), 1.], [1., 2.]], [[0., 0.], [0., 0.]]])
def test_malformed_embedding_batch_does_not_abort_retention(monkeypatch, batch):
    monkeypatch.setattr(retention, 'build_static_anchors', lambda *a: None)
    monkeypatch.setattr(retention, 'build_dynamic_anchors', lambda *a: None)
    monkeypatch.setattr(retention, 'entity_connectivity_weights', lambda *a: {})
    monkeypatch.setattr(retention, 'is_must_keep', lambda text: False)
    rows = [{'_text': 'User likes green tea.', 'entities': '[]'},
            {'_text': 'User reads books.', 'entities': '[]'}]
    memory = SimpleNamespace(embed_texts=lambda *a, **k: batch)
    kept, stats = retention.apply_retention_gate(memory, 'owner', rows)
    assert stats['candidates'] == 2
    assert len(kept) == stats['kept_candidates']
    assert all(row in rows for row in kept)
