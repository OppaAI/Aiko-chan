from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from cognition.consolidate import lifecycle as module


def test_partial_monthly_write_keeps_sources_and_remains_retryable(monkeypatch):
    daily = [{'id': 'day1', 'memory': '[2026-01-01] User likes tea.', 'pinned': 1},
             {'id': 'day2', 'memory': '[2026-01-02] User likes books.', 'pinned': 1}]
    stats = dict(must_keep=2, candidates=0, kept_candidates=0, dropped_candidates=0)
    monkeypatch.setattr(module, 'CONSOLIDATION_ENABLED', True)
    monkeypatch.setattr(module, 'CONSOLIDATION_MIN_MEMS', 1)
    monkeypatch.setattr(module, '_load_state', lambda uid: {})
    save_state = Mock()
    monkeypatch.setattr(module, '_save_state', save_state)
    monkeypatch.setattr(module, 'promote_journal_fragments', lambda *a: ([], 0))
    monkeypatch.setattr(module, 'apply_retention_gate', lambda *a: (a[-1], stats))
    items = [{'fact': 'User likes tea.', 'source_ids': ['day1']},
             {'fact': 'User likes books.', 'source_ids': ['day2']}]
    monkeypatch.setattr(module, 'extract_monthly_facts_chunk', lambda *a: items)
    monkeypatch.setattr(module, 'merge_monthly_facts', lambda *a: items)
    from cognition.consolidate import journal
    monkeypatch.setattr(journal, 'get_between', lambda *a, **k: [])
    from cognition import attention
    monkeypatch.setattr(attention, 'for_identity', Mock(side_effect=RuntimeError('disabled in fixture')))
    memory = SimpleNamespace(get_between=lambda *a, **k: daily,
                             add_raw=Mock(side_effect=['monthly1', OSError('disk failure')]), delete=Mock())
    result = module.maybe_run_consolidation(memory, datetime(2026, 2, 2, tzinfo=timezone.utc), 'owner')
    assert result['reason'] == 'partial_write'
    assert result['facts_written'] == 1
    memory.delete.assert_not_called()
    save_state.assert_not_called()


def test_daily_journal_scrubs_credentials_and_keeps_stable_id(tmp_path, monkeypatch):
    from cognition.consolidate import journal
    from cognition.memory.vecstore import initialize_store_db
    path = tmp_path / 'journal.db'
    monkeypatch.setattr(journal, 'connect', lambda uid=None: initialize_store_db(path, journal._DDL, user_id=uid, vector=False))
    date = datetime(2026, 1, 1)
    first = journal.pin_daily_journal('Contact owner@example.org api_key=secret-value', date, user_id='owner')
    second = journal.pin_daily_journal('Contact owner@example.org password=another-secret', date, user_id='owner')
    assert first == second
    rows = journal.get_between(date, datetime(2026, 1, 2), user_id='owner')
    assert 'another-secret' not in rows[0]['body']
    assert 'owner@example.org' in rows[0]['body']
    assert not journal.get_between(date, datetime(2026, 1, 2), user_id='other')


def test_lesson_consumption_preserves_new_lessons_arriving_during_consolidation():
    from cognition.attention import EdgeCognitiveState
    state = EdgeCognitiveState('consolidation-fixture')
    state._lessons.extend(['Saved lesson', 'New lesson'])
    assert state.consume_lessons(expected=['Saved lesson']) == ['Saved lesson']
    assert state.consume_lessons() == ['New lesson']
