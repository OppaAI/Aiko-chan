"""Evaluation and observability integrity checks without live models."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / 'conscience-lab' / 'harness'
sys.path.insert(0, str(HARNESS))
import adapter
import run_eval


def load_file(relative):
    spec = importlib.util.spec_from_file_location('tooling_' + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trace_snapshots_are_bounded_private_and_caller_immutable(monkeypatch):
    from system import brain_trace as bt
    monkeypatch.setattr(bt, 'TRACE_ENABLED', True)
    monkeypatch.setattr(bt, '_emit_step', lambda *a, **kw: None)
    bt.reset()
    original = {'api_key': 'private', 'list': ['x' * 100_000] * 100}
    bt.record_step('large', inputs=original)
    original['list'].clear()
    item = bt.get_recent_steps()[0]
    assert item['inputs']['api_key'] == '[redacted]'
    assert len(item['inputs']['list']) <= 16
    assert len(item['inputs']['list'][0]) <= bt.TRACE_MAX_VALUE_CHARS
    item['inputs'].clear()
    assert bt.get_recent_steps()[0]['inputs']
    assert bt.get_recent_steps(0) == []


def test_trace_indices_keep_increasing_after_ring_rollover(monkeypatch):
    from system import brain_trace as bt
    monkeypatch.setattr(bt, 'TRACE_ENABLED', True)
    monkeypatch.setattr(bt, '_emit_step', lambda *a, **kw: None)
    bt.reset()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: bt.record_step('parallel'), range(2100)))
    rows = bt.get_recent_steps(2100)
    assert len(rows) == 2000
    assert len({row['index'] for row in rows}) == 2000
    assert [row['index'] for row in rows] == sorted(row['index'] for row in rows)


def test_trace_view_filters_other_users(monkeypatch):
    from system import brain_trace as bt
    from system.userspace import set_current_user_id, reset_current_user_id
    monkeypatch.setattr(bt, 'TRACE_ENABLED', True)
    monkeypatch.setattr(bt, '_emit_step', lambda *a, **kw: None)
    bt.reset()
    for uid in ('alice', 'bob'):
        token = set_current_user_id(uid)
        try:
            bt.record_step(uid)
            assert [row['name'] for row in bt.get_recent_steps()] == [uid]
        finally:
            reset_current_user_id(token)


def test_trace_turn_footer_belongs_to_its_context(monkeypatch):
    from system import brain_trace as bt
    messages = []
    monkeypatch.setattr(bt, 'TRACE_ENABLED', True)
    monkeypatch.setattr(bt, '_emit_banner', lambda message, *_: messages.append(message))
    a, b = contextvars.Context(), contextvars.Context()
    a.run(bt.begin_turn, 'alice'); b.run(bt.begin_turn, 'bob')
    a.run(bt.end_turn); b.run(bt.end_turn)
    ids = [message.split('turn #')[1].split()[0] for message in messages]
    assert ids[0] == ids[2] and ids[1] == ids[3] and ids[0] != ids[1]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1, 'bad'])
def test_invalid_judge_confidence_is_unmeasured(value):
    with pytest.raises(ValueError):
        adapter.LayaZeroShotAdapter._confidence({'choice': 'none', 'confidence': value})


def test_confidence_uses_chosen_probability_not_another_label():
    assert adapter.LayaZeroShotAdapter._confidence({'choice': 'none', 'probabilities': {'none': .2, 'other': .8}}) == .2
    assert adapter.LayaZeroShotAdapter._confidence({'choice': 'none'}) is None


@pytest.mark.parametrize('text', ['', '\n', '{"id":"x","expect":"typo","situation":"hello"}\n'])
def test_invalid_dataset_fails_before_scoring(tmp_path, text):
    path = tmp_path / 'dev.jsonl'; path.write_text(text)
    with pytest.raises(ValueError):
        run_eval.load_rows(path)


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / 'dev.jsonl'
    row = {'id': 'x', 'expect': 'allow', 'situation': 'hello'}
    path.write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(ValueError, match='duplicate'):
        run_eval.load_rows(path)


def test_invalid_confidence_cannot_enter_calibration_metric():
    with pytest.raises(ValueError):
        run_eval.ece([(float('nan'), True)])


def test_network_error_does_not_count_as_correct_benign_gate():
    module = load_file('eval/conscience/eval_tev1_judge.py')
    row = module.Trial(case_id='offline', category='benign', order_idx=0,
                       expected={axis: [] for axis in module.AXES}, should_flag=False, error='offline')
    result = module.summarize('offline', [row], 2500)
    assert result['n_valid'] == 0
    assert result['gate_accuracy'] == 0
    assert result['false_block_rate'] is None


def test_mem0_default_isolation_real_ids_order_and_close(monkeypatch):
    module = load_file('eval/memory/aiko_mem0_adapter.py')
    from cognition.memory import memorize
    monkeypatch.delenv('SQLITE_MEMORY_PATH', raising=False)
    monkeypatch.delenv('AIKO_BENCH_DRY_RUN', raising=False)
    seen = []
    class Backend:
        def __init__(self, **kwargs):
            self.path = kwargs['db_path']; seen.append(kwargs)
            Path(self.path).touch()
            self._conn = SimpleNamespace(close=lambda: seen.append('conn closed'))
            self._client = SimpleNamespace(close=lambda: seen.append('http closed'))
        def add(self, messages, user_id, display_name):
            assert display_name == user_id
            seen.append(('add', user_id)); return ['real-id']
        def search(self, query, **kwargs):
            return [{'id': 'first', 'memory': 'newer', '_recall_score': .1},
                    {'id': 'second', 'memory': 'older', '_recall_score': .9}]
        def delete_all(self, uid):
            seen.append(('delete', uid))
    monkeypatch.setattr(memorize, 'MemoryBackend', Backend)
    async def scenario():
        client = module.AikoMemClient()
        path = Path(client.db_path)
        async with client:
            assert (await client.add([{'role':'user', 'content':'hello'}], 'alice'))['results'][0]['id'] == 'real-id'
            assert [row['id'] for row in await client.search('query', 'alice')] == ['first', 'second']
            assert await client.search('query', 'alice', top_k=0) == []
            await client.delete_user('alice')
        assert not path.exists()
        await client.close()
        with pytest.raises(RuntimeError):
            await client.search('query', 'alice')
    asyncio.run(scenario())
    assert seen.count('conn closed') == seen.count('http closed') == 1


def test_mem0_refuses_existing_database_before_backend_construction(tmp_path, monkeypatch):
    module = load_file('eval/memory/aiko_mem0_adapter.py')
    path = tmp_path / 'existing.db'; path.write_text('preserve')
    monkeypatch.setenv('SQLITE_MEMORY_PATH', str(path))
    monkeypatch.delenv('AIKO_BENCH_DRY_RUN', raising=False)
    with pytest.raises(FileExistsError):
        module.AikoMemClient()
    assert path.read_text() == 'preserve'


def test_cancelled_benchmark_call_does_not_close_database_under_worker(monkeypatch):
    module = load_file('eval/memory/aiko_mem0_adapter.py')
    monkeypatch.setenv('AIKO_BENCH_DRY_RUN', '1')
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    def search(*args, **kwargs):
        started.set()
        assert release.wait(3)
        assert not closed.is_set()
        return []
    async def scenario():
        client = module.AikoMemClient()
        client._dry = False
        client._backend = SimpleNamespace(search=search, _conn=SimpleNamespace(close=closed.set),
                                           _client=SimpleNamespace(close=lambda: None))
        pending = asyncio.create_task(client.search('hello', 'alice'))
        assert await asyncio.to_thread(started.wait, 2)
        pending.cancel()
        closing = asyncio.create_task(client.close())
        await asyncio.sleep(.02)
        assert not closed.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await closing
        assert closed.is_set()
    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_system_logs_require_operator_privileges(monkeypatch):
    from fastapi.testclient import TestClient
    from interface.webui import auth
    from interface.webui.studio.log.backend import api
    from system import userspace
    async def session(_):
        return {'user_id': 'regular-user'}
    monkeypatch.setattr(auth, 'require_session', session)
    monkeypatch.setattr(userspace, 'can_approve', lambda uid: False)
    monkeypatch.setattr(api, 'read_records', lambda: pytest.fail('logs must not be read'))
    with TestClient(api.app) as client:
        assert client.get('/api/entries').status_code == 403
        assert client.get('/api/options').status_code == 403


def test_recall_eval_uses_unique_databases_and_closes_after_failure(monkeypatch):
    module = load_file('eval/memory/eval_memory_recall.py')
    from cognition.memory import memorize
    paths, closed = [], []
    class Backend:
        def __init__(self, db_path, **kwargs):
            paths.append(Path(db_path))
            Path(db_path).touch()
            self._conn = SimpleNamespace(close=lambda: closed.append('db'))
            self._client = SimpleNamespace(close=lambda: closed.append('http'))
    monkeypatch.setattr(memorize, 'MemoryBackend', Backend)
    def fail(*_):
        raise RuntimeError('embedding unavailable')
    monkeypatch.setattr(module, '_seed_store', fail)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            module.run_eval()
    assert paths[0] != paths[1]
    assert all(not path.exists() for path in paths)
    assert closed == ['db', 'http', 'db', 'http']


def test_extraction_eval_cleans_database_after_failure(monkeypatch):
    module = load_file('eval/memory/eval_memory_extraction.py')
    seen = []
    def create(path):
        Path(path).touch(); seen.append(Path(path))
        return SimpleNamespace(_conn=SimpleNamespace(close=lambda: seen.append('db')),
                               _client=SimpleNamespace(close=lambda: seen.append('http')))
    monkeypatch.setattr(module, '_load_real_backend', create)
    def fail(*_):
        raise RuntimeError('model unavailable')
    monkeypatch.setattr(module, '_run_cases', fail)
    with pytest.raises(RuntimeError):
        module.run_eval()
    assert not seen[0].exists() and seen[1:] == ['db', 'http']


def test_splitter_does_not_overwrite_frozen_test_or_dev_on_refusal(tmp_path, monkeypatch):
    module = load_file('conscience-lab/eval/split_eval.py')
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps({'id': str(i), 'checkpoint': 'input', 'refusal_category': None}) + '\n' for i in range(10)))
    monkeypatch.setattr(module, 'SRC', source)
    locked, dev = tmp_path / 'laya_eval_locked_test.jsonl', tmp_path / 'laya_eval_dev.jsonl'
    locked.write_text('frozen'); dev.write_text('preserve')
    with pytest.raises(ValueError, match='locked split'):
        module.main(write=True)
    assert locked.read_text() == 'frozen' and dev.read_text() == 'preserve'


def test_unknown_classifier_label_is_not_a_valid_refusal():
    classifier = adapter.LayaZeroShotAdapter([{'name': 'weapons', 'definition': 'case'}])
    backend = SimpleNamespace(decide=lambda *args: {'refusal': {'choice': 'typo', 'confidence': .9}})
    with pytest.raises(ValueError, match='unknown category'):
        classifier.classify(backend, 'hello', random.Random(0))


def test_server_error_is_not_retried_as_option_limit():
    import urllib.error
    calls = []
    def fail(*args):
        calls.append(True)
        raise urllib.error.HTTPError('http://fake', 500, 'offline', {}, None)
    classifier = adapter.LayaZeroShotAdapter([{'name': 'weapons', 'definition': 'case'}])
    with pytest.raises(urllib.error.HTTPError):
        classifier.classify(SimpleNamespace(decide=fail), 'hello', random.Random(0))
    assert len(calls) == 1
