"""Storage stays intact and keyword recall survives embedding outages."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import requests
import sqlite_vec

from cognition.memory import vecstore
from cognition.knowledge import schema as ks, search as search_module, lifecycle


@pytest.fixture(autouse=True)
def explicit_embedding_mode(monkeypatch):
    monkeypatch.setattr(vecstore, '_flyal_mode', lambda: 'off')


def embedder(path=None, **kwargs):
    client = vecstore.HarrierEmbedder(base_url='http://mock', model='test', dims=2,
                                     cache_path=str(path) if path else None, **kwargs)
    client._session.post = Mock(side_effect=lambda *a, **k: NS(
        raise_for_status=lambda: None,
        json=lambda: [{'embedding': [3., 4.]} for _ in k['json']['content']]))
    return client


def test_embedding_batches_survive_restart_and_arrays_are_private(tmp_path):
    path = tmp_path / 'cache.jsonl'
    first = embedder(path)
    result = first.embed_batch(['one', 'two'])
    result[:] = 0
    np.testing.assert_allclose(first.embed_batch(['one', 'two']), [[.6, .8], [.6, .8]])
    second = embedder(path)
    result = second.embed_batch(['one', 'two'])
    result[:] = 0
    np.testing.assert_allclose(second.embed_batch(['one', 'two']), [[.6, .8], [.6, .8]])
    second._session.post.assert_not_called()


def test_model_changes_do_not_reuse_same_dimension_cache(tmp_path):
    path = tmp_path / 'cache.jsonl'
    first = embedder(path)
    first.embed_batch(['one'])
    first.model = 'changed'
    first.embed_batch(['one'])
    assert first._session.post.call_count == 2
    second = embedder(path)
    second.model = 'third'
    second.embed_batch(['one'])
    second._session.post.assert_called_once()


@pytest.mark.parametrize('payload', [[], [{'embedding': []}], [{'embedding': [0, 0]}],
    [{'embedding': [1, 2, 3]}], [{'embedding': [float('nan'), 1]}]])
def test_invalid_responses_never_enter_cache(payload):
    client = embedder()
    client._session.post.return_value = NS(raise_for_status=lambda: None, json=lambda: payload)
    client._session.post.side_effect = None
    with pytest.raises(ValueError):
        client.embed_batch(['one'])
    assert not client._cache and not client._disk_cache


def test_context_400_retries_do_not_open_backend_circuit():
    client = embedder()
    valid = client._session.post.side_effect
    def respond(*args, **kwargs):
        if len(kwargs['json']['content'][0]) > 300:
            response = requests.Response()
            response.status_code = 400
            raise requests.HTTPError('input too long', response=response)
        return valid(*args, **kwargs)
    client._session.post.side_effect = respond
    np.testing.assert_allclose(client.embed_query('x' * 2000), [.6, .8])
    assert client._session.post.call_count >= 4
    assert client._cb_open_until == 0


def test_embedding_iterator_does_not_materialize_all_documents():
    client = embedder(batch_size=2)
    consumed = []
    def texts():
        for i in range(10000):
            consumed.append(i)
            yield str(i)
    output = client.embed(texts())
    np.testing.assert_allclose(next(output), [.6, .8])
    assert consumed == [0, 1]
    output.close()
    assert client.embed_batch([]).shape == (0, 2)
    assert client.embed_queries([]).shape == (0, 2)


def test_cache_bounds_count_vectors_not_batches(tmp_path):
    client = embedder(tmp_path / 'cache.jsonl')
    client._CACHE_MAX = client._DISK_CACHE_MAX = 3
    for prefix in ('a', 'b', 'c'):
        client.embed_batch([prefix + '1', prefix + '2'])
    assert sum(len(value[1]) for value in client._cache.values()) <= 3
    assert sum(len(value) for value in client._disk_cache.values()) <= 3


def test_concurrent_compaction_preserves_other_instances_appends(tmp_path):
    path = tmp_path / 'cache.jsonl'
    clients = [embedder(path), embedder(path)]
    for client in clients:
        client._DISK_CACHE_FILE_MAX_BYTES = 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i: clients[i].embed_batch([str(i)]), range(2)))
    loaded = embedder(path)
    loaded.embed_batch(['0'])
    loaded.embed_batch(['1'])
    loaded._session.post.assert_not_called()


def test_disk_load_discards_legacy_invalid_and_excess_entries(tmp_path, monkeypatch):
    path = tmp_path / 'cache.jsonl'
    client = embedder()
    entries = [{'texts': ['legacy'], 'vector': [1, 0]}]
    for i in range(6):
        key = client._embedding_key([str(i)])
        entries.append(client._disk_entry(key, np.array([[1., 0.]])))
    entries.append(client._disk_entry(client._embedding_key(['bad']), np.array([[float('nan'), 0.]])))
    path.write_text('\n'.join(json.dumps(entry) for entry in entries))
    monkeypatch.setattr(vecstore.HarrierEmbedder, '_DISK_CACHE_MAX', 2)
    loaded = embedder(path)
    assert [key[-1] for key in loaded._disk_cache] == [('4',), ('5',)]


class FailingMigrationConnection(sqlite3.Connection):
    fail_rebuild = False
    def execute(self, sql, *args, **kwargs):
        if self.fail_rebuild and sql.startswith('CREATE VIRTUAL TABLE vectors'):
            raise sqlite3.OperationalError('simulated rebuild failure')
        return super().execute(sql, *args, **kwargs)


@pytest.mark.parametrize('fail', [False, True])
def test_metric_migration_is_atomic_and_preserves_vector_pairs(tmp_path, fail):
    conn = sqlite3.connect(tmp_path / 'migration.db', factory=FailingMigrationConnection)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute('CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[2])')
    rows = [('b', sqlite_vec.serialize_float32([0., 1.])), ('a', sqlite_vec.serialize_float32([1., 0.]))]
    conn.executemany('INSERT INTO vectors(id,embedding) VALUES(?,?)', rows)
    conn.commit()
    original = conn.execute("SELECT sql FROM sqlite_master WHERE name='vectors'").fetchone()[0]
    conn.fail_rebuild = fail
    try:
        if fail:
            with pytest.raises(sqlite3.OperationalError, match='simulated'):
                vecstore.ensure_vec0_cosine_metric(conn)
            assert conn.execute("SELECT sql FROM sqlite_master WHERE name='vectors'").fetchone()[0] == original
        else:
            assert vecstore.ensure_vec0_cosine_metric(conn) == ['vectors']
            assert vecstore.ensure_vec0_cosine_metric(conn) == []
        assert sorted(conn.execute('SELECT id,embedding FROM vectors').fetchall()) == sorted(rows)
    finally:
        conn.close()


@pytest.fixture
def knowledge_db(tmp_path, monkeypatch):
    path = tmp_path / 'knowledge.db'
    def connect(uid=None):
        return vecstore.initialize_store_db(path, ks._DDL, user_id=uid, vector=True)
    monkeypatch.setattr(search_module, 'connect', connect)
    monkeypatch.setattr(ks, 'connect', connect)
    conn = connect('owner')
    for uid in ('owner', 'other'):
        conn.execute('INSERT INTO learned_docs VALUES(?,?,?,?,?,?)', (uid, uid, uid, '', 'test', '2026-01-01'))
        for i in range(2):
            conn.execute('INSERT INTO learned_chunks(id,doc_id,user_id,chunk_index,text,created_at) VALUES(?,?,?,?,?,?)',
                         (f'{uid}{i}', uid, uid, i, 'Jetson autumn memory', '2026-01-01'))
    conn.commit()
    conn.close()
    search_module.maybe_clear_knowledge_cache()
    yield connect
    search_module.flush_access_counts('owner')
    search_module.maybe_clear_knowledge_cache()


def test_knowledge_embedding_outage_retains_scoped_lexical_results(knowledge_db):
    provider = NS(embed_query=Mock(side_effect=RuntimeError('offline')))
    results = search_module.search_knowledge('Jetson autumn', embedder=provider, user_id='owner')
    assert {row['id'] for row in results} == {'owner0', 'owner1'}
    assert search_module.search_knowledge('Jetson', limit=0, user_id='owner') == []


def test_access_updates_cannot_touch_another_users_rows(knowledge_db):
    search_module._write_access_counts('owner', ['owner0', 'other0'])
    conn = knowledge_db()
    try:
        assert conn.execute("SELECT access_count FROM learned_chunks WHERE id='owner0'").fetchone()[0] == 1
        assert conn.execute("SELECT access_count FROM learned_chunks WHERE id='other0'").fetchone()[0] == 0
    finally:
        conn.close()


def test_invalid_dedupe_vectors_do_not_archive_unrelated_chunks(knowledge_db):
    conn = knowledge_db()
    try:
        provider = NS(embed_queries=lambda texts: np.full((len(texts), 2), np.nan))
        assert lifecycle._deduplicate_chunks(conn, 'owner', provider, .95) == 0
        assert conn.execute('SELECT count(*) FROM learned_chunks').fetchone()[0] == 4
        assert conn.execute('SELECT count(*) FROM learned_chunks_archive').fetchone()[0] == 0
    finally:
        conn.close()


def test_cache_invalidation_cannot_be_undone_by_an_inflight_search():
    generation = search_module._CACHE_GENERATION
    search_module.maybe_clear_knowledge_cache()
    search_module._search_cache_set('query', 'owner', 1, 'provider', [{'id': 'deleted'}], generation=generation)
    assert search_module._search_cache_get('query', 'owner', 1, 'provider') is None


def test_codebase_ingestion_and_recall_work_without_embedding_service(tmp_path, monkeypatch):
    from cognition.knowledge import codebase
    path = tmp_path / 'codebase.db'
    def connect(uid=None):
        return vecstore.initialize_store_db(path, codebase._DDL, user_id=uid, vector=True)
    monkeypatch.setattr(codebase, 'connect_codebase', connect)
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'main.py').write_text('def autumn_memory():\n    return "Jetson autumn leaves"\n' * 3)
    provider = NS(embed=Mock(side_effect=RuntimeError('offline')),
                  embed_query=Mock(side_effect=RuntimeError('offline')))
    result = codebase.ingest_codebase('owner', repo_root=repo, embedder=provider)
    assert result['ok'], result
    assert result['chunks_added'] > 0
    conn = connect()
    try:
        assert conn.execute('SELECT count(*) FROM codebase_chunks_vec').fetchone()[0] == 0
    finally:
        conn.close()
    hits = codebase.search_codebase('autumn leaves', user_id='owner', embedder=provider)
    assert hits and hits[0]['path'] == 'main.py'
    assert codebase.search_codebase('autumn leaves', user_id='other', embedder=provider) == []
    recovered = NS(embed=lambda texts: [np.ones(ks.EMBED_DIMS, dtype=np.float32) for _ in texts])
    second = codebase.ingest_codebase('owner', repo_root=repo, embedder=recovered)
    assert second['ok'] and second['docs_added'] == 0
    conn = connect()
    try:
        assert conn.execute('SELECT count(*) FROM codebase_chunks_vec').fetchone()[0] == result['chunks_added']
    finally:
        conn.close()


def test_foreign_embedder_cache_keys_keep_the_provider_alive():
    import gc
    import weakref
    from cognition import reason
    reason._embed_query_cache.clear()
    class Provider:
        def embed_query(self, text, instruct=''):
            return np.array([1., 0.])
    provider = Provider()
    ref = weakref.ref(provider)
    reason.cached_embed_query(provider, 'query')
    del provider
    gc.collect()
    assert ref() is not None  # its id cannot be recycled under a live cache key
    reason._embed_query_cache.clear()
    gc.collect()
    assert ref() is None


def test_precomputed_knowledge_query_vector_needs_no_embedder(knowledge_db, monkeypatch):
    knn = Mock(return_value=[])
    monkeypatch.setattr(search_module, 'user_scoped_vec_knn', knn)
    conn = knowledge_db()
    try:
        search_module._knn(conn, 'autumn', None, 'owner', 2, query_vector=[1, 0])
    finally:
        conn.close()
    assert knn.call_args.kwargs['vector'] == [1, 0]


def test_knowledge_embedding_does_not_hold_the_database_writer_lock(knowledge_db, monkeypatch):
    from cognition.knowledge import ingest
    touched = []
    class Provider:
        def embed_queries(self, texts):
            conn = knowledge_db('other')
            try:
                conn.execute("UPDATE learned_docs SET title='changed' WHERE id='other'")
                conn.commit()
                touched.append(True)
            finally:
                conn.close()
            return None
        def embed_query(self, text, instruct=''):
            raise RuntimeError('offline')
    result = ingest.ingest_text('new document', 'A new autumn memory on Jetson.',
                                user_id='owner', embedder=Provider())
    assert result and touched == [True]


def test_invalid_chunk_embeddings_preserve_durable_knowledge_text(knowledge_db):
    from cognition.knowledge import ingest
    provider = NS(embed_queries=lambda texts: np.full((len(texts), ks.EMBED_DIMS), np.nan))
    result = ingest.ingest_text('new document', 'A newly learned autumn memory on Jetson.',
                                user_id='owner', embedder=provider)
    assert result
    conn = knowledge_db()
    try:
        assert conn.execute('SELECT count(*) FROM learned_chunks WHERE doc_id=?', (result,)).fetchone()[0] > 0
        assert conn.execute('SELECT count(*) FROM learned_chunks_vec').fetchone()[0] == 0
    finally:
        conn.close()


def test_guest_shutdown_sentinel_does_not_open_a_database(monkeypatch):
    import queue
    from cognition.memory.memorize import AikoMemorize
    memo = AikoMemorize.__new__(AikoMemorize)
    memo._mem_backend = None
    memo._write_queue = queue.Queue()
    memo._write_queue.put(None)
    open_db = Mock(side_effect=AssertionError('guest store must stay lazy'))
    monkeypatch.setattr(memo, '_open', open_db)
    memo._write_loop()
    open_db.assert_not_called()
    assert memo._write_queue.unfinished_tasks == 0


def test_episodic_shutdown_preserves_inflight_embedding_and_closes_later(tmp_path, monkeypatch):
    import threading
    from cognition.memory import episode
    started, release = threading.Event(), threading.Event()
    class Provider:
        def embed(self, texts):
            started.set()
            assert release.wait(5)
            return [np.ones(ks.EMBED_DIMS, dtype=np.float32)]
    store = episode.EpisodicStore(str(tmp_path / 'episode.db'), user_id='owner', embedder=Provider())
    store._conn.execute('INSERT INTO emc_storage(id,user_id,timestamp,date,trace) VALUES(1,?,?,?,?)',
                        ('owner', '2026-01-01T00:00:00+00:00', '2026-01-01', 'autumn leaves'))
    store._conn.commit()
    store._enqueue_embed(1, 'autumn leaves')
    assert started.wait(5)
    worker = store._embed_worker
    real_join = worker.join
    monkeypatch.setattr(worker, 'join', lambda timeout=None: None)
    try:
        store.close()
        store._conn.execute('SELECT 1')  # still owned by the active worker
        store._enqueue_embed(2, 'late input')
        assert store._embed_queue.qsize() == 1  # only the shutdown sentinel
    finally:
        release.set()
        real_join(timeout=5)
    assert not worker.is_alive()
    with pytest.raises(sqlite3.ProgrammingError):
        store._conn.execute('SELECT 1')
    with sqlite3.connect(tmp_path / 'episode.db') as conn:
        assert conn.execute('SELECT encoding IS NOT NULL FROM emc_storage WHERE id=1').fetchone()[0] == 1


def test_plain_document_extraction_bounds_reads_and_normalizes_across_chunks(tmp_path, monkeypatch):
    import io
    from pathlib import Path
    from cognition.knowledge import ingest
    text = '  ' + 'word\t\n' * 10000
    path = tmp_path / 'large.log'
    path.write_text(text)
    reads = []
    class Bounded(io.StringIO):
        def read(self, size=-1):
            assert 0 < size <= 8192
            reads.append(size)
            return super().read(size)
    monkeypatch.setattr(Path, 'open', lambda *a, **k: Bounded(text))
    assert ingest._read_plain_text(path, 250) == ingest._sanitize_text(text, 250).rstrip()
    assert len(reads) == 1


def test_knowledge_watcher_retains_owner_context_and_restarts(tmp_path, monkeypatch):
    from cognition.knowledge import watcher as module
    from system import userspace
    seen = []
    monkeypatch.setattr(module, 'user_workspace_root', lambda uid: tmp_path / uid)
    monkeypatch.setattr(module, '_inotify_init', lambda: 123)
    monkeypatch.setattr(module, '_inotify_add_watch', lambda *a: 1)
    monkeypatch.setattr(module, '_inotify_close', lambda *a: None)
    watcher = module.KnowledgeFolderWatcher(user_id='owner')
    def run():
        seen.append((userspace.current_user_id(), watcher.folder, watcher._stop.is_set()))
    monkeypatch.setattr(watcher, '_run', run)
    for _ in range(2):
        assert watcher.start()
        watcher._thread.join(5)
        watcher.stop()
    assert seen == [('owner', tmp_path / 'owner/library', False)] * 2


def test_knowledge_watcher_ignores_truncated_kernel_events():
    from pathlib import Path
    from cognition.knowledge import watcher
    assert watcher._parse_events(b'\x01', Path('/tmp')) == []
    partial = watcher._INOTIFY_EVENT.pack(1, watcher._IN_CREATE, 0, 10) + b'abc'
    assert watcher._parse_events(partial, Path('/tmp')) == []


def test_failed_vector_extension_load_disables_loading_and_closes_connection(monkeypatch):
    conn = Mock()
    monkeypatch.setattr(vecstore, 'connect_sqlite_db', lambda *a, **k: conn)
    monkeypatch.setattr(sqlite_vec, 'load', Mock(side_effect=sqlite3.OperationalError('load failed')))
    with pytest.raises(sqlite3.OperationalError, match='load failed'):
        vecstore.connect_sqlite_vec(':memory:')
    assert [call.args for call in conn.enable_load_extension.call_args_list] == [(True,), (False,)]
    conn.close.assert_called_once()


def test_knowledge_maintenance_embeds_without_holding_writer_lock(knowledge_db, monkeypatch):
    main = knowledge_db()
    monkeypatch.setattr(lifecycle, 'connect', lambda uid=None: main)
    def embed(texts):
        assert not main.in_transaction
        return np.ones((len(texts), 2), dtype=np.float32)
    result = lifecycle.prune_knowledge(user_id='owner', embedder=NS(embed_queries=embed),
                                      archive_days=10000, delete_days=10000)
    assert result['errors'] == 0
    assert result['deduped'] == 1


def test_knowledge_dedupe_rejects_snapshot_changed_during_embedding(knowledge_db):
    main = knowledge_db()
    def embed(texts):
        assert not main.in_transaction
        other = knowledge_db()
        try:
            other.execute("UPDATE learned_chunks SET access_count=8 WHERE id='owner1'")
            other.commit()
        finally:
            other.close()
        return np.ones((len(texts), 2), dtype=np.float32)
    try:
        assert lifecycle._deduplicate_chunks(main, 'owner', NS(embed_queries=embed), .95) == 0
        assert main.execute('SELECT count(*) FROM learned_chunks_archive').fetchone()[0] == 0
        assert main.execute("SELECT access_count FROM learned_chunks WHERE id='owner1'").fetchone()[0] == 8
    finally:
        main.close()
