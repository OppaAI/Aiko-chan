"""Changed library files refresh atomically, inspired by Aidon's KB manifest."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from cognition.knowledge import ingest, schema, search
from cognition.memory.vecstore import initialize_store_db


@pytest.fixture
def library(tmp_path, monkeypatch):
    roots = {uid: tmp_path / uid for uid in ('owner', 'other')}
    for root in roots.values():
        (root / 'library').mkdir(parents=True)
    monkeypatch.setattr(ingest, 'user_workspace_root', lambda uid=None: roots[uid or 'owner'])
    monkeypatch.setattr(ingest, 'KNOWLEDGE_WORKSPACE_DIR', 'library')
    path = tmp_path / 'knowledge.db'  # shared to exercise explicit user scoping
    def connect(uid=None):
        return initialize_store_db(path, schema._DDL, user_id=uid, vector=True)
    monkeypatch.setattr(ingest, 'connect', connect)
    monkeypatch.setattr(schema, 'connect', connect)
    monkeypatch.setattr(search, 'connect', connect)
    search.maybe_clear_knowledge_cache()
    yield NS(roots=roots, path=path, connect=connect, file=roots['owner'] / 'library/guide.txt')
    search.maybe_clear_knowledge_cache()


def rows(library, sql, args=()):
    conn = library.connect()
    try:
        return [dict(row) for row in conn.execute(sql, args)]
    finally:
        conn.close()


def scan(uid='owner', embedder=None):
    return ingest.ingest_workspace_knowledge_folder(user_id=uid, embedder=embedder)


def test_hash_skips_unchanged_file_before_extraction_or_embedding(library, monkeypatch):
    library.file.write_text('Jetson startup instructions before the autumn update.')
    assert len(scan()) == 1
    extract = Mock(side_effect=AssertionError('unchanged source was reparsed'))
    monkeypatch.setattr(ingest, 'extract_text_from_file', extract)
    assert scan(embedder=Mock()) == []
    extract.assert_not_called()
    assert len(rows(library, 'SELECT id FROM learned_docs')) == 1


def test_scan_reuses_hash_but_detects_same_size_atomic_replacement(library, monkeypatch):
    library.file.write_text('First maple instructions.')
    assert scan()
    old_stamp = library.file.stat()
    real_hash = ingest._file_fingerprint
    hashing = Mock(wraps=real_hash)
    monkeypatch.setattr(ingest, '_file_fingerprint', hashing)
    assert scan() == []
    hashing.assert_not_called()
    replacement = library.file.with_name('replacement.tmp')
    replacement.write_text('Other maple instructions.')
    os.utime(replacement, ns=(old_stamp.st_atime_ns, old_stamp.st_mtime_ns))
    replacement.replace(library.file)
    assert scan()
    assert hashing.call_count > 0
    assert search.search_knowledge('Other maple instructions', user_id='owner')


def test_library_scan_excludes_hidden_directories_and_directory_links(library):
    folder = library.roots['owner'] / 'library'
    hidden = folder / '.private'
    hidden.mkdir()
    (hidden / 'credentials.txt').write_text('Private directory must not be indexed.')
    outside_library = library.roots['owner'] / 'unshared'
    outside_library.mkdir()
    (outside_library / 'notes.txt').write_text('Unshared directory must not be indexed.')
    (folder / 'alias').symlink_to(outside_library, target_is_directory=True)
    library.file.write_text('Public maple instructions.')
    assert len(scan()) == 1
    assert [row['source'] for row in rows(library, 'SELECT source FROM learned_docs')] == ['library/guide.txt']


def test_scan_hash_cache_is_bounded_and_recovers_after_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, '_SCAN_HASH_LIMIT', 2)
    monkeypatch.setattr(ingest, '_SCAN_HASH_CACHE', ingest.OrderedDict())
    paths = [tmp_path / str(i) for i in range(3)]
    for path in paths:
        path.write_text('Maple')
        ingest._scan_fingerprint(path)
    assert len(ingest._SCAN_HASH_CACHE) == 2
    assert paths[0] not in ingest._SCAN_HASH_CACHE
    assert ingest._scan_fingerprint(paths[0])[0] == hashlib.sha256(b'Maple').hexdigest()


def test_import_status_survives_reopen_and_failed_refresh_keeps_document(library, monkeypatch):
    from cognition.knowledge.status import ingestion_status
    library.file.write_text('Stable library knowledge about maple trees.')
    first = scan()[0]
    assert ingestion_status(user_id='owner')[0]['state'] == 'ready'
    library.file.write_text('Updated maple tree knowledge.')
    monkeypatch.setattr(ingest, 'extract_text_from_file', Mock(side_effect=ValueError('parse failure')))
    assert scan() == []
    status = ingestion_status(user_id='owner')[0]
    assert status['state'] == 'failed'
    assert status['doc_id'] == first
    assert not ingestion_status(user_id='other')
    assert search.search_knowledge('Stable maple trees', user_id='owner')


def test_image_caption_is_searchable_and_linked_to_workspace_asset(library):
    image_path = library.roots['owner'] / 'autumn.png'
    image_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 24)
    doc_id = ingest.ingest_image('autumn.png', 'A red maple beside a quiet garden path', user_id='owner')
    assert doc_id
    hit = search.search_knowledge('maple garden path', user_id='owner')[0]
    assert hit['doc_id'] == doc_id
    assert hit['asset_path'] == 'autumn.png'
    assert hit['asset_media_type'] == 'image/png'
    context = search.knowledge_context_for('maple garden path', user_id='owner')
    assert 'asset_path="autumn.png"' in context
    image_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x01' * 24)
    updated = ingest.ingest_image('autumn.png', 'A red maple with golden leaves', user_id='owner')
    assert updated and updated != doc_id
    assert not search.search_knowledge('quiet garden path', user_id='owner')
    assert search.search_knowledge('golden leaves maple', user_id='owner')[0]['doc_id'] == updated


def test_changed_source_refreshes_recall_and_retains_history(library):
    library.file.write_text('Jetson winter cedar instructions are obsolete.')
    first = scan()[0]
    assert search.search_knowledge('winter cedar', user_id='owner')
    # The cached context must be invalidated after replacement too.
    search.knowledge_context_for('winter cedar', user_id='owner')
    library.file.write_text('Jetson autumn maple instructions are current.')
    second = scan()[0]
    assert first != second
    assert not search.search_knowledge('winter cedar', user_id='owner')
    assert search.search_knowledge('autumn maple', user_id='owner')[0]['doc_id'] == second
    assert all(row['status'] == 'superseded' for row in rows(
        library, 'SELECT status FROM learned_chunks WHERE doc_id=?', (first,)))
    state = rows(library, 'SELECT * FROM knowledge_file_state')[0]
    assert state['doc_id'] == second
    assert state['source_hash'] == hashlib.sha256(library.file.read_bytes()).hexdigest()
    assert scan() == []
    search.flush_access_counts('owner')


def test_failed_update_keeps_prior_chunks_and_fingerprint(library, monkeypatch):
    library.file.write_text('Reliable Jetson winter instructions.')
    first = scan()[0]
    original = rows(library, 'SELECT * FROM knowledge_file_state')
    library.file.write_text('Updated Jetson autumn instructions.')
    monkeypatch.setattr(ingest, 'extract_entities', Mock(side_effect=ValueError('simulated insertion failure')))
    assert scan() == []
    assert rows(library, 'SELECT * FROM knowledge_file_state') == original
    assert rows(library, 'SELECT DISTINCT doc_id,status FROM learned_chunks') == [
        {'doc_id': first, 'status': 'active'}]
    assert len(rows(library, 'SELECT id FROM learned_docs')) == 1


def test_modification_during_extraction_keeps_prior_version(library, monkeypatch):
    library.file.write_text('Reliable Jetson winter instructions.')
    scan()
    original = rows(library, 'SELECT * FROM knowledge_file_state')
    library.file.write_text('Intermediate instructions.')
    real = ingest.extract_text_from_file
    def extract(*args, **kwargs):
        output = real(*args, **kwargs)
        library.file.write_text('Changed while parsing.')
        return output
    monkeypatch.setattr(ingest, 'extract_text_from_file', extract)
    assert scan() == []
    assert rows(library, 'SELECT * FROM knowledge_file_state') == original


def test_modification_during_embedding_does_not_publish_stale_text(library):
    library.file.write_text('Reliable Jetson winter instructions.')
    scan()
    original = rows(library, 'SELECT * FROM knowledge_file_state')
    library.file.write_text('Intermediate instructions.')
    class Provider:
        def embed_queries(self, texts):
            library.file.write_text('Changed while embedding.')
            return np.ones((len(texts), schema.EMBED_DIMS), dtype=np.float32)
    assert scan(embedder=Provider()) == []
    assert rows(library, 'SELECT * FROM knowledge_file_state') == original


def test_concurrent_imports_publish_only_one_version(library):
    library.file.write_text('Concurrent Jetson autumn instructions.')
    barrier = threading.Barrier(2)
    class Provider:
        def embed_queries(self, texts):
            barrier.wait(timeout=5)
            return np.ones((len(texts), schema.EMBED_DIMS), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ingest.ingest_file(
            'library/guide.txt', user_id='owner', embedder=Provider()), range(2)))
    assert sum(value is not None for value in results) == 1
    assert len(rows(library, 'SELECT id FROM learned_docs')) == 1
    assert len(rows(library, 'SELECT * FROM knowledge_file_state')) == 1


def test_same_relative_path_cannot_replace_another_users_document(library):
    library.file.write_text('Owner Jetson winter instructions.')
    (library.roots['other'] / 'library/guide.txt').write_text('Other user instructions.')
    first = scan()[0]
    other = scan('other')[0]
    library.file.write_text('Owner Jetson autumn update.')
    scan()
    assert rows(library, 'SELECT DISTINCT status FROM learned_chunks WHERE doc_id=?', (other,)) == [{'status': 'active'}]
    assert rows(library, 'SELECT DISTINCT status FROM learned_chunks WHERE doc_id=?', (first,)) == [{'status': 'superseded'}]


def test_legacy_source_gets_a_fingerprint_and_rollback_remains_safe(library):
    conn = library.connect()
    conn.execute('DROP TABLE knowledge_file_state')
    conn.commit()
    schema.ensure_knowledge_schema_migrated(conn, 'owner')
    conn.close()
    first = ingest.ingest_text('old import', 'Legacy winter instructions.',
        source='library/guide.txt', user_id='owner')
    library.file.write_text('Current autumn instructions.')
    assert len(scan()) == 1
    assert rows(library, 'SELECT DISTINCT status FROM learned_chunks WHERE doc_id=?', (first,)) == [{'status': 'superseded'}]
    assert len(rows(library, 'SELECT * FROM knowledge_file_state')) == 1


def test_retired_versions_do_not_crowd_out_current_vector_recall(library):
    class Provider:
        def embed_queries(self, texts):
            return np.ones((len(texts), schema.EMBED_DIMS), dtype=np.float32)
        def embed_query(self, text, instruct=''):
            return np.ones(schema.EMBED_DIMS, dtype=np.float32)
    provider = Provider()
    for version in range(22):
        library.file.write_text(f'Version {version}: Jetson autumn instructions.')
        latest = scan(embedder=provider)[0]
    matches = search.search_knowledge('unmatched lexical query', embedder=provider, user_id='owner')
    assert [row['doc_id'] for row in matches] == [latest]
    assert len(rows(library, 'SELECT id FROM learned_chunks_vec')) == 1
    assert len(rows(library, 'SELECT id FROM learned_docs')) == 22


def test_deleting_a_source_file_does_not_implicitly_forget_learned_history(library):
    library.file.write_text('Durable Jetson autumn instructions.')
    original = scan()[0]
    library.file.unlink()
    assert scan() == []
    assert search.search_knowledge('autumn instructions', user_id='owner')[0]['doc_id'] == original


def test_embedding_recovery_backfills_without_reimporting_unchanged_file(library):
    library.file.write_text('Maple leaves in the quiet garden.')
    first = scan()[0]
    assert not rows(library, 'SELECT id FROM learned_chunks_vec')
    class Provider:
        def embed_queries(self, texts):
            return np.ones((len(texts), schema.EMBED_DIMS), dtype=np.float32)
    assert scan(embedder=Provider()) == []
    assert len(rows(library, 'SELECT id FROM learned_docs')) == 1
    assert rows(library, 'SELECT doc_id FROM knowledge_file_state')[0]['doc_id'] == first
    assert rows(library, 'SELECT id FROM learned_chunks_vec')


def test_vector_repair_revalidates_chunk_after_remote_embedding(library):
    from cognition.knowledge.backfill import backfill_missing_vectors
    library.file.write_text('Maple leaves in the quiet garden.')
    scan()
    class Provider:
        def embed_queries(self, texts):
            conn = library.connect()
            try:
                conn.execute("UPDATE learned_chunks SET text='Changed while embedding' WHERE user_id='owner'")
                conn.commit()
            finally:
                conn.close()
            return np.ones((len(texts), schema.EMBED_DIMS), dtype=np.float32)
    assert backfill_missing_vectors(embedder=Provider(), user_id='owner') == 0
    assert not rows(library, 'SELECT id FROM learned_chunks_vec')


def test_vector_repair_is_bounded_and_user_scoped(library):
    from cognition.knowledge.backfill import backfill_missing_vectors
    for i in range(3):
        assert ingest.ingest_text(str(i), f'Maple garden instructions {i}', user_id='owner')
    assert ingest.ingest_text('Other', 'Other user private knowledge', user_id='other')
    class Provider:
        def embed_queries(self, texts):
            assert len(texts) == 2
            return np.ones((len(texts), schema.EMBED_DIMS), dtype=np.float32)
    assert backfill_missing_vectors(embedder=Provider(), user_id='owner', limit=2) == 2
    assert len(rows(library, 'SELECT id FROM learned_chunks_vec')) == 2
    assert not rows(library, "SELECT v.id FROM learned_chunks_vec v JOIN learned_chunks c ON c.id=v.id WHERE c.user_id='other'")
