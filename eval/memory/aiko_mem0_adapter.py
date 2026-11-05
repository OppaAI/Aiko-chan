"""Mem0 benchmark facade over Aiko's real semantic MemoryBackend.

Set AIKO_ROOT to this checkout when copying this adapter to another repository.
By default each client owns a private temporary database, closed and removed by
close(). SQLITE_MEMORY_PATH may select a NEW scratch file; existing files are
refused so a benchmark cannot clear an existing store by mistake. Calls use
explicit benchmark user IDs and never switch Aiko's live user/profile state.

This measures semantic extraction and retrieval, not episodic consolidation or
persona behavior. AIKO_BENCH_DRY_RUN=1 is a protocol smoke test, not a score.
"""
from __future__ import annotations

import asyncio
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_AIKO_ROOT = os.environ.get('AIKO_ROOT', '').strip()
if _AIKO_ROOT:
    sys.path.insert(0, _AIKO_ROOT)


def _resolve_user_id(user_id: str) -> str:
    return os.getenv('AIKO_BENCH_USER', '').strip() or user_id or 'bench'


def _messages_to_text(messages):
    return '\n'.join(f"{m.get('role') or 'user'}: {m.get('content') or ''}" for m in messages if (m.get('content') or '').strip())


def _normalize_search_hits(raw, top_k):
    if top_k <= 0:
        return []
    hits = []
    for index, row in enumerate(raw or []):
        if not isinstance(row, dict):
            continue
        score = row.get('_recall_score', row.get('score', 0))
        try:
            score = float(score)
            if not math.isfinite(score):
                score = 0.0
        except (TypeError, ValueError):
            score = 0.0
        hit = {'memory': str(row.get('memory') or row.get('text') or ''),
               'score': score, 'id': str(row.get('id') or index)}
        for key in ('created_at', 'kind', 'pinned', 'valence_tag', 'valence_score'):
            if row.get(key) is not None:
                hit[key] = row[key]
        hits.append(hit)
    # The runtime's final order may include recency/lineage adjustments beyond
    # the raw score. Preserve it instead of silently reranking the benchmark.
    return hits[:top_k]


class AikoMemClient:
    def __init__(self, mode='oss', host=None, api_key=None, organization_id=None,
                 project_id=None, max_retries=3, retry_delay=1.0, rpm=120,
                 timeout=300.0, **kwargs):
        self.mode, self.host = 'aiko', host or 'aiko-local'
        self._dry = os.getenv('AIKO_BENCH_DRY_RUN', '').lower() in {'1', 'true', 'yes'}
        self._backend = None
        self._scratch = None
        self._closed = False
        self._lock = asyncio.Lock()
        if self._dry:
            return
        from system.config import load_config
        load_config()
        from cognition.memory.memorize import MemoryBackend
        explicit = os.getenv('SQLITE_MEMORY_PATH', '').strip()
        if explicit:
            path = Path(explicit).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents both accidental reuse and symlink use.
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        else:
            self._scratch = tempfile.TemporaryDirectory(prefix='aiko-mem0-')
            path = Path(self._scratch.name) / 'memory.db'
        self.db_path = str(path)
        try:
            self._backend = MemoryBackend(db_path=self.db_path, user_id='bench',
                llm_base_url=os.getenv('LLM_BASE_URL', 'http://localhost:8080/v1'),
                model=os.getenv('EXTRACT_MODEL') or os.getenv('LLM_MODEL', 'ministral'))
        except BaseException:
            if self._scratch is not None:
                self._scratch.cleanup()
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def _ready(self):
        if self._closed:
            raise RuntimeError('benchmark client is closed')

    async def _call(self, function, *args, **kwargs):
        # Cancelling an await cannot stop native SQLite work. Retain the lock
        # until that worker exits before allowing close/delete/another call.
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancelled:
            try:
                await asyncio.shield(task)
            finally:
                raise cancelled

    async def close(self):
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._backend is not None:
                await self._call(self._backend._conn.close)
                await self._call(self._backend._client.close)
            if self._scratch is not None:
                self._scratch.cleanup()

    async def add(self, messages, user_id, observation_date=None, timestamp=None,
                  custom_instructions=None, metadata=None):
        async with self._lock:
            self._ready()
            if not _messages_to_text(messages).strip():
                return {'results': []}
            if self._dry:
                return {'results': [{'memory': _messages_to_text(messages), 'event': 'ADD', 'id': 'dry'}]}
            uid = _resolve_user_id(user_id)
            ids = await self._call(self._backend.add, messages, user_id=uid, display_name=uid)
            return {'results': [{'id': str(mid), 'event': 'ADD'} for mid in ids]}

    async def search(self, query, user_id, top_k=200, rerank=False, score_debug=False):
        async with self._lock:
            self._ready()
            if top_k <= 0:
                return []
            if self._dry:
                return [{'memory': '[dry] ' + query, 'score': 1.0, 'id': 'dry'}]
            rows = await self._call(self._backend.search, query,
                user_id=_resolve_user_id(user_id), limit=top_k, include_history=False)
            return _normalize_search_hits(rows, top_k)

    async def delete_user(self, user_id):
        async with self._lock:
            self._ready()
            if not self._dry:
                await self._call(self._backend.delete_all, _resolve_user_id(user_id))
            return True


if __name__ == '__main__':
    async def smoke():
        async with AikoMemClient() as client:
            print('Aiko benchmark client ready:', getattr(client, 'db_path', 'dry-run'))
    asyncio.run(smoke())
