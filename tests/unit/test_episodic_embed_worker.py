"""Regression tests for BUG-4 (P1): EMC embed-worker lifecycle.

- _enqueue_embed's check-then-act worker (re)start must be guarded so two
  concurrent enqueues can't spawn two threads on one queue.
- The embed queue is bounded: a stalled embed path drops the oldest item
  (best-effort semantics) instead of growing memory without bound or
  stalling the conversation thread.
"""
from __future__ import annotations

import queue
import threading

from cognition.memory import episode as em


def _store(tmp_path):
    return em.EpisodicStore(
        db_path=str(tmp_path / "emc.db"), user_id="u1", embedder=object()
    )


def _draining_loop(self):
    """Worker-loop stub: consume the queue without touching embedder/DB."""
    while True:
        item = self._embed_queue.get()
        try:
            if item is None:
                return
        finally:
            self._embed_queue.task_done()


class TestEmbedWorkerSpawnRace:
    def test_concurrent_enqueue_spawns_single_worker(self, tmp_path, monkeypatch):
        """Deterministic race test: the first spawner blocks inside
        Thread.__init__, holding the old check-then-act window open until a
        second spawner also passes the is_alive() check. Old code then
        creates 2 workers; the _embed_worker_lock serializes it to 1."""
        store = _store(tmp_path)
        assert store._embed_worker is None
        monkeypatch.setattr(em.EpisodicStore, "_embed_worker_loop", _draining_loop)

        created = []
        second_in_init = threading.Event()
        RealThread = threading.Thread

        class RacyThread(RealThread):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                if kw.get("name") == "emc-embed-worker":
                    created.append(self)
                    if len(created) == 1:
                        second_in_init.wait(timeout=5)
                    else:
                        second_in_init.set()

        monkeypatch.setattr(threading, "Thread", RacyThread)
        t1 = RealThread(target=store._enqueue_embed, args=(1, "trace-1"))
        t2 = RealThread(target=store._enqueue_embed, args=(2, "trace-2"))
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)
        assert len(created) == 1
        # Cleanup: stop the worker.
        store._embed_queue.put_nowait(None)
        store._embed_worker.join(timeout=5)
        assert not store._embed_worker.is_alive()


class TestEmbedQueueBound:
    def test_full_queue_drops_oldest_without_blocking(
        self, tmp_path, monkeypatch, caplog
    ):
        store = _store(tmp_path)
        # Small bound for the test; a live-looking dummy worker suppresses
        # the spawn path so the queue stays full.
        store._embed_queue = queue.Queue(maxsize=4)

        class _Alive:
            def is_alive(self):
                return True

        store._embed_worker = _Alive()
        for i in range(4):
            store._enqueue_embed(i, f"trace-{i}")
        assert store._embed_queue.qsize() == 4

        with caplog.at_level("WARNING", logger="cognition.memory.episode"):
            # Run in a thread: on the old unbounded-queue code this put()
            # would block forever — the test must fail, not hang.
            t = threading.Thread(
                target=store._enqueue_embed, args=(99, "trace-new"), daemon=True
            )
            t.start()
            t.join(timeout=10)
            assert not t.is_alive(), "enqueue blocked on a full queue"
        assert store._embed_queue.qsize() == 4
        assert any("embed queue full" in r.message for r in caplog.records)

        ids = []
        while True:
            try:
                item = store._embed_queue.get_nowait()
            except queue.Empty:
                break
            ids.append(item[0])
            store._embed_queue.task_done()
        assert 0 not in ids  # oldest dropped
        assert 99 in ids  # newest kept

    def test_queue_bound_is_sane(self):
        # A few flush batches (EMC_FLUSH_BATCH=32); not absurd in either direction.
        assert 16 <= em.EMC_EMBED_QUEUE_MAXSIZE <= 4096
