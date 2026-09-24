"""Regression coverage for buffered audit, memory, and teaching writes."""
from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest


def test_ledger_retries_after_partial_insert_and_exposes_same_turn_rows(tmp_path):
    from cognition.conscience.ledger import ConscienceLedger
    from cognition.conscience.schema import LEDGER_DDL, Verdict

    class FailingConnection(sqlite3.Connection):
        fail_batch = False

        def executemany(self, sql, rows):
            if self.fail_batch:
                self.fail_batch = False
                super().executemany(sql, rows[:1])
                raise sqlite3.OperationalError("injected failure")
            return super().executemany(sql, rows)

    path = tmp_path / "ledger.db"
    conn = sqlite3.connect(path, factory=FailingConnection, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(LEDGER_DDL)
    ledger = ConscienceLedger("user-1")
    ledger._conn = conn
    first = ledger.record(Verdict(decision="caution"))
    second = ledger.record(Verdict(decision="refuse"))

    conn.fail_batch = True
    assert ledger.flush() == 0
    assert len(ledger._pending) == 2
    with sqlite3.connect(path) as reader:
        assert reader.execute("SELECT count(*) FROM conscience_ledger").fetchone()[0] == 0

    assert ledger.stats()["total"] == 2
    assert {row["id"] for row in ledger.harvest()} == {first, second}
    assert ledger.set_gold(first, 0.3, 0.4)
    assert len(ledger.harvest(only_gold=True)) == 1
    ledger.close()


@pytest.mark.parametrize("action", ["pending", "resolve"])
def test_ledger_reads_and_updates_wait_for_batch_commit(tmp_path, action):
    from cognition.conscience.ledger import ConscienceLedger
    from cognition.conscience.schema import LEDGER_DDL, Verdict

    started = threading.Event()
    release = threading.Event()
    reading_started = threading.Event()

    class BlockingConnection(sqlite3.Connection):
        def executemany(self, sql, rows):
            result = super().executemany(sql, rows)
            started.set()
            assert release.wait(3)
            return result

    path = tmp_path / "ledger.db"
    conn = sqlite3.connect(path, factory=BlockingConnection, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(LEDGER_DDL)
    ledger = ConscienceLedger("user-1")
    ledger._conn = conn
    row_id = ledger.record(Verdict(decision="escalate"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        flushing = pool.submit(ledger.flush)
        assert started.wait(3)
        def access_pending():
            reading_started.set()
            if action == "pending":
                return ledger.pending()
            return ledger.resolve(row_id, "approved")

        reading = pool.submit(access_pending)
        assert reading_started.wait(3)
        assert not reading.done()
        release.set()
        assert flushing.result(timeout=3) == 1
        result = reading.result(timeout=3)
        if action == "pending":
            assert [row["id"] for row in result] == [row_id]
        else:
            assert result is True
    ledger.close()


def test_dopamine_debounce_expires_and_serializes_same_user(monkeypatch):
    from cognition import fly_registry
    from cognition.flymemory import dopamine

    now = [0.0]
    calls = []
    mb = SimpleNamespace(reinforce=lambda *_args: calls.append(True) or 0.2)
    monkeypatch.setattr(dopamine, "_mb_mode", lambda: "live")
    monkeypatch.setattr(dopamine.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(fly_registry, "get_flymb", lambda _uid: mb)
    monkeypatch.setattr(dopamine, "_LAST_PULSE", {})
    kc = np.array([1.0, 0.0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: dopamine.pulse(0.4, user_id="dedup-user", kc=kc), range(2)))
    assert sorted(result["reason"] for result in results) == ["dedup", "ok"]
    assert len(calls) == 1
    now[0] = 3.0
    assert dopamine.pulse(0.4, user_id="dedup-user", kc=kc)["applied"] is True
    assert len(calls) == 2


@pytest.mark.parametrize("credit_result", [None, {}, {"flushed": True}])
def test_direct_teaching_flushes_when_credit_does_not(monkeypatch, credit_result):
    from cognition import fly_registry, neural_state
    from cognition.flymemory import dopamine, eligibility, online_teach, teach_api

    flushed = []
    mb = SimpleNamespace(encode=lambda _: np.array([1.0]), valence_bias=lambda _: 0.2)
    store = SimpleNamespace(flush_mb=lambda _mb: flushed.append(True))
    monkeypatch.setattr(online_teach, "_mode", lambda: "live")
    monkeypatch.setattr(teach_api, "extract_topic", lambda _: None)
    monkeypatch.setattr(fly_registry, "get_flymb", lambda _: mb)
    monkeypatch.setattr(fly_registry, "get_fly_store", lambda _: store)
    monkeypatch.setattr(dopamine, "pulse", lambda *_args, **_kwargs: {"reason": "ok", "applied": True, "delta": 0.2})
    monkeypatch.setattr(neural_state, "get_neural_state", lambda _: SimpleNamespace(publish_mb=lambda *_args, **_kwargs: None, record_influence=lambda _: None))
    if credit_result is None:
        def fail_credit(*_args):
            raise RuntimeError("credit unavailable")
        monkeypatch.setattr(eligibility, "assign_credit", fail_credit)
    else:
        monkeypatch.setattr(eligibility, "assign_credit", lambda *_args: credit_result)

    online_teach.teach_from_user_text("thanks!", user_id="direct-user")
    online_teach.teach_interrupt_honored(user_id="direct-user")
    assert len(flushed) == (0 if credit_result and credit_result.get("flushed") else 2)


def test_episode_direct_writes_and_read_only_touches_commit(tmp_path):
    from cognition.memory.episode import EpisodicStore

    path = tmp_path / "episodes.db"
    store = EpisodicStore(str(path), user_id="episode-user")
    timestamp = "2026-09-24T12:00:00+00:00"
    first = store.bind(timestamp=timestamp, trace="We walked through a beautiful forest.")
    with sqlite3.connect(path) as reader:
        assert reader.execute("SELECT trace FROM emc_staging WHERE id = ?", (first,)).fetchone()[0].startswith("We walked")

    second = store.ingest_turn("I will hike with Ashley through the green forest tomorrow.",
                               "I will remember the hike with Ashley through the forest tomorrow.")
    assert second > 0
    with sqlite3.connect(path) as reader:
        assert reader.execute(
            "SELECT count(*) FROM emc_staging WHERE id = ?", (second,)
        ).fetchone()[0] == 1

    store._conn.execute(
        "INSERT INTO emc_storage (id, user_id, timestamp, date, trace) VALUES (1, ?, ?, ?, ?)",
        ("episode-user", timestamp, "2026-09-24", "We walked through a forest."),
    )
    store._conn.commit()
    store._touch_episodes([1])
    assert store._touch_timer is not None
    store._touch_timer.cancel()
    store._touch_timer.function()
    with sqlite3.connect(path) as reader:
        assert reader.execute("SELECT recall_count FROM emc_storage WHERE id = 1").fetchone()[0] == 1
    store._touch_episodes([1])
    store.close()
    with sqlite3.connect(path) as reader:
        assert reader.execute("SELECT recall_count FROM emc_storage WHERE id = 1").fetchone()[0] == 2


def test_knowledge_access_flushes_below_threshold(monkeypatch):
    from cognition.knowledge import search

    written = []
    monkeypatch.setattr(search, "_write_access_counts", lambda uid, ids: written.append((uid, ids)))
    search._increment_access_count(["one"], "access-user")
    assert search._ACCESS_TIMERS["access-user"] is not None
    search._ACCESS_TIMERS["access-user"].cancel()
    search._ACCESS_TIMERS["access-user"].function("access-user")
    assert written == [("access-user", ["one"])]

    monkeypatch.setattr(search, "_ACCESS_FLUSH_THRESHOLD", 2)
    search._increment_access_count(["one", "two"], "access-user")
    assert written[-1] == ("access-user", ["one", "two"])
    assert "access-user" not in search._ACCESS_TIMERS


def test_embed_query_cache_separates_geometry_and_flyal_mode(monkeypatch):
    from cognition import reason
    from cognition.memory.memorize import _MemoryBackend
    from cognition.memory import vecstore

    mode = ["off"]
    calls = []
    monkeypatch.setattr(vecstore, "_flyal_mode", lambda: mode[0])
    reason._embed_query_cache.clear()
    first = vecstore.HarrierEmbedder(base_url="http://unused", model="one", dims=2)
    second = vecstore.HarrierEmbedder(base_url="http://unused", model="one", dims=2)
    backend = object.__new__(_MemoryBackend)
    backend._embedder = first
    monkeypatch.setattr(first, "embed_query", lambda *_args, **_kwargs: calls.append("first") or np.array([1.0, 0.0]))
    monkeypatch.setattr(second, "embed_query", lambda *_args, **_kwargs: calls.append("second") or np.array([0.0, 1.0]))

    assert backend._embed("hello", query=True) == [1.0, 0.0]
    backend._embedder = second
    assert backend._embed("hello", query=True) == [1.0, 0.0]
    second.model = "two"
    assert backend._embed("hello", query=True) == [0.0, 1.0]
    mode[0] = "live"
    assert reason.cached_embed_query(first, "hello").tolist() == [1.0, 0.0]
    assert calls == ["first", "second", "first"]
    reason._embed_query_cache.clear()
