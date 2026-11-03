"""Unit tests: episodic forgetting curve + recall hygiene. No services needed."""
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cognition.memory import episode as em


def _iso_days_ago(n: float) -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=n)).isoformat()


def test_decay_fresh_is_one():
    assert abs(em.decay_factor(_iso_days_ago(0), 0, 0.0) - 1.0) < 0.01


def test_decay_fades_with_age():
    assert em.decay_factor(_iso_days_ago(90), 0, 0.0) < 0.5


def test_decay_recall_resists():
    old, rec = _iso_days_ago(90), _iso_days_ago(90)
    assert em.decay_factor(old, 20, 0.0) > em.decay_factor(rec, 0, 0.0)


def test_decay_salience_slows():
    old = _iso_days_ago(90)
    assert em.decay_factor(old, 0, 1.0) > em.decay_factor(old, 0, 0.0)


def test_decay_never_zero_or_raises():
    assert em.decay_factor(None) == 1.0
    assert em.decay_factor("garbage") == 1.0
    assert em.decay_factor(_iso_days_ago(3650), 0, 0.0) >= 0.0


def test_decay_disabled_returns_one(monkeypatch):
    monkeypatch.setattr(em, "EMC_DECAY_HALF_LIFE_DAYS", 0.0)
    assert em.decay_factor(_iso_days_ago(365), 0, 0.0) == 1.0


def test_trivial_input_skipped_without_embed(tmp_path, monkeypatch):
    store = em.EpisodicStore(db_path=str(tmp_path / "emc.db"), user_id="u1",
                             embedder=object())
    called = []
    monkeypatch.setattr(store, "_conn", None, raising=False)
    # Must return before touching embedder/DB: greetings pay nothing.
    assert store.search("hi", user_id="u1") == []
    assert store.search("ok thanks!", user_id="u1") == []


def test_maybe_flush_cap_trigger(tmp_path, monkeypatch):
    store = em.EpisodicStore(db_path=str(tmp_path / "emc.db"), user_id="u1",
                             embedder=object())
    monkeypatch.setattr(store, "staging_count", lambda: em.EMC_STAGING_MAX + 5)
    monkeypatch.setattr(store, "flush_staging", lambda *a, **k: 7)
    assert store.maybe_flush() == 7


def test_retire_distilled_old_unrecalled(tmp_path, monkeypatch):
    from cognition.memory import episode as em
    import sqlite3
    db = tmp_path / "emc.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE emc_storage (id INTEGER PRIMARY KEY, user_id TEXT, timestamp TEXT, recall_count INTEGER DEFAULT 0, superseded_by TEXT)")
    old = "2020-01-01T00:00:00+00:00"
    new = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute("INSERT INTO emc_storage (id, user_id, timestamp, recall_count) VALUES (1, 'u', ?, 0)", (old,))
    conn.execute("INSERT INTO emc_storage (id, user_id, timestamp, recall_count) VALUES (2, 'u', ?, 50)", (old,))
    conn.execute("INSERT INTO emc_storage (id, user_id, timestamp, recall_count) VALUES (3, 'u', ?, 0)", (new,))
    conn.commit()
    store = mock_store(conn)
    batch = [{"id": 1, "timestamp": old, "recall_count": 0},
             {"id": 2, "timestamp": old, "recall_count": 50},
             {"id": 3, "timestamp": new, "recall_count": 0}]
    assert em._retire_distilled(store, batch, ["m1"]) == 1
    got = {r[0]: r[1] for r in conn.execute("SELECT id, superseded_by FROM emc_storage")}
    assert got[1] == "distilled:m1"   # old + unrecalled -> retired
    assert got[2] is None             # recalled -> kept
    assert got[3] is None             # fresh -> kept
    conn.close()


def mock_store(conn):
    class S:
        _lock = __import__("threading").RLock()
        _conn = conn
    return S()


def test_retire_disabled_or_empty(monkeypatch):
    from cognition.memory import episode as em
    assert em._retire_distilled(None, [], []) == 0
    monkeypatch.setattr(em, "EMC_SUPERSEDE_ENABLED", False)
    assert em._retire_distilled(object(), [{"id": 1}], ["m"]) == 0
