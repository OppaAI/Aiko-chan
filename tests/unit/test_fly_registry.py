"""Isolation and migration tests for the fly-brain registry."""
from __future__ import annotations

import sqlite3

import numpy as np

from cognition import fly_registry
from cognition.flymemory.store import PlasticityStore


def _write_v2_database(path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """CREATE TABLE mb_plastic
               (pre INTEGER NOT NULL, post INTEGER NOT NULL, delta REAL NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY (pre, post))"""
        )
        conn.execute(
            """CREATE TABLE cx_state
               (key TEXT PRIMARY KEY, value REAL NOT NULL, updated_at TEXT NOT NULL)"""
        )
        conn.execute("INSERT INTO mb_plastic VALUES (1, 2, 0.25, 'now')")
        conn.execute("INSERT INTO cx_state VALUES ('sleep_pressure', 0.75, 'now')")
        conn.commit()
    finally:
        conn.close()


def test_norm_id_uses_complete_identity() -> None:
    assert fly_registry._norm_id(None) == "default"
    assert fly_registry._norm_id("  ") == "default"
    assert fly_registry._norm_id("user/name") != fly_registry._norm_id("user_name")
    prefix = "x" * 128
    assert fly_registry._norm_id(prefix + "a") != fly_registry._norm_id(prefix + "b")


def test_exact_file_override_migrates_legacy_state(tmp_path, monkeypatch) -> None:
    legacy = tmp_path / "legacy.sqlite"
    _write_v2_database(legacy)
    monkeypatch.setenv("FLY_PLASTICITY_DB", str(legacy))
    monkeypatch.delenv("FLY_PLASTICITY_SHARED", raising=False)

    identity = "github/123"
    target = fly_registry._default_db_for(identity)

    assert target != legacy
    assert target.exists()
    assert not legacy.exists()
    store = PlasticityStore(target, identity=fly_registry._norm_id(identity))
    assert store.load_mb(10, 10) == {(1, 2): 0.25}
    assert store.load_cx() == 0.75


def test_previous_normalized_path_migrates(tmp_path, monkeypatch) -> None:
    identity = "github/user"
    legacy = tmp_path / f"fly_plasticity_{fly_registry._legacy_norm_id(identity)}.db"
    _write_v2_database(legacy)
    monkeypatch.setenv("FLY_PLASTICITY_DB", str(tmp_path))

    target = fly_registry._default_db_for(identity)

    assert target.exists()
    assert not legacy.exists()
    assert PlasticityStore(
        target, identity=fly_registry._norm_id(identity)
    ).load_cx() == 0.75


def test_shared_store_scopes_every_mb_and_cx_row(tmp_path) -> None:
    path = tmp_path / "shared.db"
    alice = PlasticityStore(path, identity="alice")
    bob = PlasticityStore(path, identity="bob")

    alice.save_mb(np.array([1, 2]), np.array([2, 3]), np.array([0.1, 0.2]))
    alice.save_cx(0.25)
    bob.save_mb(np.array([4]), np.array([5]), np.array([0.9]))
    bob.save_cx(0.8)

    assert alice.load_mb(10, 10) == {(1, 2): 0.1, (2, 3): 0.2}
    assert bob.load_mb(10, 10) == {(4, 5): 0.9}
    assert alice.load_cx() == 0.25
    assert bob.load_cx() == 0.8

    alice.save_mb(np.array([6]), np.array([7]), np.array([0.3]))
    assert alice.load_mb(10, 10) == {(6, 7): 0.3}
    assert bob.load_mb(10, 10) == {(4, 5): 0.9}


def test_flush_cx_snapshot_holds_identity_lock() -> None:
    identity = "flush-lock-test"
    key = fly_registry._norm_id(identity)
    lock = fly_registry.get_flycx_lock(identity)

    class Store:
        def flush_cx(self, value):
            assert lock._is_owned()
            assert value == 0.6

    class Compass:
        sleep_pressure = 0.6

    with fly_registry._lock:
        fly_registry._stores[key] = Store()
        fly_registry._cxs[key] = Compass()
    try:
        assert fly_registry.flush_all(identity)["cx"] is True
    finally:
        with fly_registry._lock:
            fly_registry._stores.pop(key, None)
            fly_registry._cxs.pop(key, None)
            fly_registry._cx_locks.pop(key, None)
            fly_registry._identity_inputs.pop(key, None)


def test_attention_steps_and_persists_under_identity_lock(monkeypatch) -> None:
    import cognition.attention as attention

    class CheckedLock:
        owned = False

        def __enter__(self):
            self.owned = True

        def __exit__(self, exc_type, exc, traceback):
            self.owned = False

    lock = CheckedLock()

    class Compass:
        bump = np.array([0.25, 0.75])
        sleep_pressure = 0.2

        def step(self, features, **kwargs):
            assert lock.owned
            self.bump = np.array([0.9, 0.1])
            self.sleep_pressure = 0.4
            return {
                "heading_deg": 0.0,
                "sharpness": 0.5,
                "decisiveness": 0.6,
                "pfl_drive": 0.0,
                "sleep_pressure": self.sleep_pressure,
            }

    class Store:
        def save_if_due_cx(self, value):
            assert lock.owned
            assert value == 0.4

    compass = Compass()
    monkeypatch.setattr(fly_registry, "get_flycx", lambda identity: compass)
    monkeypatch.setattr(fly_registry, "get_flycx_lock", lambda identity: lock)
    monkeypatch.setattr(fly_registry, "get_fly_store", lambda identity: Store())
    monkeypatch.setattr(attention, "MEMORY_FLYCX_MODE", "shadow")

    state = attention.EdgeCognitiveState("lock-test")
    attention.flycx_state_for_record(state, "is this locked?")
    decisiveness = attention.flycx_decisiveness_for_text("is this locked?")

    assert decisiveness == 0.6
    assert np.array_equal(compass.bump, np.array([0.9, 0.1]))
    assert compass.sleep_pressure == 0.4
