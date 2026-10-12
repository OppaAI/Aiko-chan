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
