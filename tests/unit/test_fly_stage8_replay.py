"""Phase 8 — offline replay <-> Memory Bank (tests).

Covers: eligibility scoring math (bounds, determinism, decay), candidate
selection (ordering, cap, min-elig cutoff), mode gating (off/shadow/live),
shadow-no-write, live application through the real MB circuit, scheduler
wiring, and a DB-backed integration test with a temp episodic store.
"""
import os
import sqlite3
import time

import pytest

from cognition.flymemory import replay


def _uid(tag):
    return f"fly-replay-test-{tag}"


# ── scoring math ────────────────────────────────────────────────────────────

def test_score_candidate_bounds():
    for age in (0.0, 1.0, 12.0, 100.0):
        for val in (-1.0, -0.3, 0.0, 0.5, 1.0):
            for sal in (None, 0.0, 0.5, 2.0):
                s = replay.score_candidate(age, val, sal)
                assert 0.0 <= s <= 1.0, (age, val, sal, s)


def test_score_candidate_determinism():
    a = replay.score_candidate(2.5, 0.6, 0.4)
    b = replay.score_candidate(2.5, 0.6, 0.4)
    assert a == b


def test_score_candidate_recency_decay_monotonic():
    scores = [replay.score_candidate(age, 0.8, 0.0) for age in (0, 3, 6, 12, 24)]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1] > 0.0


def test_score_candidate_zero_signal_is_zero():
    assert replay.score_candidate(0.0, 0.0, None) == 0.0
    assert replay.score_candidate(0.0, 0.0, 0.0) == 0.0


def test_score_candidate_garbage_never_raises():
    assert replay.score_candidate("x", None, object()) == 0.0


# ── selection ───────────────────────────────────────────────────────────────

def _ep(eid, age_h, trace="some trace text"):
    return {"id": eid, "trace": trace, "salience": 0.5, "age_h": age_h}


def test_select_candidates_ordering_deterministic():
    eps = [_ep(3, 1.0), _ep(1, 10.0), _ep(2, 1.0)]
    val = lambda t: 0.9  # noqa: E731
    first = [c["id"] for c in replay.select_candidates(eps, valence_of=val)]
    second = [c["id"] for c in replay.select_candidates(eps, valence_of=val)]
    assert first == second
    # ids 2 and 3 tie on elig (same age/valence) -> id asc; id 1 is older
    assert first == [2, 3, 1]


def test_select_candidates_cap(monkeypatch):
    monkeypatch.setenv("FLY_REPLAY_MAX_ITEMS", "2")
    eps = [_ep(i, 0.5) for i in range(10)]
    out = replay.select_candidates(eps, valence_of=lambda t: 0.9)
    assert len(out) == 2


def test_select_candidates_min_elig_cutoff(monkeypatch):
    monkeypatch.setenv("FLY_REPLAY_MIN_ELIG", "0.99")
    eps = [_ep(1, 20.0)]
    out = replay.select_candidates(eps, valence_of=lambda t: 0.1)
    assert out == []


def test_select_candidates_skips_valence_errors():
    def bad(t):
        raise RuntimeError("nope")
    assert replay.select_candidates([_ep(1, 0.5)], valence_of=bad) == []


# ── mode gating ─────────────────────────────────────────────────────────────

def test_replay_mode_default_shadow(monkeypatch):
    monkeypatch.delenv("AIKO_FLY_REPLAY_MODE", raising=False)
    assert replay.replay_mode() == "shadow"


def test_replay_mode_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("AIKO_FLY_REPLAY_MODE", "turbo")
    assert replay.replay_mode() == "shadow"


def test_run_replay_off_short_circuits(monkeypatch):
    monkeypatch.setenv("AIKO_FLY_REPLAY_MODE", "off")
    out = replay.run_replay(_uid("off"))
    assert out["ran"] is False
    assert out["reason"] == "mode_off"
    assert out["n_replayed"] == 0


def test_run_replay_needs_mb_live(monkeypatch):
    monkeypatch.setenv("AIKO_FLY_REPLAY_MODE", "live")
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "off")
    out = replay.run_replay(_uid("mboff"))
    assert out["ran"] is False
    assert out["reason"] == "mb_not_live"


def test_run_replay_never_raises(monkeypatch):
    monkeypatch.setenv("AIKO_FLY_REPLAY_MODE", "live")
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setattr(replay, "_recent_episodes",
                        lambda uid, now: (_ for _ in ()).throw(RuntimeError("db gone")))
    out = replay.run_replay(_uid("boom"))
    assert isinstance(out, dict)
    assert out["ran"] is False


# ── shadow vs live through the real MB ──────────────────────────────────────

def _teach(mb, text, reward, n=5):
    from cognition.flymemory.circuit import text_features
    feats = text_features(text)
    for _ in range(n):
        mb.reinforce(mb.encode(feats), reward)


def _plastic_mass(mb):
    import numpy as np
    return float(np.abs(mb._plastic).sum())


def test_shadow_computes_but_writes_nothing(monkeypatch):
    uid = _uid("shadow")
    monkeypatch.setenv("AIKO_FLY_REPLAY_MODE", "shadow")
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    from cognition.fly_registry import get_flymb
    mb = get_flymb(uid)
    _teach(mb, "replay shadow probe alpha", +1.0)
    before = _plastic_mass(mb)
    eps = [_ep(1, 0.5, "replay shadow probe alpha"),
           _ep(2, 1.0, "replay shadow probe beta")]
    monkeypatch.setattr(replay, "_recent_episodes", lambda u, now: eps)
    out = replay.run_replay(uid)
    assert out["ran"] is True
    assert out["mode"] == "shadow"
    assert out["n_replayed"] == 0
    assert all(it["reason"] == "would_replay" for it in out["items"])
    assert _plastic_mass(mb) == before  # untouched
    assert replay.last_run(uid)["n_candidates"] == 2


def test_live_replay_strengthens_taught_pattern(monkeypatch, tmp_path):
    uid = _uid("live")
    monkeypatch.setenv("AIKO_FLY_REPLAY_MODE", "live")
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("FLY_PLASTICITY_DB", str(tmp_path))
    from cognition.fly_registry import get_flymb
    from cognition.flymemory.circuit import text_features
    mb = get_flymb(uid)
    text = "replay live probe gamma wonderful"
    _teach(mb, text, +1.0)
    before_mass = _plastic_mass(mb)
    before_bias = float(mb.valence_bias(text_features(text)))
    monkeypatch.setattr(replay, "_recent_episodes",
                        lambda u, now: [_ep(1, 0.5, text)])
    out = replay.run_replay(uid)
    assert out["ran"] is True
    assert out["n_replayed"] == 1
    assert _plastic_mass(mb) > before_mass
    after_bias = float(mb.valence_bias(text_features(text)))
    # Re-teaching a positive pattern must not flip it negative.
    assert after_bias >= before_bias - 1e-9
    assert after_bias > 0


def test_live_replay_no_episodes(monkeypatch):
    uid = _uid("empty")
    monkeypatch.setenv("AIKO_FLY_REPLAY_MODE", "live")
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setattr(replay, "_recent_episodes", lambda u, now: [])
    out = replay.run_replay(uid)
    assert out["ran"] is False
    assert out["reason"] == "no_episodes"


# ── DB-backed integration ───────────────────────────────────────────────────

def _make_episode_db(path, user_id):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE emc_storage (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL,"
        " timestamp TEXT NOT NULL, date TEXT NOT NULL, trace TEXT NOT NULL,"
        " valence_tag TEXT, arousal_score REAL, salience_score REAL,"
        " superseded_by INTEGER)"
    )
    now = time.time()
    rows = [
        # (id, age_h, trace, salience, superseded_by)
        (1, 1.0, "db replay probe delta fantastic", 0.9, None),
        (2, 2.0, "db replay probe epsilon terrible", 0.8, None),
        (3, 0.5, "superseded and should be skipped", 1.0, 99),
        (4, 30.0, "too old to matter much", 0.1, None),
    ]
    from datetime import datetime, timezone
    for eid, age_h, trace, sal, sup in rows:
        ts = datetime.fromtimestamp(now - age_h * 3600, tz=timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO emc_storage (id, user_id, timestamp, date, trace,"
            " salience_score, superseded_by) VALUES (?,?,?,?,?,?,?)",
            (eid, user_id, ts, ts[:10], trace, sal, sup),
        )
    conn.commit()
    conn.close()


def test_integration_db_to_mb(monkeypatch, tmp_path):
    uid = _uid("dbinteg")
    db = tmp_path / "episodes.db"
    _make_episode_db(str(db), uid)
    monkeypatch.setattr(replay, "_db_path_for_user", lambda u: str(db))
    monkeypatch.setenv("AIKO_FLY_REPLAY_MODE", "live")
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("FLY_REPLAY_LOOKBACK_H", "24")
    monkeypatch.setenv("FLY_PLASTICITY_DB", str(tmp_path))
    from cognition.fly_registry import get_flymb
    mb = get_flymb(uid)
    _teach(mb, "db replay probe delta fantastic", +1.0)
    _teach(mb, "db replay probe epsilon terrible", -1.0)
    before = _plastic_mass(mb)
    out = replay.run_replay(uid)
    assert out["ran"] is True
    # id 3 superseded -> skipped; id 4 too old/weak -> below cutoff
    replayed_ids = {it["id"] for it in out["items"] if it["applied"]}
    assert replayed_ids == {1, 2}, out["items"]
    assert _plastic_mass(mb) > before
    # trail recorded on the neural state
    from cognition.neural_state import get_neural_state
    kinds = [e.get("kind") for e in get_neural_state(uid)._influence]
    assert "fly_replay" in kinds


# ── scheduler wiring ────────────────────────────────────────────────────────

def test_next_fly_replay_is_future():
    import system.schedule as sched
    from system import bioclock
    nxt = sched._next_fly_replay()
    assert nxt > bioclock.local_now()


def test_next_fly_replay_respects_env(monkeypatch):
    import system.schedule as sched
    from system import bioclock
    monkeypatch.setenv("FLY_REPLAY_HOUR", "3")
    monkeypatch.setenv("FLY_REPLAY_MINUTE", "15")
    nxt = sched._next_fly_replay()
    assert (nxt.hour, nxt.minute) == (3, 15)
    assert nxt > bioclock.local_now()


def test_run_fly_replay_no_users_ok(monkeypatch):
    import system.schedule as sched
    monkeypatch.setattr(sched, "all_user_ids", lambda: [])
    runner = sched.ScheduleRunner(user_id="github_alice")
    assert runner._run_fly_replay() is True


def test_run_fly_replay_never_raises(monkeypatch):
    import system.schedule as sched
    monkeypatch.setattr(sched, "all_user_ids", lambda: ["u1"])
    import cognition.flymemory.replay as rp
    monkeypatch.setattr(rp, "run_replay",
                        lambda uid: (_ for _ in ()).throw(RuntimeError("x")))
    runner = sched.ScheduleRunner(user_id="github_alice")
    assert runner._run_fly_replay() is False  # reports failure, no raise


def test_failed_fly_replay_advances_schedule(monkeypatch, caplog):
    import system.schedule as sched
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 9, 24, 23, 30, tzinfo=timezone.utc)
    next_replay = now + timedelta(days=1)
    monkeypatch.setattr(sched.bioclock, "local_now", lambda: now)
    monkeypatch.setattr(sched, "_next_fly_replay", lambda: next_replay)
    monkeypatch.setattr(sched, "all_user_ids", lambda: [])
    monkeypatch.setattr(sched, "acquire_busy", lambda **kwargs: True)
    monkeypatch.setattr(sched, "release_busy", lambda: None)
    monkeypatch.setattr(sched.ScheduleRunner, "_missing_reflection_dates", lambda self: [])
    monkeypatch.setattr(sched.ScheduleRunner, "_monthly_catchup_needed", lambda self: False)

    runner = sched.ScheduleRunner(user_id="github_alice")
    runner._next_daily = now + timedelta(days=2)
    runner._next_monthly = now + timedelta(days=2)
    runner._next_ledger_prune = now + timedelta(days=2)
    runner._next_fly_replay = now
    attempts = []
    monkeypatch.setattr(runner, "_run_fly_replay", lambda: attempts.append(True) and False)
    sleeps = []

    def wait_seconds(_wakeup, seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            runner.stop()

    monkeypatch.setattr(sched.bioclock, "wait_seconds", wait_seconds)
    runner._run()

    assert len(attempts) == 1
    assert runner._next_fly_replay == next_replay
    assert sleeps == [86400, 86400]
    assert "fly_replay failed" in caplog.text
