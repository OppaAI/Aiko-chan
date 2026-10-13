"""Background fullbrain stepping (latency #6): the 470ms connectome
propagation runs off the turn's critical path, every turn.

Locked here:
  * _fullbrain_step_bg publishes the step's outputs for the next turn
  * outputs are keyed by user_id (no cross-user clobbering)
  * the overlap guard is released even on failure (no wedged lock)
  * a slow/failed step never raises into the turn path
  * turn-level: lock held -> whole_brain_throttled, no bg thread started
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cognition.fly_behavior import turn as turn_module


class FakeFullbrain:
    def __init__(self, delay: float = 0.0, fail: bool = False):
        self.delay = delay
        self.fail = fail
        self.calls = 0

    def step(self, kc, kc_body_ids):
        self.calls += 1
        if self.fail:
            raise RuntimeError("connectome exploded")
        if self.delay:
            time.sleep(self.delay)
        return {
            "dn_drive": 50e-6,
            "clock_drive": 0.12345,
            "arousal": 0.5,
            "n_seed": 10,
        }


@pytest.fixture(autouse=True)
def _clean_state():
    # Isolate module-level background state between tests.
    with turn_module._fullbrain_latest_lock:
        turn_module._fullbrain_latest.clear()
    # Ensure the bg lock starts free (a prior test's thread released it).
    deadline = time.monotonic() + 5.0
    while turn_module._fullbrain_bg_lock.locked():
        if time.monotonic() >= deadline:
            pytest.fail("bg lock still held from a previous test")
        time.sleep(0.02)
    yield
    with turn_module._fullbrain_latest_lock:
        turn_module._fullbrain_latest.clear()


def _run_bg_step(fb, user_id="u1"):
    assert turn_module._fullbrain_bg_lock.acquire(blocking=False)
    t = threading.Thread(
        target=turn_module._fullbrain_step_bg,
        args=(fb, np.zeros(8, dtype=np.float32), np.arange(8), user_id),
        daemon=True,
    )
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    return t


def test_bg_step_publishes_outputs_for_next_turn():
    fb = FakeFullbrain()
    _run_bg_step(fb, user_id="u1")
    assert fb.calls == 1
    # Outputs are published under this user's key for the next turn.
    with turn_module._fullbrain_latest_lock:
        latest = dict(turn_module._fullbrain_latest.get("u1") or {})
    assert latest["motor_vigor"] == pytest.approx(1.0, abs=0.01)
    assert latest["clock_drive"] == pytest.approx(0.12345)
    assert latest["brain_arousal"] == pytest.approx(0.5)
    assert latest["whole_brain_ms"] >= 0.0
    # Guard released by the bg thread.
    assert not turn_module._fullbrain_bg_lock.locked()


def test_bg_step_keyed_by_user_id_no_cross_user_clobber():
    fb = FakeFullbrain()
    _run_bg_step(fb, user_id="alice")
    _run_bg_step(fb, user_id="bob")
    with turn_module._fullbrain_latest_lock:
        assert set(turn_module._fullbrain_latest.keys()) == {"alice", "bob"}
        # Each user's entry is intact (not cleared by the other's step).
        assert turn_module._fullbrain_latest["alice"]["clock_drive"] == pytest.approx(0.12345)
        assert turn_module._fullbrain_latest["bob"]["clock_drive"] == pytest.approx(0.12345)


def test_bg_step_failure_releases_guard_and_never_raises():
    fb = FakeFullbrain(fail=True)
    _run_bg_step(fb, user_id="u1")
    # Guard released despite the exception; no outputs published.
    assert not turn_module._fullbrain_bg_lock.locked()
    with turn_module._fullbrain_latest_lock:
        assert turn_module._fullbrain_latest == {}


def test_overlap_guard_skips_when_step_running(monkeypatch):
    # Hold the guard (simulating a step in flight); a non-blocking
    # acquire must fail so apply_turn_priors skips instead of piling up.
    assert turn_module._fullbrain_bg_lock.acquire(blocking=False)
    try:
        assert not turn_module._fullbrain_bg_lock.acquire(blocking=False)

        # Turn-level: with the lock held, apply_turn_priors must set
        # whole_brain_throttled and must NOT start a bg thread.
        monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
        fake_mb = MagicMock()
        fake_mb.encode.return_value = np.zeros(8, dtype=np.float32)
        fake_mb.valence_bias.return_value = 0.0
        fake_mb.kc_body_ids = np.arange(8)
        started = []

        orig_thread = threading.Thread

        def _spy_thread(*a, **kw):
            if kw.get("name") == "aiko-fullbrain-step":
                started.append(True)
            return orig_thread(*a, **kw)

        monkeypatch.setattr(threading, "Thread", _spy_thread)
        with patch("cognition.fly_registry.get_flymb", return_value=fake_mb), \
             patch("cognition.fly_registry.get_fullbrain", return_value=FakeFullbrain()):
            out = turn_module.apply_turn_priors("hello", user_id="u1")
        assert out.get("whole_brain_throttled") is True
        assert out.get("whole_brain_bg") is None
        assert started == []
    finally:
        turn_module._fullbrain_bg_lock.release()


def test_thread_start_failure_releases_guard(monkeypatch):
    # If Thread() or start() raises, the bg lock must be released so
    # future turns are not wedged into whole_brain_throttled forever.
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    fake_mb = MagicMock()
    fake_mb.encode.return_value = np.zeros(8, dtype=np.float32)
    fake_mb.valence_bias.return_value = 0.0
    fake_mb.kc_body_ids = np.arange(8)

    def _boom(*a, **kw):
        raise RuntimeError("cannot start thread")

    monkeypatch.setattr(threading, "Thread", _boom)
    with patch("cognition.fly_registry.get_flymb", return_value=fake_mb), \
         patch("cognition.fly_registry.get_fullbrain", return_value=FakeFullbrain()):
        # Must not raise into the turn; the outer handler logs and continues.
        out = turn_module.apply_turn_priors("hello", user_id="u1")
    # Lock released despite the failure.
    assert not turn_module._fullbrain_bg_lock.locked()
    # No bg thread was marked started.
    assert out.get("whole_brain_bg") is None
