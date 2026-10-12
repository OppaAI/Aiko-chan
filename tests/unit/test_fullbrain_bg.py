"""Background fullbrain stepping (latency #6): the 470ms connectome
propagation runs off the turn's critical path, every turn.

Locked here:
  * _fullbrain_step_bg publishes the step's outputs for the next turn
  * the overlap guard is released even on failure (no wedged lock)
  * a slow/failed step never raises into the turn path
"""
from __future__ import annotations

import threading
import time

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


def test_bg_step_publishes_outputs_for_next_turn():
    fb = FakeFullbrain()
    # Simulate what apply_turn_priors does: acquire guard, run in bg.
    assert turn_module._fullbrain_bg_lock.acquire(blocking=False)
    t = threading.Thread(
        target=turn_module._fullbrain_step_bg,
        args=(fb, np.zeros(8, dtype=np.float32), np.arange(8), "u1"),
        daemon=True,
    )
    t0 = time.monotonic()
    t.start()
    t.join(timeout=5.0)
    dt = time.monotonic() - t0
    assert not t.is_alive()
    assert fb.calls == 1
    # Outputs are published for the next turn to consume.
    with turn_module._fullbrain_latest_lock:
        latest = dict(turn_module._fullbrain_latest)
    assert latest["motor_vigor"] == pytest.approx(1.0, abs=0.01)
    assert latest["clock_drive"] == pytest.approx(0.12345)
    assert latest["brain_arousal"] == pytest.approx(0.5)
    assert latest["whole_brain_ms"] >= 0.0
    # Guard released even though the caller acquired it (bg thread owns release).
    assert not turn_module._fullbrain_bg_lock.locked()
    assert dt < 5.0


def test_bg_step_failure_releases_guard_and_never_raises():
    fb = FakeFullbrain(fail=True)
    assert turn_module._fullbrain_bg_lock.acquire(blocking=False)
    t = threading.Thread(
        target=turn_module._fullbrain_step_bg,
        args=(fb, np.zeros(8, dtype=np.float32), np.arange(8), "u1"),
        daemon=True,
    )
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    # Guard released despite the exception; no outputs published.
    assert not turn_module._fullbrain_bg_lock.locked()
    with turn_module._fullbrain_latest_lock:
        assert turn_module._fullbrain_latest == {}


def test_overlap_guard_skips_when_step_running():
    # Hold the guard (simulating a step in flight); a non-blocking
    # acquire must fail so apply_turn_priors skips instead of piling up.
    assert turn_module._fullbrain_bg_lock.acquire(blocking=False)
    try:
        assert not turn_module._fullbrain_bg_lock.acquire(blocking=False)
    finally:
        turn_module._fullbrain_bg_lock.release()
