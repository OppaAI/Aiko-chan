"""Isolation and concurrency regressions for the fly neural-state wiring."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from cognition.fly_behavior import lateral_horn
from cognition.neural_state import NeuralState, clear_neural_state, get_neural_state, peek_neural_state


def test_lateral_horn_ignores_blank_identity(monkeypatch):
    monkeypatch.setattr(lateral_horn, "_mode", lambda: "live")
    lateral_horn._seen.clear()

    neutral = {"mode": "live", "familiarity": 0.5, "novel": False}
    assert lateral_horn.context_prior("same context", user_id=None) == neutral
    assert lateral_horn.context_prior("same context", user_id="   ") == neutral
    assert lateral_horn._seen == {}


def test_lateral_horn_uses_trimmed_identity(monkeypatch):
    monkeypatch.setattr(lateral_horn, "_mode", lambda: "live")
    lateral_horn._seen.clear()

    first = lateral_horn.context_prior("same context", user_id="  alice  ")
    second = lateral_horn.context_prior("same context", user_id="alice")

    assert first["novel"] is True
    assert second["novel"] is False
    assert set(lateral_horn._seen) == {"alice"}


def test_neural_state_lock_is_per_instance_and_not_serialized():
    first = NeuralState(user_id="first")
    second = NeuralState(user_id="second")

    assert first._instance_lock is not second._instance_lock
    assert "_instance_lock" not in asdict(first)
    assert "_instance_lock" not in first.snapshot()


def test_peek_neural_state_does_not_create_state():
    user_id = "peek-only"
    clear_neural_state(user_id)

    assert peek_neural_state(user_id) is None

    created = get_neural_state(user_id)
    assert peek_neural_state(user_id) is created


def test_neural_state_concurrent_publish_and_snapshot_remain_consistent():
    state = NeuralState(user_id="concurrent")

    def publish(value: float) -> None:
        for _ in range(500):
            state.publish_mb(value)
            if value > 0:
                state.publish_cx(
                    heading_deg=45.0,
                    sharpness=0.25,
                    decisiveness=0.75,
                    sleep_pressure=0.1,
                )
            else:
                state.publish_cx(
                    heading_deg=225.0,
                    sharpness=0.8,
                    decisiveness=0.2,
                    sleep_pressure=0.9,
                )

    def inspect() -> None:
        for _ in range(500):
            snapshot = state.snapshot()
            valence = snapshot["valence"]
            assert snapshot["approach"] == max(0.0, valence)
            assert snapshot["avoidance"] == max(0.0, -valence)
            cx_values = (
                snapshot["focus_heading"],
                snapshot["focus_sharpness"],
                snapshot["decisiveness"],
                snapshot["sleep_pressure"],
            )
            assert cx_values in {
                (0.0, 0.0, 0.5, 0.0),
                (45.0, 0.25, 0.75, 0.1),
                (225.0, 0.8, 0.2, 0.9),
            }

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(publish, -1.0), pool.submit(publish, 1.0)]
        futures.extend(pool.submit(inspect) for _ in range(2))
        for future in futures:
            future.result()
