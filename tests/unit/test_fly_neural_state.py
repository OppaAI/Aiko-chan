"""Isolation and concurrency regressions for the fly neural-state wiring."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from cognition.fly_behavior import lateral_horn
from cognition.fly_behavior import sleep_sched
from cognition.neural_state import NeuralState, clear_neural_state, get_neural_state, peek_neural_state


def _clear_seen():
    lateral_horn._seen_fp.clear()
    lateral_horn._seen_vec.clear()


def _seen_users():
    return set(lateral_horn._seen_fp) | set(lateral_horn._seen_vec)


def test_lateral_horn_ignores_blank_identity(monkeypatch):
    monkeypatch.setattr(lateral_horn, "_mode", lambda: "live")
    _clear_seen()

    neutral = {"mode": "live", "familiarity": 0.5, "novel": False}
    assert lateral_horn.context_prior("same context", user_id=None) == neutral
    assert lateral_horn.context_prior("same context", user_id="   ") == neutral
    assert lateral_horn._seen_fp == {} and lateral_horn._seen_vec == {}


def test_lateral_horn_uses_trimmed_identity(monkeypatch):
    monkeypatch.setattr(lateral_horn, "_mode", lambda: "live")
    _clear_seen()

    first = lateral_horn.context_prior("same context", user_id="  alice  ")
    second = lateral_horn.context_prior("same context", user_id="alice")

    assert first["novel"] is True
    assert second["novel"] is False
    assert _seen_users() == {"alice"}


def test_lateral_horn_read_only_lookup_does_not_record(monkeypatch):
    monkeypatch.setattr(lateral_horn, "_mode", lambda: "live")
    _clear_seen()

    prior = lateral_horn.context_prior("unseen context", user_id="alice", record=False)

    assert prior["novel"] is True
    assert lateral_horn._seen_fp == {} and lateral_horn._seen_vec == {}


def test_shadow_maintenance_logs_pressure_prediction_without_acting(monkeypatch, caplog):
    user_id = "sleep-shadow"
    clear_neural_state(user_id)
    get_neural_state(user_id).sleep_pressure = 0.8
    monkeypatch.setattr(sleep_sched, "_mode", lambda: "shadow")
    monkeypatch.setenv("MEMORY_FLYSLEEP_MODE", "shadow")

    with caplog.at_level("DEBUG", logger="aiko.fly_behavior.sleep_sched"):
        assert sleep_sched.should_prefer_maintenance(user_id) is False

    assert "would=maintenance action_level=normal" in caplog.text


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


def test_invalid_circuit_signals_return_to_neutral_baselines():
    import math
    for invalid in (float('nan'), float('inf'), -float('inf'), None, 'invalid'):
        state = NeuralState(user_id='invalid-signal')
        state.publish_mb(invalid)
        state.publish_cx(heading_deg=invalid, sharpness=invalid, decisiveness=invalid, sleep_pressure=invalid)
        state.publish_dn(arousal=invalid, rate_mult=invalid)
        state.publish_lh(invalid)
        state.publish_gf(invalid)
        state.publish_motion(invalid)
        state.publish_sensory_gain(invalid)
        state.publish_circadian(invalid)
        snapshot = state.snapshot()
        assert snapshot['urgency'] == 0.0
        assert snapshot['sleep_pressure'] == 0.0
        assert snapshot['valence'] == 0.0
        assert snapshot['decisiveness'] == 0.5
        assert snapshot['motor_vigor'] == 1.0
        assert snapshot['sensory_gain'] == 1.0
        assert snapshot['context_familiarity'] == 0.5
        assert all(math.isfinite(value) for value in snapshot.values() if isinstance(value, float))


def test_neural_trace_owns_nested_input_and_output_values():
    state = NeuralState(user_id='trace-isolation')
    event = {'kind': 'signal', 'values': {'sources': ['mb']}}
    state.record_influence(event)
    event['values']['sources'].append('external-mutation')
    snapshot = state.snapshot()
    assert snapshot['influence'][0]['values']['sources'] == ['mb']
    snapshot['influence'][0]['values']['sources'].append('consumer-mutation')
    assert state.snapshot()['influence'][0]['values']['sources'] == ['mb']


def test_invalid_priority_cannot_trigger_gf_cancellation(monkeypatch):
    from cognition.fly_behavior import giant_fiber
    monkeypatch.setattr(giant_fiber, '_mode', lambda: 'live')
    for value in (float('nan'), float('inf'), -float('inf'), 'invalid', None):
        result = giant_fiber.assess_interrupt('Let us see the autumn leaves tomorrow.', priority=value,
                                             voice_urgency=value, motion_salience=value, subliminal_urgency=value)
        assert not result['interrupt']
        assert result['urgency'] == 0.0
    assert giant_fiber.assess_interrupt('stop')['interrupt']
    assert giant_fiber.assess_interrupt('system task', priority=1.0)['interrupt']


def test_unseen_fingerprint_is_not_reported_as_familiar(monkeypatch):
    from cognition.memory import preference_store
    monkeypatch.setattr(lateral_horn, '_mode', lambda: 'live')
    monkeypatch.setattr(preference_store, '_cached_vec', lambda text: None)
    _clear_seen()
    first = lateral_horn.context_prior('Autumn leaves and a coding afternoon', user_id='owner')
    second = lateral_horn.context_prior('Autumn leaves and a coding afternoon', user_id='owner')
    assert first['novel'] and first['familiarity'] < .5 and first['bias'] < 0
    assert not second['novel'] and second['familiarity'] > .5 and second['bias'] > 0


def test_invalid_familiarity_vectors_do_not_poison_history(monkeypatch):
    from cognition.memory import preference_store
    monkeypatch.setattr(lateral_horn, '_mode', lambda: 'live')
    monkeypatch.setattr(preference_store, '_cached_vec', lambda text: [float('nan'), 1.0])
    _clear_seen()
    prior = lateral_horn.context_prior('A quiet new context', user_id='owner')
    assert prior['method'] == 'fingerprint'
    assert lateral_horn._seen_vec == {}
    assert lateral_horn._cosine([float('nan'), 1.], [1., 1.]) == 0.0


def test_working_memory_score_does_not_record_familiarity(monkeypatch):
    from cognition.memory import grasp, preference_store
    monkeypatch.setattr(lateral_horn, '_mode', lambda: 'live')
    monkeypatch.setattr(preference_store, '_cached_vec', lambda text: None)
    monkeypatch.setattr(grasp, 'MEMORY_FLYMB_MODE', 'off')
    _clear_seen()
    turn = grasp.GraspTurn(user='autumn leaves', assistant='a quiet walk', tokens=10)
    first = grasp.compute_score(turn, 1, user_id='readonly-score')
    second = grasp.compute_score(turn, 1, user_id='readonly-score')
    assert first == second
    assert 'readonly-score' not in _seen_users()
