"""Regression checks for identity/mode boundaries and valid zero signals."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from cognition import attention
from cognition.neural_state import clear_neural_state, get_neural_state


@pytest.mark.parametrize('mode', ['off', 'shadow', 'live'])
def test_cx_only_publishes_live_readouts(mode, monkeypatch):
    from cognition import fly_registry
    uid = 'cx-mode-boundary'
    clear_neural_state(uid)
    neural = get_neural_state(uid)
    neural.publish_cx(heading_deg=123, sharpness=.2, decisiveness=.3, sleep_pressure=.4)
    initial = neural.snapshot()
    state = SimpleNamespace(_identity=uid, _lock=threading.RLock(),
                            _turn_latencies=[], _affect=0., _energy=.8,
                            _uncertainty=.1, _open_loops=[], _goals=[])
    calls = []
    def step(*args, **kwargs):
        calls.append(True)
        return dict(heading_deg=45., sharpness=.9, decisiveness=.7, sleep_pressure=.6)
    monkeypatch.setattr(attention, 'MEMORY_FLYCX_MODE', mode)
    monkeypatch.setattr(fly_registry, 'get_flycx', lambda user_id: SimpleNamespace(step=step, bump=[.2, .8], sleep_pressure=.4))
    monkeypatch.setattr(fly_registry, 'get_flycx_lock', lambda user_id: threading.RLock())
    monkeypatch.setattr(fly_registry, 'get_fly_store', lambda user_id: None)
    try:
        out = attention.flycx_state_for_record(state, 'how is the task going?')
        actual = neural.snapshot()
        if mode == 'live':
            assert out is not None and calls
            assert actual['focus_heading'] == 45
            assert actual['sleep_pressure'] == .6
            assert state._energy < .8
        else:
            assert actual == initial  # Shadow cannot steer downstream DN/body consumers.
            assert state._energy == .8
            assert bool(calls) is (mode == 'shadow')
    finally:
        clear_neural_state(uid)


@pytest.mark.parametrize('uid', ['cx-routing-alice', 'cx-routing-bob'])
def test_attempt_gate_queries_the_callers_compass(uid, monkeypatch):
    seen = []
    def decisiveness(text, uncertainty=0., user_id=None):
        seen.append(user_id)
        return .5
    monkeypatch.setattr(attention, 'MEMORY_FLYCX_MODE', 'live')
    monkeypatch.setattr(attention, 'flycx_decisiveness_for_text', decisiveness)
    monkeypatch.setattr(attention._intent_classifier, 'predict', lambda _: (0., .5))
    state = attention.EdgeCognitiveState(uid)
    state.should_attempt('write a draft outline', mode='route')
    assert seen == [uid]


def test_zero_intent_probability_does_not_become_neutral(monkeypatch):
    monkeypatch.setattr(attention, 'MEMORY_FLYCX_MODE', 'off')
    ok, reason, action = attention.should_attempt(user_input='write a draft outline',
        mode='route', intent_confidence=.99, intent_success_prob=0.)
    assert not ok and action == 'clarify'
    assert '(0.0%)' in reason


def test_zero_energy_does_not_become_normal_energy(monkeypatch):
    monkeypatch.setattr(attention._intent_classifier, 'predict', lambda _: (0., .5))
    state = attention.EdgeCognitiveState('zero-energy-test')
    state._energy = 0.
    ok, reason, action = state.should_attempt('write a draft outline', mode='route')
    assert not ok and action == 'defer'
    assert 'energy low' in reason
