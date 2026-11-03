"""Urgency is an attention signal; cancellation requires an actual stop."""
from __future__ import annotations

import pytest

from cognition.fly_behavior import giant_fiber as gf
from cognition.fly_behavior.gf_global import should_cancel_tools, should_cancel_tts
from cognition.neural_state import NeuralState, clear_neural_state, get_neural_state


@pytest.mark.parametrize('text', [
    'Can you help me fix this code now?', 'This is urgent, please help me.',
    'How can I stop this Python process?', 'Cancel my subscription tomorrow.',
    'Please stop refusing to answer.', "Don't stop coding.",
    'Explain what the word "stop" means.', 'Sorry I did something wrong.',
    'I will bring her to see autumn leaves tomorrow, or if she wanna stay home to do coding?',
])
def test_ordinary_requests_do_not_cancel(text, monkeypatch):
    monkeypatch.setenv('MEMORY_FLYGF_MODE', 'live')
    out = gf.assess_interrupt(text)
    assert not out['interrupt'], out
    assert not out['would_interrupt']


@pytest.mark.parametrize('text', ['stop', 'Please stop.', 'Aiko, stop talking now!',
    'cancel the current task', 'abort this plan', 'wait a second', 'やめて', '멈춰'])
def test_explicit_stop_still_cancels(text, monkeypatch):
    monkeypatch.setenv('MEMORY_FLYGF_MODE', 'live')
    assert gf.assess_interrupt(text)['interrupt'] is True


def test_urgency_threshold_is_configurable_without_disabling_stop(monkeypatch):
    monkeypatch.setenv('MEMORY_FLYGF_MODE', 'live')
    monkeypatch.setenv('MEMORY_FLYGF_CANCEL_THRESHOLD', '0.85')
    assert not gf.assess_interrupt('help', priority=0.7)['interrupt']
    assert gf.assess_interrupt('', priority=0.9)['interrupt']
    assert gf.assess_interrupt('', system_error=True)['interrupt']
    assert not gf.assess_interrupt('', motion_salience=1, motion_sudden=True,
                                   subliminal_urgency=1, voice_urgency=0.45)['interrupt']
    monkeypatch.setenv('MEMORY_FLYGF_CANCEL_THRESHOLD', '0.95')
    assert not gf.assess_interrupt('', priority=0.9)['interrupt']
    assert gf.assess_interrupt('stop')['interrupt']


def test_decaying_urgency_cannot_rearm_cancel_and_identities_stay_isolated(monkeypatch):
    monkeypatch.setenv('MEMORY_FLYGF_MODE', 'live')
    uid = 'cancel-intent-test'
    other = 'cancel-other-test'
    clear_neural_state(uid)
    clear_neural_state(other)
    try:
        st = get_neural_state(uid)
        st.publish_gf(1.0, False, source='turn+cx')
        assert not gf.should_abort_plan(uid)
        assert not should_cancel_tools(uid)
        assert not should_cancel_tts(uid)
        st.publish_gf(1.0, True, source='turn')
        assert should_cancel_tools(uid)
        assert should_cancel_tts(uid)
        assert not should_cancel_tools(other)
        # A subsequent ordinary turn releases the stop, even if CX remains high.
        st.publish_gf(0.95, False, source='turn+cx')
        assert not should_cancel_tools(uid)
    finally:
        clear_neural_state(uid)
        clear_neural_state(other)


def test_shadow_computes_stop_but_never_cancels(monkeypatch):
    monkeypatch.setenv('MEMORY_FLYGF_MODE', 'shadow')
    out = gf.assess_interrupt('stop')
    assert out['would_interrupt'] and not out['interrupt']
