import threading
from types import SimpleNamespace

import pytest
from sensory import perception_events as events
from system import orchestrate as O
from system.userspace import set_current_user_id, reset_current_user_id


@pytest.fixture
def runner(monkeypatch):
    token = set_current_user_id('perception-user')
    monkeypatch.setattr(O, 'PROACTIVE_ENABLED', True)
    monkeypatch.setattr(O, 'PROACTIVE_USE_LLM', True)
    monkeypatch.setattr(O, '_in_proactive_silence_window', lambda: False)
    sent = []
    ui = SimpleNamespace(add_message=lambda *args: sent.append(args), _draw=lambda:None)
    result = O.ProactiveIdleRunner(ui, None, lambda:False, threading.Event(), generate_fn=lambda _: '')
    result._last_activity -= 20
    events.watching('perception-user', True)
    yield result, sent
    result.stop(); events.watching('perception-user', False)
    reset_current_user_id(token)


def signal():
    events.attention_ready('perception-user', 'open_palm')


def test_analyzed_attention_can_initiate_without_new_user_turn(runner):
    item, sent = runner
    signal()
    due = item._prompt_due(O.time.monotonic())
    assert due[0] == 'perception_prompt'
    item._deliver_prompt(*due, O.time.monotonic())
    assert sent == []  # Choosing silence must not produce a canned fallback.
    assert events.peek('perception-user') is None


def test_human_input_and_current_turn_take_priority(runner):
    item, _ = runner
    signal()
    item._active_turn.set()
    assert item._prompt_due(O.time.monotonic()) is None
    item._active_turn.clear()
    item.touch()
    assert events.peek('perception-user') is None


def test_stop_watching_discards_pending_opportunity(runner):
    signal()
    events.watching('perception-user', False)
    assert events.peek('perception-user') is None


def test_stop_watching_during_generation_discards_reply(runner):
    item, sent = runner
    def generate(_):
        events.watching('perception-user', False)
        return 'stale scene'
    item._generate_fn = generate
    item._deliver_prompt('perception_prompt', 'hint', O.time.monotonic())
    assert sent == []


def test_speech_generated_during_user_activity_is_not_delivered(runner):
    item, sent = runner
    def generate(_):
        item.touch()
        return 'old opportunity'
    item._generate_fn = generate
    item._deliver_prompt('perception_prompt', 'hint', O.time.monotonic())
    assert sent == []


def test_automatic_vlm_cooldown_survives_camera_restart(runner, monkeypatch):
    monkeypatch.setenv('CAMERA_ATTENTION_COOLDOWN_S', '90')
    # Production uses OrderedDict to bound entries.
    from collections import OrderedDict
    monkeypatch.setattr(events, '_attention_calls', OrderedDict())
    assert events.claim_attention('perception-user')
    events.watching('perception-user', False)
    events.watching('perception-user', True)
    assert not events.claim_attention('perception-user')
    assert not events.claim_attention('not-watching')


def test_ordinary_hand_cues_update_context_without_initiating(runner):
    from sensory import vision
    vision.note_cues('perception-user', {'gestures':['Thumb_Up'], 'posture':'arms_raised'})
    vision.note_cues('perception-user', {'gestures':['Thumb_Up'], 'posture':'arms_raised'})
    assert events.peek('perception-user') is None
    assert 'Thumb_Up' in vision.context('perception-user')


def test_invalid_or_stopped_attention_cannot_initiate(runner):
    events.attention_ready('perception-user', 'fist')
    assert events.peek('perception-user') is None
    events.watching('perception-user', False)
    signal()
    assert events.peek('perception-user') is None
