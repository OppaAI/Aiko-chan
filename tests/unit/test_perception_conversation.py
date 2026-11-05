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
    events.offer('perception-user', ('Thumb_Up',), 'Thumb up')
    events.offer('perception-user', ('Thumb_Up',), 'Thumb up')


def test_repeated_cue_can_initiate_without_new_user_turn(runner):
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


def test_gesture_can_recur_after_it_disappears(runner):
    signal()
    events.consume('perception-user')
    events.offer('perception-user', (), '')
    signal()
    assert events.peek('perception-user') is not None


def test_speech_generated_during_user_activity_is_not_delivered(runner):
    item, sent = runner
    def generate(_):
        item.touch()
        return 'old opportunity'
    item._generate_fn = generate
    item._deliver_prompt('perception_prompt', 'hint', O.time.monotonic())
    assert sent == []


def test_scene_change_requires_two_updates_without_cue_interference(runner):
    events.scene('perception-user', 'Laptop keyboard desk monitor coding computer')
    events.scene('perception-user', 'Outdoor garden flowers trees sunshine grass')
    assert events.peek('perception-user') is None
    events.scene('perception-user', 'Outdoor garden flowers trees sunshine grass')
    assert events.peek('perception-user') is not None
