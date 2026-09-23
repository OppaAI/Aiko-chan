"""Regression tests for Stage 6 STOP dispatch and output guards."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("method", ["speak", "speak_synced", "start_speech_stream"])
def test_stop_prevents_speech_playback(monkeypatch, method):
    from cognition.neural_state import clear_neural_state, get_neural_state
    from sensory import speak
    from system import userspace

    user_id = "stage6-speech-cancel"
    clear_neural_state(user_id)
    get_neural_state(user_id).publish_gf(0.7, True)
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "live")
    monkeypatch.setenv("MEMORY_FLYGF_MODE", "live")
    monkeypatch.setattr(userspace, "current_user_id", lambda: user_id)

    def unexpected_thread(*_args, **_kwargs):
        raise AssertionError("cancelled speech started a playback thread")

    monkeypatch.setattr(speak.threading, "Thread", unexpected_thread)
    speaker = speak.AikoSpeak(silent=True)
    result = getattr(speaker, method)("hello") if method != "start_speech_stream" else speaker.start_speech_stream()

    assert result is (None if method == "start_speech_stream" else False)
    assert speaker._stop_flag.is_set()
    assert not speaker._playing.is_set()
    assert not speaker._streaming_active
    assert speaker._stream_queue is None
    clear_neural_state(user_id)


def test_cancelled_scheduled_jobs_remain_due_except_cleanup(monkeypatch, tmp_path):
    from cognition.neural_state import clear_neural_state, get_neural_state
    from system import schedule

    user_id = "stage6-scheduler-cancel"
    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    clear_neural_state(user_id)
    get_neural_state(user_id).publish_gf(0.7, True)
    monkeypatch.setenv("MEMORY_FLYGF_MODE", "live")
    jobs = [
        {"id": name, "handler": handler, "next_due": "2000-01-01T00:00:00+00:00", "frequency": "once"}
        for name, handler in (("normal", "test_handler"), ("cleanup", "deep_study_stop"), ("callback", None))
    ]
    fired = []
    monkeypatch.setattr(schedule, "_read_all", lambda user_id=None: jobs)
    monkeypatch.setattr(schedule, "_write_all", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(schedule._SYSTEM_HANDLERS, "test_handler", lambda _memory: fired.append("normal"))
    monkeypatch.setitem(schedule._SYSTEM_HANDLERS, "deep_study_stop", lambda _memory: fired.append("cleanup"))
    runner = schedule.ScheduleRunner(on_due=lambda event: fired.append(event.id), user_id=user_id)

    runner._fire_due_user_jobs_locked(user_id)

    assert fired == ["cleanup"]
    assert jobs[0].get("enabled", True) is True
    assert jobs[2].get("enabled", True) is True
    assert jobs[1]["enabled"] is False
    clear_neural_state(user_id)


def test_stop_after_queueing_does_not_dispatch_due_callback(monkeypatch, tmp_path):
    from cognition.fly_behavior import gf_global
    from system import schedule

    user_id = "stage6-scheduler-midflight"
    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    calls = []
    job = {"id": "queued", "next_due": "2000-01-01T00:00:00+00:00", "frequency": "once"}
    monkeypatch.setattr(schedule, "_read_all", lambda user_id=None: [job])
    monkeypatch.setattr(schedule, "_write_all", lambda *_args, **_kwargs: None)

    def cancel_after_queue(uid):
        calls.append(uid)
        return len(calls) > 1

    monkeypatch.setattr(gf_global, "should_cancel_scheduler", cancel_after_queue)
    runner = schedule.ScheduleRunner(on_due=lambda _event: pytest.fail("cancelled callback fired"), user_id=user_id)

    runner._fire_due_user_jobs_locked(user_id)

    assert calls == [user_id, user_id]


def test_scheduled_tool_checks_current_user_before_invocation(monkeypatch):
    from agentic import agentic
    from cognition.fly_behavior import gf_global

    calls = []
    monkeypatch.setattr(agentic, "current_user_id", lambda: "stage6-tool-user")
    monkeypatch.setattr(agentic, "_TOOLS", {"scheduled": ({}, lambda args: calls.append(args) or "completed")})
    monkeypatch.setattr(gf_global, "should_cancel_tools", lambda uid: calls.append(uid) or True)

    assert agentic.invoke_registered_tool("scheduled", {"arg": 1}) == {"skipped": True, "reason": "gf_interrupt"}
    assert calls == ["stage6-tool-user"]

    monkeypatch.setattr(gf_global, "should_cancel_tools", lambda uid: False)
    assert agentic.invoke_registered_tool("scheduled", {"arg": 1}) == "completed"
    assert calls[-1] == {"arg": 1}


def test_turn_publishes_speech_rate_not_action_vigor(monkeypatch):
    from cognition.flysense import dn
    from cognition.neural_state import clear_neural_state, get_neural_state
    from cognition.fly_behavior.turn import apply_turn_priors

    user_id = "stage6-dn-rate"
    clear_neural_state(user_id)
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "live")
    monkeypatch.setattr(dn.FlyDN, "drive", lambda self, **_kwargs: {"rate_mult": 0.9, "action_vigor": 1.4})

    apply_turn_priors("hello", user_id=user_id)

    assert get_neural_state(user_id).motor_vigor == 0.9
    clear_neural_state(user_id)
