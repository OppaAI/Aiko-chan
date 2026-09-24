"""Regression tests for Stage 6 STOP dispatch and output guards."""

from __future__ import annotations

import copy

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


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("speak", False),
        ("speak_synced", False),
        ("play_async", None),
        ("feed_and_play", None),
        ("start_speech_stream", None),
    ],
)
def test_all_playback_entry_points_share_cancellation_gate(monkeypatch, method, expected):
    from sensory import speak

    def unexpected_thread(*_args, **_kwargs):
        raise AssertionError("cancelled speech started a playback thread")

    monkeypatch.setattr(speak.threading, "Thread", unexpected_thread)
    speaker = speak.AikoSpeak(silent=True)
    gate_state = []
    consumed = []

    def playback_cancelled():
        gate_state.append((list(speaker._token_buf), list(consumed)))
        return True

    monkeypatch.setattr(speaker, "_playback_cancelled", playback_cancelled)
    if method == "play_async":
        speaker.feed("hello")
        result = speaker.play_async()
    elif method == "feed_and_play":
        def tokens():
            consumed.append("hello")
            yield "hello"
        result = speaker.feed_and_play(tokens())
    elif method == "start_speech_stream":
        result = speaker.start_speech_stream()
    else:
        result = getattr(speaker, method)("hello")

    assert result is expected
    expected_gate_state = [([], ["hello"])] if method == "feed_and_play" else [([], [])]
    assert gate_state == expected_gate_state
    assert not speaker._playing.is_set()
    assert not speaker._streaming_active


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
    jobs = [
        {"id": name, "next_due": "2000-01-01T00:00:00+00:00", "frequency": "once"}
        for name in ("current", "remaining")
    ]
    monkeypatch.setattr(schedule, "_read_all", lambda user_id=None: jobs)
    writes = []
    monkeypatch.setattr(schedule, "_write_all", lambda records, **_kwargs: writes.append(copy.deepcopy(records)))

    def cancel_after_queue(uid):
        calls.append(uid)
        return len(calls) > len(jobs)

    monkeypatch.setattr(gf_global, "should_cancel_scheduler", cancel_after_queue)
    runner = schedule.ScheduleRunner(on_due=lambda _event: pytest.fail("cancelled callback fired"), user_id=user_id)

    runner._fire_due_user_jobs_locked(user_id)

    assert calls == [user_id, user_id, user_id]
    assert writes == []
    for job in jobs:
        assert job.get("enabled", True) is True
        assert job["next_due"] == "2000-01-01T00:00:00+00:00"
        assert "last_ran_at" not in job


def test_due_callback_is_persisted_only_after_successful_dispatch(monkeypatch, tmp_path):
    from cognition.fly_behavior import gf_global
    from system import schedule

    user_id = "stage6-scheduler-success"
    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    job = {"id": "queued", "next_due": "2000-01-01T00:00:00+00:00", "frequency": "once"}
    writes = []
    monkeypatch.setattr(schedule, "_read_all", lambda user_id=None: [job])
    monkeypatch.setattr(schedule, "_write_all", lambda records, **_kwargs: writes.append(copy.deepcopy(records)))
    monkeypatch.setattr(gf_global, "should_cancel_scheduler", lambda _uid: False)

    def on_due(_event):
        assert job.get("enabled", True) is True
        assert job["next_due"] == "2000-01-01T00:00:00+00:00"
        assert "last_ran_at" not in job
        assert writes == []

    schedule.ScheduleRunner(on_due=on_due, user_id=user_id)._fire_due_user_jobs_locked(user_id)

    assert job["enabled"] is False
    assert "last_ran_at" in job
    assert writes == [[job]]


def test_failed_due_callback_remains_due(monkeypatch, tmp_path):
    from cognition.fly_behavior import gf_global
    from system import schedule

    user_id = "stage6-scheduler-failure"
    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    job = {"id": "queued", "next_due": "2000-01-01T00:00:00+00:00", "frequency": "once"}
    writes = []
    monkeypatch.setattr(schedule, "_read_all", lambda user_id=None: [job])
    monkeypatch.setattr(schedule, "_write_all", lambda records, **_kwargs: writes.append(copy.deepcopy(records)))
    monkeypatch.setattr(gf_global, "should_cancel_scheduler", lambda _uid: False)

    def fail(_event):
        raise RuntimeError("dispatch failed")

    schedule.ScheduleRunner(on_due=fail, user_id=user_id)._fire_due_user_jobs_locked(user_id)

    assert writes == []
    assert job.get("enabled", True) is True
    assert job["next_due"] == "2000-01-01T00:00:00+00:00"
    assert "last_ran_at" not in job


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


def test_sudden_motion_bumps_gf_without_changing_regular_salience(monkeypatch):
    from cognition.fly_behavior.giant_fiber import assess_interrupt

    monkeypatch.setenv("MEMORY_FLYGF_MODE", "live")
    regular = assess_interrupt("", motion_salience=0.6)
    sudden = assess_interrupt("", motion_salience=0.6, motion_sudden=True)

    assert regular["sources"]["motion"] == 0.5
    assert sudden["sources"]["motion"] == 0.7


@pytest.mark.parametrize("mode", ["off", "shadow", "live"])
def test_turn_only_uses_live_voice_and_sudden_motion(monkeypatch, mode):
    from cognition.fly_behavior import giant_fiber
    from cognition.fly_behavior.turn import apply_turn_priors
    from cognition.flysense import dn, pathways
    from cognition.neural_state import clear_neural_state

    user_id = f"stage6-senses-{mode}"
    clear_neural_state(user_id)
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "live")
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "off")
    monkeypatch.setenv("MEMORY_FLYGF_MODE", "live")
    monkeypatch.setattr(pathways, "encode_turn_senses", lambda *_args, **_kwargs: {
        "mode": mode,
        "voice": {"urgency": 0.4, "energy": 0.0},
        "motion": {"sudden": True},
    })
    gf_inputs = []
    real_assess = giant_fiber.assess_interrupt

    def capture_assess(*args, **kwargs):
        gf_inputs.append(kwargs)
        return real_assess(*args, **kwargs)

    monkeypatch.setattr(giant_fiber, "assess_interrupt", capture_assess)
    energies = []
    monkeypatch.setattr(dn.FlyDN, "drive", lambda self, **kwargs: energies.append(kwargs["energy"]) or {"rate_mult": 1.0})

    apply_turn_priors("hello", user_id=user_id)

    assert len(gf_inputs) == 1 and len(energies) == 1
    assert gf_inputs[0]["voice_urgency"] == (0.4 if mode == "live" else 0.0)
    assert gf_inputs[0]["motion_sudden"] is (mode == "live")
    assert energies[0] == (0.6 if mode == "live" else 1.0)
    clear_neural_state(user_id)
