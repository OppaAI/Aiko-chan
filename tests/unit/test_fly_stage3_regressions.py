"""Regression tests for Stage 3 review fixes."""

from __future__ import annotations

import json
import multiprocessing


def _record_preference_in_process(topic: str) -> None:
    from cognition.memory.preference_store import record_preference

    record_preference(topic, "prefer", user_id="concurrent-user")


def test_dn_prosody_requires_live_mode_and_preserves_zero_controls(monkeypatch):
    from cognition import neural_state
    from cognition.fly_behavior.dn_tts import apply_dn_prosody

    class State:
        motor_vigor = 0.0
        action_drive = 0.0

    class Speaker:
        def __init__(self):
            self.expressions = []

        def set_expression(self, **values):
            self.expressions.append(values)

    calls = []
    monkeypatch.setattr(
        neural_state,
        "peek_neural_state",
        lambda user_id: calls.append(user_id) or State(),
    )
    speaker = Speaker()

    monkeypatch.setenv("MEMORY_FLYDN_MODE", "shadow")
    assert apply_dn_prosody(speaker, user_id="user-1") == {
        "applied": False,
        "rate": 1.0,
        "volume": 1.0,
    }
    assert calls == []

    monkeypatch.setenv("MEMORY_FLYDN_MODE", "live")
    assert apply_dn_prosody(speaker, user_id="user-1") == {
        "applied": True,
        "rate": 0.92,
        "volume": 0.9,
    }
    assert calls == ["user-1"]


def test_antiloop_preserves_an_explicit_zero_score(monkeypatch):
    from cognition.memory import antiloop

    monkeypatch.setattr(antiloop, "freshness_penalty", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(antiloop, "remember_shown", lambda *_args, **_kwargs: None)
    zero = {"text": "explicit zero score", "score": 0, "rank": -10}
    negative = {"text": "negative score", "score": -1}

    assert antiloop.apply_antiloop([zero, negative], weight=0.0) == [zero, negative]


def test_diversify_applies_user_scoped_antiloop_once_for_single_row(monkeypatch):
    from cognition.memory import antiloop
    from cognition.memory.diversity import diversify_and_freshness

    calls = []

    def fake_apply(rows, *, user_id=None, text_of=None, weight=None):
        calls.append((list(rows), user_id, text_of, weight))
        return list(rows)

    monkeypatch.setattr(antiloop, "apply_antiloop", fake_apply)
    row = {"memory": "one recalled memory"}

    assert diversify_and_freshness([row], user_id="user-2") == [row]
    assert len(calls) == 1
    assert calls[0][1] == "user-2"


def test_fly_rank_applies_preference_without_flymb(monkeypatch):
    from cognition.memory import fly_rank, preference_store

    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setattr(fly_rank, "mb_bias_for_text", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        preference_store,
        "preference_delta",
        lambda text, *, user_id=None: -0.08,
    )

    score, meta = fly_rank.adjust_recall_score(
        1.0,
        "fruit tarts",
        user_id=None,
        log_influence=False,
    )

    assert score == 0.92
    assert meta == {
        "mode": "live",
        "bias": None,
        "delta": -0.08,
        "weight": 0.0,
        "pref_delta": -0.08,
    }


def test_preference_path_passes_none_through(monkeypatch, tmp_path):
    from cognition.memory import preference_store
    from system import userspace

    seen = []
    monkeypatch.setattr(
        userspace,
        "user_state_dir",
        lambda user_id=None: seen.append(user_id) or tmp_path,
    )

    assert preference_store._path(None) == tmp_path / "fly_preferences.json"
    assert seen == [None]


def test_preference_updates_are_cross_process_safe(monkeypatch, tmp_path):
    from cognition.memory.preference_store import load_preferences

    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    topics = [f"topic-{i}" for i in range(12)]
    ctx = multiprocessing.get_context("fork")
    processes = [ctx.Process(target=_record_preference_in_process, args=(topic,)) for topic in topics]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    rows = load_preferences("concurrent-user")
    assert {row["topic"] for row in rows} == set(topics)
    path = tmp_path / "concurrent-user" / "fly_preferences.json"
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), list)


def test_speech_paths_pass_the_current_user_to_dn_prosody(monkeypatch):
    from cognition.fly_behavior import dn_tts
    from sensory import speak
    from system import userspace

    user_ids = []
    monkeypatch.setattr(userspace, "current_user_id", lambda: "authenticated-user")
    monkeypatch.setattr(
        dn_tts,
        "apply_dn_prosody",
        lambda _speaker, *, user_id=None: user_ids.append(user_id),
    )

    class Thread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(speak.threading, "Thread", Thread)
    speaker = speak.AikoSpeak(silent=True)
    monkeypatch.setattr(speaker, "stop", lambda: None)

    assert speaker.speak("hello") is True
    assert speaker.speak_synced("hello") is True
    assert user_ids == ["authenticated-user", "authenticated-user"]


def test_stage3_eval_gates_mb_and_gf_independently(monkeypatch, capsys):
    from cognition.fly_behavior import giant_fiber
    from cognition.flymemory import teach_api
    from cognition.memory import fly_rank, preference_store
    from tests.eval import fly_stage3_ab

    monkeypatch.setattr(teach_api, "teach_preference", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(preference_store, "record_preference", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(preference_store, "preference_delta", lambda *_args, **_kwargs: -0.08)
    monkeypatch.setattr(
        fly_rank,
        "adjust_recall_score",
        lambda score, text, **_kwargs: ((0.9 if "tarts" in text else score), {}),
    )
    monkeypatch.setattr(giant_fiber, "assess_interrupt", lambda _text: {"interrupt": False, "urgency": 0.0})
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("MEMORY_FLYGF_MODE", "off")

    assert fly_stage3_ab.main() == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    monkeypatch.setattr(preference_store, "preference_delta", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        fly_rank,
        "adjust_recall_score",
        lambda score, _text, **_kwargs: (score, {}),
    )
    monkeypatch.setattr(giant_fiber, "assess_interrupt", lambda _text: {"interrupt": True, "urgency": 1.0})
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "off")
    monkeypatch.setenv("MEMORY_FLYGF_MODE", "live")

    assert fly_stage3_ab.main() == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
