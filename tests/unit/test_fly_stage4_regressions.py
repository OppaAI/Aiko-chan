"""Regression tests for Stage 4 review fixes."""

from __future__ import annotations

import importlib
from collections import deque
from types import SimpleNamespace

import pytest


class _MB:
    def __init__(self, delta: float = 0.25):
        self.delta = delta
        self.reinforce_calls = 0

    def encode(self, _features):
        return "kc"

    def reinforce(self, _kc, _reward):
        self.reinforce_calls += 1
        return self.delta

    def valence_bias(self, _features):
        return 0.2


def test_maintenance_consolidates_once_per_user_transition(monkeypatch):
    from cognition.fly_behavior import sleep_sched

    consolidate_mb = importlib.import_module("cognition.flymemory.consolidate_mb")

    active = {"stage4-maint-a": True, "stage4-maint-b": True}
    calls = []
    monkeypatch.setattr(sleep_sched, "should_prefer_maintenance", lambda user_id: active[user_id])
    monkeypatch.setattr(
        consolidate_mb,
        "consolidate",
        lambda user_id: calls.append(user_id) or {"ran": True, "reason": "ok"},
    )
    sleep_sched._maintenance_users.difference_update(active)

    assert sleep_sched.maybe_consolidate_mb("stage4-maint-a")["ran"] is True
    assert sleep_sched.maybe_consolidate_mb("stage4-maint-a") == {
        "ran": False,
        "reason": "already_consolidated",
    }
    assert sleep_sched.maybe_consolidate_mb("stage4-maint-b")["ran"] is True

    active["stage4-maint-a"] = False
    assert sleep_sched.maybe_consolidate_mb("stage4-maint-a")["reason"] == "not_maintenance"
    active["stage4-maint-a"] = True
    assert sleep_sched.maybe_consolidate_mb("stage4-maint-a")["ran"] is True
    assert calls == ["stage4-maint-a", "stage4-maint-b", "stage4-maint-a"]

    sleep_sched._maintenance_users.difference_update(active)


def test_stage4_hook_records_only_when_teaching_did_not(monkeypatch):
    from cognition.fly_behavior import sleep_sched, stage4_hooks
    from cognition.flymemory import eligibility

    calls = []
    monkeypatch.setattr(eligibility, "record_step", lambda user_id, text: calls.append((user_id, text)) or True)
    monkeypatch.setattr(sleep_sched, "maybe_consolidate_mb", lambda _user_id: {"ran": False})

    already_recorded = {"online_teach": {"eligibility_recorded": True}}
    stage4_hooks.after_online_teach("user-1", "hello", already_recorded)
    assert calls == []

    needs_record = {"online_teach": {"eligibility_recorded": False}}
    stage4_hooks.after_online_teach("user-1", "hello", needs_record)
    assert calls == [("user-1", "hello")]
    assert needs_record["online_teach"]["eligibility_recorded"] is True


@pytest.mark.parametrize(
    "pulse_result",
    [
        {"reason": "mode_off", "applied": True, "delta": 0.25},
        {"reason": "ok", "applied": False, "delta": 0.25},
        {"reason": "error", "applied": False, "delta": 0.25},
    ],
)
def test_assign_credit_requires_successful_applied_pulse(monkeypatch, pulse_result):
    from cognition import fly_registry
    from cognition.flymemory import dopamine, eligibility

    uid = "stage4-credit-validation"
    mb = _MB()
    eligibility.clear(uid)
    eligibility._TRACES[uid] = deque([("kc", 0.0)])
    monkeypatch.setattr(eligibility, "_enabled", lambda: True)
    monkeypatch.setattr(eligibility, "_mb_mode", lambda: "live")
    monkeypatch.setattr(fly_registry, "get_flymb", lambda _user_id: mb)
    monkeypatch.setattr(fly_registry, "get_fly_store", lambda _user_id: None)
    monkeypatch.setattr(dopamine, "pulse", lambda *_args, **_kwargs: pulse_result)

    result = eligibility.assign_credit(uid, 0.5)

    assert result["taught"] is False
    assert result["steps"] == 0
    assert result["delta"] == 0.0
    assert mb.reinforce_calls == 0
    eligibility.clear(uid)


@pytest.mark.parametrize("applied, reason", [
    (True, "ok"), (False, "shadow"), (False, "dedup"),
    (False, "mb_unavailable"),
])
def test_assign_credit_flushes_only_applied_events(monkeypatch, applied, reason):
    from cognition import fly_registry
    from cognition.flymemory import credit, eligibility

    flushed = []
    mb = _MB()
    monkeypatch.setattr(eligibility, "_enabled", lambda: True)
    monkeypatch.setattr(eligibility, "_mb_mode", lambda: "live")
    monkeypatch.setattr(credit, "credit_event", lambda *_args, **_kwargs: {
        "applied": applied, "reason": reason, "n_applied_traces": int(applied),
        "delta": 0.25 if applied else 0.0,
    })
    monkeypatch.setattr(fly_registry, "get_flymb", lambda _uid: mb)
    store = SimpleNamespace(flush_mb=lambda value: flushed.append(value))
    monkeypatch.setattr(fly_registry, "get_fly_store", lambda _uid: store)

    result = eligibility.assign_credit("stage4-flush", 0.5)

    assert result["flushed"] is applied
    assert flushed == ([mb] if applied else [])


def test_assign_credit_fails_soft_when_engine_raises(monkeypatch):
    # Phase 10A: assign_credit is one rate-based engine event. If the engine
    # itself raises, the call must not propagate — it reports instead of
    # writing (the old per-step pulse→reinforce fallback no longer exists;
    # per-trace failures are already contained inside the engine).
    from cognition import fly_registry
    from cognition.flymemory import credit, eligibility

    uid = "stage4-credit-engine-fail"
    mb = _MB()
    eligibility.clear(uid)
    monkeypatch.setattr(eligibility, "_enabled", lambda: True)
    monkeypatch.setattr(eligibility, "_mb_mode", lambda: "live")
    monkeypatch.setattr(fly_registry, "get_flymb", lambda _user_id: mb)
    monkeypatch.setattr(fly_registry, "get_fly_store", lambda _user_id: None)
    monkeypatch.setattr(
        credit, "credit_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no credit")),
    )

    result = eligibility.assign_credit(uid, 0.5)

    assert result["taught"] is False
    assert result["steps"] == 0
    assert result["delta"] == 0.0
    assert "no credit" in result["reason"]
    assert mb.reinforce_calls == 0
    eligibility.clear(uid)


@pytest.mark.parametrize(
    "pulse_result",
    [
        {"reason": "ok", "applied": False, "delta": 0.25},
        {"reason": "error", "applied": True, "delta": 0.25},
    ],
)
def test_teaching_rejects_unsuccessful_pulse_but_records_direct_call(monkeypatch, pulse_result):
    from cognition import fly_registry, neural_state
    from cognition.flymemory import dopamine, eligibility, online_teach, teach_api

    class State:
        def __init__(self):
            self.published = []

        def publish_mb(self, bias, *, source):
            self.published.append((bias, source))

    mb = _MB()
    state = State()
    monkeypatch.setattr(online_teach, "_mode", lambda: "live")
    monkeypatch.setattr(teach_api, "extract_topic", lambda _text: None)
    monkeypatch.setattr(fly_registry, "get_flymb", lambda _user_id: mb)
    monkeypatch.setattr(neural_state, "get_neural_state", lambda _user_id: state)
    monkeypatch.setattr(
        dopamine,
        "pulse",
        lambda *_args, **_kwargs: pulse_result,
    )
    monkeypatch.setattr(eligibility, "record_step", lambda _user_id, _text: True)

    result = online_teach.teach_from_user_text("thanks, perfect!", user_id="user-1")

    assert result["taught"] is False
    assert result["eligibility_recorded"] is True
    assert state.published == []
    assert mb.reinforce_calls == 0


@pytest.mark.parametrize(
    "pulse_result",
    [
        {"reason": "ok", "applied": False, "delta": 0.25},
        {"reason": "error", "applied": True, "delta": 0.25},
    ],
)
def test_interrupt_teaching_rejects_unsuccessful_pulse(monkeypatch, pulse_result):
    from cognition import fly_registry
    from cognition.flymemory import dopamine, eligibility, online_teach

    mb = _MB()
    monkeypatch.setattr(online_teach, "_mode", lambda: "live")
    monkeypatch.setattr(fly_registry, "get_flymb", lambda _user_id: mb)
    monkeypatch.setattr(dopamine, "pulse", lambda *_args, **_kwargs: pulse_result)
    monkeypatch.setattr(eligibility, "record_step", lambda *_args, **_kwargs: True)

    result = online_teach.teach_interrupt_honored("user-1", "stop")

    assert result["taught"] is False
    assert result["eligibility_recorded"] is True
    assert mb.reinforce_calls == 0


def test_interrupt_teaching_reuses_existing_eligibility_entry(monkeypatch):
    from cognition import fly_registry
    from cognition.flymemory import dopamine, eligibility, online_teach

    mb = _MB()
    record_calls = []
    monkeypatch.setattr(online_teach, "_mode", lambda: "live")
    monkeypatch.setattr(fly_registry, "get_flymb", lambda _user_id: mb)
    monkeypatch.setattr(
        dopamine,
        "pulse",
        lambda *_args, **_kwargs: {"reason": "ok", "applied": True, "delta": 0.25},
    )
    monkeypatch.setattr(eligibility, "assign_credit", lambda *_args, **_kwargs: {"taught": True})
    monkeypatch.setattr(eligibility, "record_step", lambda *_args, **_kwargs: record_calls.append(True) or True)

    result = online_teach.teach_interrupt_honored(
        "user-1",
        "stop",
        eligibility_recorded=True,
    )

    assert result["taught"] is True
    assert result["eligibility_recorded"] is True
    assert record_calls == []


@pytest.mark.parametrize(
    ("dopamine_result", "consolidate_result", "expected"),
    [
        ({"reason": "ok", "applied": True}, {"ran": True}, 0),
        ({"reason": "mode_off", "applied": False}, {"ran": True}, 2),
        ({"reason": "error", "applied": True}, {"ran": True}, 2),
        ({"reason": "ok"}, {"ran": True}, 2),
        ({"reason": "ok", "applied": True}, {"ran": False}, 2),
    ],
)
def test_stage4_eval_requires_pulse_and_consolidation_to_run(
    monkeypatch,
    capsys,
    dopamine_result,
    consolidate_result,
    expected,
):
    from cognition.flymemory import dopamine, eligibility
    from tests.eval import fly_stage4_ab

    consolidate_mb = importlib.import_module("cognition.flymemory.consolidate_mb")

    monkeypatch.setattr(eligibility, "clear", lambda _user_id: None)
    monkeypatch.setattr(eligibility, "record_step", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(eligibility, "assign_credit", lambda *_args, **_kwargs: {"taught": True})
    monkeypatch.setattr(eligibility, "stats", lambda _user_id: {})
    monkeypatch.setattr(dopamine, "pulse", lambda *_args, **_kwargs: dopamine_result)
    monkeypatch.setattr(consolidate_mb, "consolidate", lambda *_args, **_kwargs: consolidate_result)

    assert fly_stage4_ab.main() == expected
    capsys.readouterr()
