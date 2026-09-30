"""Tests for scheduler record CRUD, idle-gating fields, and the Playground seeder."""
from __future__ import annotations

import pytest

from system import schedule
from system.playground import (
    PLAYGROUND_IDLE_THRESHOLD_SECONDS,
    PLAYGROUND_JOB_TITLE,
    PLAYGROUND_TICK_SECONDS,
    ensure_playground_job,
    worker_skill_text,
)
from system.schedule import DueJob


@pytest.fixture()
def user_store(monkeypatch, tmp_path):
    """Point the schedule JSON store at a throwaway file."""
    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(schedule, "schedule_path", lambda user_id=None: tmp_path / "schedule.json")
    # _read_all/_write_all cache per user; make sure the cache can't leak.
    schedule._invalidate_cache("u1")
    yield "u1"
    schedule._invalidate_cache("u1")


def test_create_stores_idle_fields(user_store):
    uid = user_store
    rec = schedule.schedule_job_record(
        "Idle job", "do work", "09:00",
        frequency="interval", interval_seconds=900, action="agentic",
        requires_idle=True, idle_seconds=600, user_id=uid,
    )
    assert rec["requires_idle"] is True
    assert rec["idle_seconds"] == 600
    stored = schedule.list_schedule_records(include_disabled=True, user_id=uid)
    assert stored[0]["requires_idle"] is True
    assert stored[0]["idle_seconds"] == 600


def test_create_defaults_to_not_idle_gated(user_store):
    rec = schedule.schedule_job_record("Plain", "task", "09:00", user_id=user_store)
    assert rec["requires_idle"] is False
    assert rec["idle_seconds"] is None


def test_create_rejects_bad_idle_seconds(user_store):
    with pytest.raises(ValueError):
        schedule.schedule_job_record("x", "t", "09:00", idle_seconds="abc", user_id=user_store)
    with pytest.raises(ValueError):
        schedule.schedule_job_record("x", "t", "09:00", idle_seconds=-5, user_id=user_store)


def test_update_timing_recalculates_next_due(user_store):
    uid = user_store
    rec = schedule.schedule_job_record("T", "t", "09:00", frequency="daily", user_id=uid)
    old_due = rec["next_due"]
    updated = schedule.update_schedule_record(
        rec["id"],
        {"time_of_day": "18:30", "frequency": "weekly", "days_of_week": ["mon", "wed"],
         "timezone": "America/Vancouver"},
        user_id=uid,
    )
    assert updated is not None
    assert updated["time_of_day"] == "18:30"
    assert updated["frequency"] == "weekly"
    assert updated["days_of_week"] == [0, 2]  # weekdays preserved as ints
    assert updated["timezone"] == "America/Vancouver"  # timezone preserved
    assert updated["next_due"] != old_due


def test_update_non_timing_fields_keep_next_due(user_store):
    uid = user_store
    rec = schedule.schedule_job_record("T", "t", "09:00", user_id=uid)
    updated = schedule.update_schedule_record(
        rec["id"], {"title": "Renamed", "enabled": False, "requires_idle": True, "idle_seconds": 300},
        user_id=uid,
    )
    assert updated["title"] == "Renamed"
    assert updated["enabled"] is False
    assert updated["requires_idle"] is True
    assert updated["idle_seconds"] == 300
    assert updated["next_due"] == rec["next_due"]


def test_update_interval_validation(user_store):
    uid = user_store
    rec = schedule.schedule_job_record("T", "t", "00:00", frequency="interval",
                                       interval_seconds=900, user_id=uid)
    with pytest.raises(ValueError):
        schedule.update_schedule_record(rec["id"], {"frequency": "bogus"}, user_id=uid)
    with pytest.raises(ValueError):
        schedule.update_schedule_record(rec["id"], {"action": "explode"}, user_id=uid)
    with pytest.raises(ValueError):
        schedule.update_schedule_record(rec["id"], {"interval_seconds": 10}, user_id=uid)
    assert schedule.update_schedule_record("no-such-id", {"title": "x"}, user_id=uid) is None


def test_delete_schedule_record(user_store):
    uid = user_store
    keep = schedule.schedule_job_record("Keep", "t", "09:00", user_id=uid)
    drop = schedule.schedule_job_record("Drop", "t", "09:00", user_id=uid)
    assert schedule.delete_schedule_record(drop["id"], user_id=uid) is True
    assert schedule.delete_schedule_record(drop["id"], user_id=uid) is False
    remaining = schedule.list_schedule_records(include_disabled=True, user_id=uid)
    assert [r["id"] for r in remaining] == [keep["id"]]


def test_due_job_carries_idle_gate():
    job = DueJob(id="1", title="t", task="t", requires_idle=True, idle_seconds=600)
    assert job.requires_idle is True
    assert job.idle_seconds == 600
    plain = DueJob(id="2", title="t", task="t")
    assert plain.requires_idle is False
    assert plain.idle_seconds is None


def test_ensure_playground_job_is_idempotent(user_store):
    uid = user_store
    first = ensure_playground_job(user_id=uid)
    second = ensure_playground_job(user_id=uid)
    assert first["id"] == second["id"]
    assert first["title"] == PLAYGROUND_JOB_TITLE
    assert first["frequency"] == "interval"
    assert first["interval_seconds"] == PLAYGROUND_TICK_SECONDS
    assert first["action"] == "agentic"
    assert first["requires_idle"] is True
    assert first["idle_seconds"] == PLAYGROUND_IDLE_THRESHOLD_SECONDS
    assert "sandbox/run.py" in (first["skill"] or "")
    records = schedule.list_schedule_records(include_disabled=True, user_id=uid)
    assert sum(1 for r in records if r["title"] == PLAYGROUND_JOB_TITLE) == 1


def test_playground_skill_states_safety_contract():
    skill = worker_skill_text()
    assert "PLAYGROUND_SMTP_PASS" in skill
    assert "loop/disabled" in skill
    assert "sandbox/run.py" in skill
    assert "Aiko-chan" in skill  # must name what it must NOT touch
    assert "never" in skill.lower()
    assert "LOG.md" in skill  # full coding log is a hard requirement
    assert "web search" in skill.lower()  # research when stuck


def test_ensure_playground_job_preserves_paused_record(user_store):
    first = ensure_playground_job(user_id=user_store)
    schedule.cancel_schedule_record(first["id"], user_id=user_store)
    second = ensure_playground_job(user_id=user_store)
    assert second["id"] == first["id"]
    assert second["enabled"] is False
    records = schedule.list_schedule_records(include_disabled=True, user_id=user_store)
    assert len(records) == 1


@pytest.mark.parametrize("initial_action,initial_call,updates", [
    ("agentic", None, {"action": "tool"}),
    ("tool", {"name": "test", "arguments": {}}, {"tool_call": None}),
    ("tool", {"name": "test", "arguments": {}}, {"action": "tool", "tool_call": None}),
])
def test_update_rejects_tool_without_configuration(user_store, initial_action, initial_call, updates):
    record = schedule.schedule_job_record(
        "Tool", "task", "09:00", action=initial_action, tool_call=initial_call, user_id=user_store,
    )
    with pytest.raises(ValueError, match="requires tool_call"):
        schedule.update_schedule_record(record["id"], updates, user_id=user_store)
    assert schedule.list_schedule_records(user_id=user_store) == [record]


def test_update_accepts_valid_tool_transitions(user_store):
    record = schedule.schedule_job_record("Tool", "task", "09:00", user_id=user_store)
    call = {"name": "test", "arguments": {"value": 1}}
    updated = schedule.update_schedule_record(record["id"], {"action": "tool", "tool_call": call}, user_id=user_store)
    assert updated["tool_call"] == call
    updated = schedule.update_schedule_record(record["id"], {"title": "Renamed"}, user_id=user_store)
    assert updated["tool_call"] == call
    updated = schedule.update_schedule_record(record["id"], {"action": "agentic", "tool_call": None}, user_id=user_store)
    assert updated["action"] == "agentic"
    assert updated["tool_call"] is None
