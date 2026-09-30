"""Idle gating applies before dispatch and preserves skipped schedule records."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import ModuleType

import pytest

from system import schedule


@pytest.fixture()
def idle_store(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(schedule, "schedule_path", lambda user_id=None: tmp_path / "schedule.json")
    monkeypatch.setattr("cognition.fly_behavior.gf_global.should_cancel_scheduler", lambda uid: False)
    schedule._invalidate_cache()
    yield
    schedule._invalidate_cache()


def _tracker(monkeypatch, value):
    fake = ModuleType("system.orchestrate")
    fake.get_idle_seconds = lambda: value
    monkeypatch.setitem(sys.modules, "system.orchestrate", fake)


@pytest.mark.parametrize("idle,threshold,expected", [
    (30, 600, False), (1200, 600, True), (599, None, False),
    (600, None, True), (None, 600, False), (0, 0, True),
])
def test_idle_threshold(monkeypatch, idle, threshold, expected):
    _tracker(monkeypatch, idle)
    assert schedule._idle_requirement_met({"idle_seconds": threshold}) is expected


def test_idle_tracker_import_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "system.orchestrate", None)
    assert schedule._idle_requirement_met({}) is False


def test_idle_tracker_failure(monkeypatch):
    _tracker(monkeypatch, None)
    def fail():
        raise RuntimeError("tracker unavailable")
    monkeypatch.setattr(sys.modules["system.orchestrate"], "get_idle_seconds", fail)
    assert schedule._idle_requirement_met({}) is False


@pytest.mark.parametrize("handler", [None, "idle_test"])
@pytest.mark.parametrize("frequency", ["once", "interval", "daily"])
@pytest.mark.parametrize("idle", [30, None])
def test_skipped_jobs_retry_or_advance(idle_store, monkeypatch, handler, frequency, idle):
    now = datetime(2030, 1, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(schedule.bioclock, "local_now", lambda *args: now)
    _tracker(monkeypatch, idle)
    calls = []
    monkeypatch.setitem(schedule._SYSTEM_HANDLERS, "idle_test", lambda mem: calls.append("handler"))
    record = schedule.schedule_job_record(
        "Idle job", "task", "09:00", frequency=frequency, timezone="UTC",
        handler=handler, interval_seconds=900, requires_idle=True, user_id="u1",
    )
    record["next_due"] = (now - timedelta(minutes=1)).isoformat()
    schedule._write_all([record], user_id="u1")
    runner = object.__new__(schedule.ScheduleRunner)
    runner._memorize = None
    runner._on_due = lambda event: calls.append("callback")
    runner._fire_due_user_jobs("u1")
    stored = schedule.list_schedule_records(include_disabled=True, user_id="u1")[0]
    assert calls == []
    assert stored["enabled"] is True
    assert stored["last_ran_at"] is None
    expected = now + (timedelta(seconds=60) if frequency == "once" else
                      timedelta(seconds=900) if frequency == "interval" else
                      timedelta(hours=21))
    assert datetime.fromisoformat(stored["next_due"]) == expected

    # No immediate re-fire; when the retry/next occurrence arrives and the
    # user is idle, both dispatch paths run exactly once.
    _tracker(monkeypatch, 1200)
    runner._fire_due_user_jobs("u1")
    assert calls == []
    monkeypatch.setattr(schedule.bioclock, "local_now", lambda *args: expected)
    runner._fire_due_user_jobs("u1")
    assert calls == ["handler" if handler else "callback"]
    stored = schedule.list_schedule_records(include_disabled=True, user_id="u1")[0]
    assert stored["enabled"] is (frequency != "once")
    assert stored["last_ran_at"] == expected.isoformat()


@pytest.mark.parametrize("handler", [None, "idle_test"])
def test_non_idle_job_dispatches_without_tracker(idle_store, monkeypatch, handler):
    now = datetime(2030, 1, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(schedule.bioclock, "local_now", lambda *args: now)
    monkeypatch.setitem(sys.modules, "system.orchestrate", None)
    calls = []
    monkeypatch.setitem(schedule._SYSTEM_HANDLERS, "idle_test", lambda mem: calls.append("handler"))
    record = schedule.schedule_job_record("Plain", "task", "09:00", frequency="once", handler=handler, user_id="u1")
    record["next_due"] = (now - timedelta(minutes=1)).isoformat()
    schedule._write_all([record], user_id="u1")
    runner = object.__new__(schedule.ScheduleRunner)
    runner._memorize = None
    runner._on_due = lambda event: calls.append("callback")
    runner._fire_due_user_jobs("u1")
    assert calls == ["handler" if handler else "callback"]
