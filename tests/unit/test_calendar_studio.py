from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi import HTTPException

from interface.webui.studio.calendar.backend import api


def _scheduler_record(schedule_id="schedule-old", *, time_of_day="09:00", frequency="daily"):
    return {
        "id": schedule_id,
        "title": "Plan",
        "task": "Plan",
        "time_of_day": time_of_day,
        "frequency": frequency,
        "timezone": "UTC",
        "days_of_week": [],
        "action": "announce",
        "tool_call": None,
        "enabled": True,
    }


def test_calendar_items_create_update_and_daily_brief(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_store_path", lambda: tmp_path / "calendar_studio.json")
    item = api.ItemInput(
        title="Send the project update",
        kind="draft",
        status="todo",
        due_at="2030-01-02T09:00",
        project="Launch",
    )

    created = api.create_item(item)
    assert created["kind"] == "draft"
    assert created["due_at"] == "2030-01-02T09:00:00"
    assert created["schedule_id"] is None

    updated = api.update_item(created["id"], item.model_copy(update={"status": "doing"}) if hasattr(item, "model_copy") else item.copy(update={"status": "doing"}))
    assert updated["id"] == created["id"]
    assert updated["status"] == "doing"

    monkeypatch.setattr(api, "list_schedule_records", lambda include_disabled=True: [])
    overview = api.overview()
    assert overview["brief"]["open_count"] == 1
    assert overview["items"][0]["project"] == "Launch"

    assert api.delete_item(created["id"]) == {"ok": True}
    assert api._load() == []


def test_timestamp_ranges_compare_instants(monkeypatch):
    monkeypatch.setenv("TIMEZONE", "UTC")
    valid = api.ItemInput(
        title="Valid across offsets",
        start_at="2030-01-02T10:00:00+02:00",
        due_at="2030-01-02T09:00:00+00:00",
    )
    assert api._serialize(valid)["due_at"] == "2030-01-02T09:00:00+00:00"

    invalid = api.ItemInput(
        title="Invalid across offsets",
        start_at="2030-01-02T10:00:00+00:00",
        due_at="2030-01-02T11:00:00+02:00",
    )
    with pytest.raises(HTTPException, match="due_at must be after start_at"):
        api._serialize(invalid)


def test_overview_uses_local_due_dates_and_instant_order(monkeypatch):
    monkeypatch.setenv("TIMEZONE", "America/Vancouver")
    monkeypatch.setattr(api.bioclock, "local_now", lambda: datetime(2030, 1, 2, 1, tzinfo=ZoneInfo("America/Vancouver")))
    items = [
        {"id": "overdue", "title": "Previous local day", "status": "todo", "due_at": "2030-01-02T06:30:00Z"},
        {"id": "today", "title": "Current local day", "status": "todo", "due_at": "2030-01-03T06:30:00Z"},
        {"id": "first", "title": "First instant", "status": "todo", "start_at": "2030-01-04T09:00:00+02:00"},
        {"id": "second", "title": "Second instant", "status": "todo", "start_at": "2030-01-04T08:00:00+00:00"},
    ]
    monkeypatch.setattr(api, "_load", lambda: items)
    monkeypatch.setattr(api, "list_schedule_records", lambda include_disabled=True: [])

    result = api.overview()

    assert [item["id"] for item in result["brief"]["due_today"]] == ["today"]
    assert [item["id"] for item in result["brief"]["overdue"]] == ["overdue"]
    ordered = [item["id"] for item in result["brief"]["next_up"]]
    assert ordered.index("first") < ordered.index("second")


def test_create_rolls_back_schedule_when_calendar_load_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_store_path", lambda: tmp_path / "calendar_studio.json")
    monkeypatch.setattr(api, "_schedule", lambda _item: "schedule-new")
    monkeypatch.setattr(api, "_load", lambda: (_ for _ in ()).throw(HTTPException(500, "load failed")))
    cancelled = []
    monkeypatch.setattr(api, "cancel_schedule_record", lambda schedule_id: cancelled.append(schedule_id) or True)
    monkeypatch.setattr(api, "notify_scheduler_new_job", lambda: None)

    with pytest.raises(HTTPException, match="load failed"):
        api.create_item(api.ItemInput(title="Plan", schedule={"time_of_day": "09:00"}))

    assert cancelled == ["schedule-new"]


def test_delete_fails_closed_for_missing_or_uncancellable_schedule(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_store_path", lambda: tmp_path / "calendar_studio.json")
    item = {"id": "item-1", "title": "Plan", "schedule_id": "missing"}
    monkeypatch.setattr(api, "_load", lambda: [item])
    monkeypatch.setattr(api, "list_schedule_records", lambda include_disabled=True: [])
    saved = []
    monkeypatch.setattr(api, "_save", lambda items: saved.append(items))

    with pytest.raises(HTTPException, match="could not be read"):
        api.delete_item("item-1")
    assert saved == []

    monkeypatch.setattr(api, "list_schedule_records", lambda include_disabled=True: [_scheduler_record("missing")])
    monkeypatch.setattr(api, "cancel_schedule_record", lambda _schedule_id: False)
    with pytest.raises(HTTPException, match="could not be cancelled"):
        api.delete_item("item-1")
    assert saved == []


def test_delete_restores_schedule_when_calendar_save_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_store_path", lambda: tmp_path / "calendar_studio.json")
    item = {"id": "item-1", "title": "Plan", "schedule_id": "schedule-old"}
    record = _scheduler_record()
    monkeypatch.setattr(api, "_load", lambda: [item])
    monkeypatch.setattr(api, "list_schedule_records", lambda include_disabled=True: [record])
    monkeypatch.setattr(api, "cancel_schedule_record", lambda _schedule_id: True)
    monkeypatch.setattr(api, "_save", lambda _items: (_ for _ in ()).throw(HTTPException(500, "save failed")))
    restored = []
    monkeypatch.setattr(api, "restore_schedule_record", lambda snapshot: restored.append(snapshot) or True)
    monkeypatch.setattr(api, "notify_scheduler_new_job", lambda: None)

    with pytest.raises(HTTPException, match="save failed"):
        api.delete_item("item-1")

    assert restored == [record]


def test_update_preserves_unchanged_schedule_and_replaces_changed_schedule(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMEZONE", "UTC")
    monkeypatch.setattr(api, "_store_path", lambda: tmp_path / "calendar_studio.json")
    existing = {"id": "item-1", "title": "Plan", "status": "todo", "schedule_id": "schedule-old", "created_at": "earlier"}
    record = _scheduler_record()
    monkeypatch.setattr(api, "_load", lambda: [existing])
    monkeypatch.setattr(api, "list_schedule_records", lambda include_disabled=True: [record])
    saved = []
    monkeypatch.setattr(api, "_save", lambda items: saved.append(items))
    scheduled = []
    monkeypatch.setattr(api, "_schedule", lambda item: scheduled.append(item.schedule) or "schedule-new")
    cancelled = []
    monkeypatch.setattr(api, "cancel_schedule_record", lambda schedule_id: cancelled.append(schedule_id) or True)
    monkeypatch.setattr(api, "notify_scheduler_new_job", lambda: None)

    unchanged = api.update_item("item-1", api.ItemInput(title="Plan", schedule={"time_of_day": "09:00", "frequency": "daily"}))
    assert unchanged["schedule_id"] == "schedule-old"
    assert scheduled == []
    assert cancelled == []

    changed = api.update_item("item-1", api.ItemInput(title="Plan", schedule={"time_of_day": "10:00", "frequency": "daily"}))
    assert changed["schedule_id"] == "schedule-new"
    assert scheduled == [{"time_of_day": "10:00", "frequency": "daily"}]
    assert cancelled == ["schedule-old"]


def test_update_clears_schedule_and_rolls_back_replacement_on_save_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_store_path", lambda: tmp_path / "calendar_studio.json")
    existing = {"id": "item-1", "title": "Plan", "status": "todo", "schedule_id": "schedule-old"}
    record = _scheduler_record()
    monkeypatch.setattr(api, "_load", lambda: [existing])
    monkeypatch.setattr(api, "list_schedule_records", lambda include_disabled=True: [record])
    cancelled = []
    monkeypatch.setattr(api, "cancel_schedule_record", lambda schedule_id: cancelled.append(schedule_id) or True)
    monkeypatch.setattr(api, "notify_scheduler_new_job", lambda: None)
    saved = []
    monkeypatch.setattr(api, "_save", lambda items: saved.append(items))

    cleared = api.update_item("item-1", api.ItemInput(title="Plan"))
    assert cleared["schedule_id"] is None
    assert cancelled == ["schedule-old"]

    monkeypatch.setattr(api, "_schedule", lambda _item: "schedule-new")
    monkeypatch.setattr(api, "_save", lambda _items: (_ for _ in ()).throw(HTTPException(500, "save failed")))
    restored = []
    monkeypatch.setattr(api, "restore_schedule_record", lambda snapshot: restored.append(snapshot) or True)
    with pytest.raises(HTTPException, match="save failed"):
        api.update_item("item-1", api.ItemInput(title="Plan", schedule={"time_of_day": "10:00"}))
    assert cancelled[-2:] == ["schedule-old", "schedule-new"]
    assert restored == [record]


def test_calendar_mutations_are_serialized(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_store_path", lambda: tmp_path / "calendar_studio.json")
    original_load = api._load
    active_loads = 0
    max_active_loads = 0
    counter_lock = threading.Lock()

    def observed_load():
        nonlocal active_loads, max_active_loads
        with counter_lock:
            active_loads += 1
            max_active_loads = max(max_active_loads, active_loads)
        time.sleep(0.03)
        try:
            return original_load()
        finally:
            with counter_lock:
                active_loads -= 1

    monkeypatch.setattr(api, "_load", observed_load)
    start = threading.Barrier(2)

    def create(title):
        start.wait()
        return api.create_item(api.ItemInput(title=title))

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(create, ["One", "Two"]))

    assert max_active_loads == 1
    assert {item["title"] for item in original_load()} == {"One", "Two"}
