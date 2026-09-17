from __future__ import annotations

from interface.webui.studio.calendar.backend import api


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
