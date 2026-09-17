"""Local-first Calendar Studio API.

The studio keeps chief-of-staff planning data in the authenticated user's
state directory.  Reminders and workflows optionally create real Aiko
schedule records, so the view is a control surface for the running scheduler
rather than a disconnected mock calendar.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from interface.webui.studio.session_binding import bind_login_session
from system import bioclock
from system.schedule import cancel_schedule_record, list_schedule_records, notify_scheduler_new_job, schedule_job_record
from system.userspace import user_state_path

app = FastAPI(title="Aiko Calendar Studio")
bind_login_session(app)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="calendar-frontend")

STATUSES = {"backlog", "todo", "doing", "waiting", "done", "cancelled"}


class ItemInput(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    kind: Literal["appointment", "task", "draft", "workflow"] = "task"
    status: str = "todo"
    start_at: str | None = None
    due_at: str | None = None
    notes: str = Field(default="", max_length=4000)
    project: str = Field(default="", max_length=80)
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    schedule: dict[str, Any] | None = None


def _store_path() -> Path:
    return user_state_path("tasks/calendar_studio.json").resolve()


def _load() -> list[dict[str, Any]]:
    path = _store_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "Calendar data could not be read.") from exc
    return payload.get("items", []) if isinstance(payload, dict) and isinstance(payload.get("items", []), list) else []


def _save(items: list[dict[str, Any]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="calendar-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "items": items}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise HTTPException(500, "Calendar data could not be saved.") from exc


def _valid_iso(value: str | None, field: str) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError as exc:
        raise HTTPException(422, f"{field} must be an ISO date or date-time.") from exc


def _schedule(item: ItemInput) -> str | None:
    if not item.schedule:
        return None
    spec = item.schedule
    try:
        record = schedule_job_record(
            item.title, item.notes or item.title, str(spec.get("time_of_day", "09:00")),
            str(spec.get("frequency", "once")), spec.get("timezone"),
            spec.get("days_of_week"), str(spec.get("action", "announce")),
            tool_call=spec.get("tool_call"),
        )
        notify_scheduler_new_job()
        return str(record["id"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"Invalid scheduler settings: {exc}") from exc


def _serialize(item: ItemInput, item_id: str | None = None) -> dict[str, Any]:
    if item.status not in STATUSES:
        raise HTTPException(422, "Unknown task status.")
    created = bioclock.local_now().isoformat()
    result = {
        "id": item_id or uuid.uuid4().hex[:12], "title": item.title.strip(), "kind": item.kind,
        "status": item.status, "start_at": _valid_iso(item.start_at, "start_at"),
        "due_at": _valid_iso(item.due_at, "due_at"), "notes": item.notes.strip(),
        "project": item.project.strip(), "priority": item.priority, "created_at": created,
    }
    if result["start_at"] and result["due_at"] and result["due_at"] < result["start_at"]:
        raise HTTPException(422, "due_at must be after start_at.")
    return result


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    items = _load()
    schedules = list_schedule_records(include_disabled=True)
    now = bioclock.local_now()
    today = now.date()
    open_items = [i for i in items if i.get("status") not in {"done", "cancelled"}]
    due_today = [i for i in open_items if (i.get("due_at") or "")[:10] == today.isoformat()]
    overdue = [i for i in open_items if i.get("due_at") and i["due_at"][:10] < today.isoformat()]
    return {
        "today": today.isoformat(), "items": items, "schedules": schedules,
        "brief": {"open_count": len(open_items), "due_today": due_today, "overdue": overdue,
                  "next_up": sorted([i for i in open_items if i.get("start_at") or i.get("due_at")], key=lambda i: i.get("start_at") or i.get("due_at"))[:5]},
    }


@app.post("/api/items", status_code=201)
def create_item(item: ItemInput) -> dict[str, Any]:
    stored = _serialize(item)
    stored["schedule_id"] = _schedule(item)
    items = _load()
    items.append(stored)
    _save(items)
    return stored


@app.put("/api/items/{item_id}")
def update_item(item_id: str, item: ItemInput) -> dict[str, Any]:
    items = _load()
    for index, existing in enumerate(items):
        if existing.get("id") == item_id:
            stored = _serialize(item, item_id)
            stored["created_at"] = existing.get("created_at", stored["created_at"])
            stored["schedule_id"] = existing.get("schedule_id")
            items[index] = stored
            _save(items)
            return stored
    raise HTTPException(404, "Calendar item not found.")


@app.delete("/api/items/{item_id}")
def delete_item(item_id: str) -> dict[str, bool]:
    items = _load()
    kept = [item for item in items if item.get("id") != item_id]
    if len(kept) == len(items):
        raise HTTPException(404, "Calendar item not found.")
    removed = next(item for item in items if item.get("id") == item_id)
    if removed.get("schedule_id"):
        cancel_schedule_record(str(removed["schedule_id"]))
        notify_scheduler_new_job()
    _save(kept)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
