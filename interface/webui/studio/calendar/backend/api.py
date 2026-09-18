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
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import fcntl

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from interface.webui.studio.session_binding import bind_login_session
from system import bioclock
from system.schedule import cancel_schedule_record, list_schedule_records, notify_scheduler_new_job, restore_schedule_record, schedule_job_record
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


@contextmanager
def _calendar_lock():
    """Serialize calendar mutations for the active user's store."""
    lock_path = _store_path().with_suffix(".lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        raise HTTPException(500, "Calendar data could not be locked.") from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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


def _parse_iso(value: str, field: str) -> datetime:
    """Parse and validate an ISO date or date-time value."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, f"{field} must be an ISO date or date-time.") from exc


def _instant(value: str, field: str) -> datetime:
    """Return a comparable instant, treating naive values as local time."""
    parsed = _parse_iso(value, field)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=bioclock.get_timezone())
    return parsed


def _valid_iso(value: str | None, field: str) -> str | None:
    if not value:
        return None
    return _parse_iso(value, field).isoformat()


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


def _schedule_record(schedule_id: str) -> dict[str, Any]:
    """Read a linked scheduler record, failing closed when it is unavailable."""
    try:
        schedules = list_schedule_records(include_disabled=True)
        record = next((record for record in schedules if isinstance(record, dict) and str(record.get("id")) == schedule_id), None)
    except Exception as exc:
        raise HTTPException(500, "Linked scheduler data could not be read.") from exc
    if record is None:
        raise HTTPException(500, "Linked scheduler record could not be read.")
    return deepcopy(record)


def _cancel_schedule(schedule_id: str) -> None:
    try:
        cancelled = cancel_schedule_record(schedule_id)
    except Exception as exc:
        raise HTTPException(500, "Linked scheduler record could not be cancelled.") from exc
    if not cancelled:
        raise HTTPException(500, "Linked scheduler record could not be cancelled.")
    notify_scheduler_new_job()


def _restore_schedule(record: dict[str, Any]) -> None:
    try:
        restored = restore_schedule_record(record)
    except Exception as exc:
        raise HTTPException(500, "Linked scheduler record could not be restored.") from exc
    if not restored:
        raise HTTPException(500, "Linked scheduler record could not be restored.")
    notify_scheduler_new_job()


def _normalized_days(value: Any) -> tuple[int, ...] | None:
    aliases = {
        "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3,
        "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }
    if value is None:
        return ()
    parts = value.replace(",", " ").split() if isinstance(value, str) else value
    try:
        days = {int(part) if str(part).isdigit() else aliases[str(part).strip().lower()] for part in parts}
    except (KeyError, TypeError, ValueError):
        return None
    return tuple(sorted(days)) if all(0 <= day <= 6 for day in days) else None


def _normalized_tool_call(value: Any) -> Any:
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return value
    return {"name": value["name"].strip(), "arguments": value.get("arguments", value.get("args", {}))}


def _schedule_unchanged(item: ItemInput, record: dict[str, Any]) -> bool:
    spec = item.schedule
    if not spec or not record.get("enabled", True):
        return False
    requested_days = _normalized_days(spec.get("days_of_week"))
    return requested_days is not None and (
        str(spec.get("time_of_day", "09:00")),
        str(spec.get("frequency", "once")).lower().strip(),
        bioclock.timezone_name(spec.get("timezone")),
        requested_days,
        str(spec.get("action", "announce")).lower().strip(),
        _normalized_tool_call(spec.get("tool_call")),
    ) == (
        str(record.get("time_of_day", "09:00")),
        str(record.get("frequency", "once")).lower().strip(),
        bioclock.timezone_name(record.get("timezone")),
        _normalized_days(record.get("days_of_week")),
        str(record.get("action", "announce")).lower().strip(),
        _normalized_tool_call(record.get("tool_call")),
    )


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
    if result["start_at"] and result["due_at"] and _instant(result["due_at"], "due_at") < _instant(result["start_at"], "start_at"):
        raise HTTPException(422, "due_at must be after start_at.")
    return result


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    items = _load()
    schedules = list_schedule_records(include_disabled=True)
    now = bioclock.local_now()
    today = now.date()
    open_items = [i for i in items if i.get("status") not in {"done", "cancelled"}]
    due_items = [(item, _instant(item["due_at"], "due_at").astimezone(bioclock.get_timezone())) for item in open_items if item.get("due_at")]
    due_today = [item for item, due in due_items if due.date() == today]
    overdue = [item for item, due in due_items if due.date() < today]
    return {
        "today": today.isoformat(), "items": items, "schedules": schedules,
        "brief": {"open_count": len(open_items), "due_today": due_today, "overdue": overdue,
                  "next_up": sorted(
                      [i for i in open_items if i.get("start_at") or i.get("due_at")],
                      key=lambda i: _instant(i.get("start_at") or i["due_at"], "start_at" if i.get("start_at") else "due_at"),
                  )[:5]},
    }


@app.post("/api/items", status_code=201)
def create_item(item: ItemInput) -> dict[str, Any]:
    stored = _serialize(item)
    with _calendar_lock():
        stored["schedule_id"] = _schedule(item)
        try:
            items = _load()
            items.append(stored)
            _save(items)
        except Exception:
            if stored["schedule_id"]:
                _cancel_schedule(stored["schedule_id"])
            raise
    return stored


@app.put("/api/items/{item_id}")
def update_item(item_id: str, item: ItemInput) -> dict[str, Any]:
    with _calendar_lock():
        items = _load()
        for index, existing in enumerate(items):
            if existing.get("id") != item_id:
                continue
            stored = _serialize(item, item_id)
            stored["created_at"] = existing.get("created_at", stored["created_at"])
            old_schedule_id = str(existing["schedule_id"]) if existing.get("schedule_id") else None
            old_schedule = _schedule_record(old_schedule_id) if old_schedule_id else None
            if old_schedule and _schedule_unchanged(item, old_schedule):
                stored["schedule_id"] = old_schedule_id
                items[index] = stored
                _save(items)
                return stored

            new_schedule_id = _schedule(item)
            if old_schedule_id:
                try:
                    _cancel_schedule(old_schedule_id)
                except Exception:
                    if new_schedule_id:
                        _cancel_schedule(new_schedule_id)
                    raise
            stored["schedule_id"] = new_schedule_id
            items[index] = stored
            try:
                _save(items)
            except Exception:
                rollback_error = None
                if new_schedule_id:
                    try:
                        _cancel_schedule(new_schedule_id)
                    except Exception as exc:
                        rollback_error = exc
                if old_schedule:
                    try:
                        _restore_schedule(old_schedule)
                    except Exception as exc:
                        rollback_error = rollback_error or exc
                if rollback_error:
                    raise HTTPException(500, "Calendar update failed and scheduler state could not be restored.") from rollback_error
                raise
            return stored
    raise HTTPException(404, "Calendar item not found.")


@app.delete("/api/items/{item_id}")
def delete_item(item_id: str) -> dict[str, bool]:
    with _calendar_lock():
        items = _load()
        kept = [item for item in items if item.get("id") != item_id]
        if len(kept) == len(items):
            raise HTTPException(404, "Calendar item not found.")
        removed = next(item for item in items if item.get("id") == item_id)
        schedule_id = str(removed["schedule_id"]) if removed.get("schedule_id") else None
        schedule = _schedule_record(schedule_id) if schedule_id else None
        if schedule_id:
            _cancel_schedule(schedule_id)
        try:
            _save(kept)
        except Exception:
            if schedule:
                _restore_schedule(schedule)
            raise
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
