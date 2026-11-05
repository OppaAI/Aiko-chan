"""Read-only Log Studio API for browsing ``logs/aiko.log``."""
from __future__ import annotations

import re
from datetime import date, datetime, time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from interface.webui.studio.session_binding import bind_login_session
from system.log import LOG_FILE

app = FastAPI(title="Aiko Log Studio")
bind_login_session(app)
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="log-frontend")

_RECORD = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})  \[(?P<level>[A-Z]+)\s*\]  (?P<component>.*?) — (?P<message>.*)$")
_MAX_ENTRIES = 10_000
_MAX_LOG_BYTES = 8 * 1024 * 1024


def parse_log_records(text: str) -> list[dict[str, str]]:
    """Split Aiko's formatted log into records, retaining traceback lines."""
    records, current = [], None
    for line in text.splitlines():
        match = _RECORD.match(line)
        if match:
            if current:
                records.append(current)
            current = match.groupdict()
        elif current:
            current["message"] += f"\n{line}"
    if current:
        records.append(current)
    return records


def _parse_boundary(value: str | None, *, end: bool) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.combine(date.fromisoformat(value), time.max if end else time.min)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Dates must use YYYY-MM-DD.") from exc


def read_records(log_file: Path = Path(LOG_FILE)) -> list[dict[str, str]]:
    try:
        with log_file.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - _MAX_LOG_BYTES))
            if size > _MAX_LOG_BYTES:
                stream.readline(_MAX_LOG_BYTES)  # discard incomplete first record line
            text = stream.read(_MAX_LOG_BYTES).decode("utf-8", errors="replace")
        return parse_log_records(text)[-_MAX_ENTRIES:]
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not read aiko.log.") from exc


def require_operator():
    from system.userspace import can_approve, current_user_id
    if not can_approve(current_user_id()):
        raise HTTPException(status_code=403, detail="System logs are available to the creator and administrators only.")


@app.get("/api/options", dependencies=[Depends(require_operator)])
def options():
    records = read_records()
    return {"levels": sorted({r["level"] for r in records}), "components": sorted({r["component"] for r in records}, key=str.casefold), "total": len(records)}


@app.get("/api/entries", dependencies=[Depends(require_operator)])
def entries(
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    level: list[str] | None = Query(None),
    component: list[str] | None = Query(None),
    last: int | None = Query(None, ge=1, le=_MAX_ENTRIES),
):
    start, end = _parse_boundary(date_from, end=False), _parse_boundary(date_to, end=True)
    if start and end and start > end:
        raise HTTPException(status_code=422, detail="From date must be on or before to date.")
    levels, components = {item.upper() for item in level or []}, set(component or [])
    records = read_records()
    def matches(record: dict[str, str]) -> bool:
        timestamp = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
        return ((not start or timestamp >= start) and (not end or timestamp <= end) and (not levels or record["level"] in levels) and (not components or record["component"] in components))
    filtered = [record for record in records if matches(record)]
    if last is None:
        filtered = filtered[-_MAX_ENTRIES:]
    else:
        filtered = filtered[-last:]
    return {"entries": filtered, "count": len(filtered), "total": len(records)}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
