"""Inner Speech Studio — read-only thought journal.

Aiko's inner speech is private by design: it is never copied into chat,
never spoken, never recalled into ordinary context. This studio is the
one sanctioned window: Oppa can *inspect* what she thought and when it
fired, but nothing here can write, edit, delete, or inject thoughts back
into her cognition. Read-only is the privacy boundary.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from interface.webui.studio.session_binding import bind_login_session

logger = logging.getLogger(__name__)

app = FastAPI(title="Aiko Inner Speech Studio")
bind_login_session(app)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True),
              name="innerspeech-frontend")
if SHARED_DIR.is_dir():
    app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True),
              name="studio-shared")


def _open_db(user_id: str) -> sqlite3.Connection | None:
    try:
        from cognition.memory.inner_speech_store import _db_path

        path = _db_path(user_id)
        p = Path(path)
        if not p.exists():
            return None
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.debug("innerspeech studio: cannot open db: %s", e)
        return None


def _thoughts(conn: sqlite3.Connection, user_id: str, *, limit: int,
              offset: int, kind: str, source: str, q: str) -> list[dict[str, Any]]:
    if q:
        rows = conn.execute(
            """SELECT t.id, t.text, t.kind, t.source, t.created_at,
                      t.consolidated, t.access_count
               FROM thoughts_fts f
               JOIN thoughts t ON t.rowid = f.rowid
               WHERE t.user_id = ? AND thoughts_fts MATCH ?
               ORDER BY t.created_at DESC LIMIT ? OFFSET ?""",
            (user_id, q, limit, offset),
        ).fetchall()
    else:
        cond: list[str] = ["t.user_id = ?"]
        args: list[Any] = [user_id]
        if kind:
            cond.append("t.kind = ?")
            args.append(kind)
        if source:
            cond.append("t.source = ?")
            args.append(source)
        rows = conn.execute(
            "SELECT t.id, t.text, t.kind, t.source, t.created_at,"
            " t.consolidated, t.access_count FROM thoughts t"
            f" WHERE {' AND '.join(cond)}"
            " ORDER BY t.created_at DESC LIMIT ? OFFSET ?",
            (*args, limit, offset),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "text": r["text"],
            "kind": r["kind"],
            "source": r["source"],
            "created_at": r["created_at"],
            "consolidated": bool(r["consolidated"]),
            "access_count": r["access_count"],
        }
        for r in rows
    ]


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "studio": "innerspeech", "mode": "read-only"}


@app.get("/api/stats")
def stats() -> JSONResponse:
    from system.userspace import current_user_id

    user_id = current_user_id()
    conn = _open_db(user_id)
    if conn is None:
        return JSONResponse({"total": 0, "exists": False})
    try:
        total = conn.execute(
            "SELECT COUNT(*) c FROM thoughts WHERE user_id = ?", (user_id,)
        ).fetchone()["c"]
        by_kind = {
            r["kind"]: r["c"]
            for r in conn.execute(
                "SELECT kind, COUNT(*) c FROM thoughts WHERE user_id = ? GROUP BY kind",
                (user_id,),
            )
        }
        by_source = {
            r["source"]: r["c"]
            for r in conn.execute(
                "SELECT source, COUNT(*) c FROM thoughts WHERE user_id = ? GROUP BY source",
                (user_id,),
            )
        }
        consolidated = conn.execute(
            "SELECT COUNT(*) c FROM thoughts WHERE user_id = ? AND consolidated = 1",
            (user_id,),
        ).fetchone()["c"]
        latest = conn.execute(
            "SELECT MAX(created_at) m FROM thoughts WHERE user_id = ?", (user_id,)
        ).fetchone()["m"]
        return JSONResponse({
            "exists": True,
            "total": total,
            "by_kind": by_kind,
            "by_source": by_source,
            "consolidated": consolidated,
            "unconsolidated": total - consolidated,
            "latest_at": latest,
        })
    finally:
        conn.close()


@app.get("/api/thoughts")
def thoughts(
    limit: int = 50,
    offset: int = 0,
    kind: str = "",
    source: str = "",
    q: str = "",
) -> JSONResponse:
    from system.userspace import current_user_id

    user_id = current_user_id()
    conn = _open_db(user_id)
    if conn is None:
        return JSONResponse({"thoughts": [], "exists": False})
    try:
        limit = max(1, min(200, int(limit or 50)))
        offset = max(0, int(offset or 0))
        return JSONResponse({
            "exists": True,
            "thoughts": _thoughts(conn, user_id, limit=limit, offset=offset,
                                  kind=kind or "", source=source or "", q=q or ""),
        })
    finally:
        conn.close()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))
