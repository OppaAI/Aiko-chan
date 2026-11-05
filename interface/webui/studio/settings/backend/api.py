"""Settings Studio backend — tabbed editor for config/*.yaml + .env.age.

Session-bound like every other studio: bind_login_session gates /api/*
behind the WebUI login; the static page itself is public shell only.
All editing rules live in settings_core (allowlist, validation, comment-
preserving YAML writes, masked secrets). Audited sampling controls apply on the next request.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from interface.webui.studio.session_binding import bind_login_session
from interface.webui.studio.settings.backend import settings_core as core
from system.log import get_logger

log = get_logger(__name__)

app = FastAPI(title="Aiko Settings Studio")
bind_login_session(app)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="settings-frontend")
app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="settings-shared")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


def require_operator():
    from system.userspace import can_approve, current_user_id
    if not can_approve(current_user_id()):
        raise HTTPException(status_code=403, detail="Settings are available to the creator and administrators only.")


@app.get("/api/settings", dependencies=[Depends(require_operator)])
def get_settings():
    try:
        return core.read_all()
    except (RuntimeError, OSError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=503)


@app.post("/api/settings", dependencies=[Depends(require_operator)])
async def post_settings(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "request body must be JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"detail": "request body must be a JSON object"}, status_code=400)
    try:
        result = await run_in_threadpool(core.write_settings, body.get("settings", {}),
                                     body.get("secrets", {}), apply_live=True, revision=body.get("revision"))
    except core.SettingsConflict as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)
    except (RuntimeError, OSError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=503)
    log.info("settings updated: %s", ", ".join(result["updated"] + result["secrets_updated"]))
    return result
