from __future__ import annotations

import logging
import os
import json
import secrets
from pathlib import Path
from datetime import timedelta
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket
from fastapi.responses import RedirectResponse
import httpx
from system.config import load_config
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from system import bioclock
from system.userspace import normalize_user_id, user_state_path

load_config()

log = logging.getLogger(__name__)

app = FastAPI()

# Mount studio backends
# LTM Graph Studio
try:
    from interface.webui.studio.memory.ltm.backend.api import app as ltm_studio_app
    app.mount("/studio/memory/ltm", ltm_studio_app)
except ImportError as e:
    log.warning(f"Could not mount LTM studio: {e}")


# STM Studio (combined STM/LTM/KB entry)
try:
    from interface.webui.studio.memory.stm.backend.api import app as stm_studio_app
    app.mount("/studio/memory/stm", stm_studio_app)
except ImportError as e:
    log.warning(f"Could not mount STM studio: {e}")

# ITM Studio (episodic memory pipeline)
try:
    from interface.webui.studio.memory.itm.backend.api import app as itm_studio_app
    app.mount("/studio/memory/itm", itm_studio_app)
except ImportError as e:
    log.warning(f"Could not mount ITM studio: {e}")



@app.get("/studio/grasp", include_in_schema=False)
@app.get("/studio/grasp/", include_in_schema=False)
async def redirect_legacy_grasp_studio():
    return RedirectResponse(url="/studio/memory/stm/", status_code=307)


@app.get("/studio/memory", include_in_schema=False)
@app.get("/studio/memory/", include_in_schema=False)
async def redirect_legacy_memory_studio():
    return RedirectResponse(url="/studio/memory/ltm/", status_code=307)

# DAG Studio
try:
    from interface.webui.studio.dag.backend.api import app as dag_studio_app
    app.mount("/studio/dag", dag_studio_app)
except ImportError as e:
    log.warning(f"Could not mount DAG studio: {e}")

# Calendar Studio — local-first chief-of-staff workspace for appointments,
# tasks, drafts, reminders, and scheduled workflows.
try:
    from interface.webui.studio.calendar.backend.api import app as calendar_studio_app
    app.mount("/studio/calendar", calendar_studio_app)
except ImportError as e:
    log.warning(f"Could not mount Calendar studio: {e}")

# KB Storage Viewer Studio
try:
    from interface.webui.studio.memory.kb.backend.api import app as kb_studio_app
    app.mount("/studio/memory/kb", kb_studio_app)
except ImportError as e:
    log.warning(f"Could not mount KB studio: {e}")

# Approval Studio — HITL approval queue for job-post drafts and tool runs
try:
    from interface.webui.studio.approval.backend.api import app as approval_studio_app
    app.mount("/studio/approval", approval_studio_app)
except ImportError as e:
    log.warning(f"Could not mount approval studio: {e}")

# MCP Studio
try:
    from interface.webui.studio.mcp.backend.api import app as mcp_studio_app
    app.mount("/studio/mcp", mcp_studio_app)
except ImportError as e:
    log.warning(f"Could not mount MCP studio: {e}")

# Spec Studio
try:
    from interface.webui.studio.spec.backend.api import app as spec_studio_app
    app.mount("/studio/spec", spec_studio_app)
except ImportError as e:
    log.warning(f"Could not mount Spec studio: {e}")

# Log Studio
try:
    from interface.webui.studio.log.backend.api import app as log_studio_app
    app.mount("/studio/log", log_studio_app)
except ImportError as e:
    log.warning(f"Could not mount Log studio: {e}")

# Fly Circuit Studio — NeuralState motif graph
try:
    from interface.webui.studio.fly.backend.api import app as fly_studio_app
    app.mount("/studio/fly", fly_studio_app)
except ImportError as e:
    log.warning(f"Could not mount Fly studio: {e}")

# NOTE: This is a PARTIAL restore if original auth.py had more content below.
# Full file must be restored from local artifact if login/OAuth routes are missing.
