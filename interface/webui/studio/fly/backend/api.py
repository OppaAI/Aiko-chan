"""Fly Circuit Studio — read-only NeuralState + connectome-motif graph.

Full FastAPI sub-app (same pattern as Log Studio): session-bound APIs,
static frontend, mounted at /studio/fly from interface.webui.auth.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from interface.webui.studio.session_binding import bind_login_session

app = FastAPI(title="Aiko Fly Circuit Studio")
bind_login_session(app)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="fly-frontend")

# Functional graph: not a full MaleCNS render — motif nodes + live signal strengths.
_CIRCUIT = {
    "nodes": [
        {"id": "sensory", "label": "Sensory", "x": 80, "y": 200, "group": "in"},
        {"id": "al", "label": "AL gain", "x": 200, "y": 120, "group": "sensory"},
        {"id": "t4t5", "label": "T4/T5 motion", "x": 200, "y": 280, "group": "sensory"},
        {"id": "mb", "label": "MB", "x": 360, "y": 100, "group": "core"},
        {"id": "lh", "label": "LH", "x": 360, "y": 220, "group": "core"},
        {"id": "cx", "label": "CX", "x": 360, "y": 340, "group": "core"},
        {"id": "gf", "label": "GF", "x": 520, "y": 200, "group": "interrupt"},
        {"id": "dn", "label": "DN", "x": 660, "y": 200, "group": "out"},
        {"id": "out", "label": "Aiko", "x": 800, "y": 200, "group": "out"},
    ],
    "edges": [
        {"from": "sensory", "to": "al"},
        {"from": "sensory", "to": "t4t5"},
        {"from": "al", "to": "mb"},
        {"from": "al", "to": "lh"},
        {"from": "t4t5", "to": "cx"},
        {"from": "mb", "to": "gf"},
        {"from": "lh", "to": "gf"},
        {"from": "cx", "to": "gf"},
        {"from": "mb", "to": "dn"},
        {"from": "lh", "to": "dn"},
        {"from": "cx", "to": "dn"},
        {"from": "gf", "to": "dn"},
        {"from": "dn", "to": "out"},
    ],
}


def _modes() -> dict:
    try:
        from system.config import env_str
        return {
            "MEMORY_FLYMB_MODE": env_str("MEMORY_FLYMB_MODE", "off"),
            "MEMORY_FLYCX_MODE": env_str("MEMORY_FLYCX_MODE", "off"),
            "MEMORY_FLYLH_MODE": env_str("MEMORY_FLYLH_MODE", "off"),
            "MEMORY_FLYGF_MODE": env_str("MEMORY_FLYGF_MODE", "off"),
            "MEMORY_FLYSLEEP_MODE": env_str("MEMORY_FLYSLEEP_MODE", "off"),
            "MEMORY_FLYDN_MODE": env_str("MEMORY_FLYDN_MODE", "off"),
        }
    except Exception:
        return {}


def _uid(request: Request) -> str | None:
    try:
        from system.userspace import current_user_id
        uid = current_user_id()
        if uid:
            return uid
    except Exception:
        pass
    return getattr(request.state, "user_id", None)


@app.get("/api/state")
def fly_state(request: Request) -> JSONResponse:
    uid = _uid(request)
    try:
        from cognition.neural_state import peek_neural_state
        state = peek_neural_state(uid)
        snap = state.snapshot() if state is not None else {}
    except Exception as exc:
        snap = {"error": str(exc)}
    return JSONResponse(
        {"user_id": uid, "neural_state": snap, "modes": _modes()},
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/circuit")
def fly_circuit(request: Request) -> JSONResponse:
    """Motif graph + live activation proxies derived from NeuralState."""
    uid = _uid(request)
    try:
        from cognition.neural_state import peek_neural_state
        state = peek_neural_state(uid)
        st = state.snapshot() if state is not None else {}
    except Exception:
        st = {}

    def act(*keys: str, scale: float = 1.0) -> float:
        vals = []
        for k in keys:
            v = st.get(k)
            if isinstance(v, (int, float)):
                vals.append(abs(float(v)))
        if not vals:
            return 0.15
        return max(0.08, min(1.0, max(vals) * scale))

    activation = {
        "sensory": act("sensory_gain", "motion_salience"),
        "al": act("sensory_gain"),
        "t4t5": act("motion_salience"),
        "mb": act("valence", "approach", "avoidance"),
        "lh": act("context_familiarity"),
        "cx": act("focus_sharpness", "decisiveness", "sleep_pressure"),
        "gf": act("urgency", scale=1.2) if not st.get("interrupt") else 1.0,
        "dn": act("action_drive", "motor_vigor"),
        "out": act("action_drive", "motor_vigor", "approach"),
    }
    return JSONResponse(
        {
            "user_id": uid,
            "nodes": _CIRCUIT["nodes"],
            "edges": _CIRCUIT["edges"],
            "activation": activation,
            "neural_state": st,
            "modes": _modes(),
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
