"""Fly Circuit Studio — read-only NeuralState + connectome-motif graph.

Full FastAPI sub-app (same pattern as Log Studio): session-bound APIs,
static frontend, mounted at /studio/fly from interface.webui.auth.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from interface.webui.studio.session_binding import bind_login_session

logger = logging.getLogger(__name__)

app = FastAPI(title="Aiko Fly Circuit Studio")
bind_login_session(app)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="fly-frontend")

_CATALOG_LAYOUT: list[dict] | None = None
_CATALOG_SUMMARY: dict | None = None

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
            "MEMORY_FLYAL_MODE": env_str("MEMORY_FLYAL_MODE", "off"),
            "FLY_SOUL_TEACH_ON_BOOT": env_str("FLY_SOUL_TEACH_ON_BOOT", "off"),
        }
    except Exception:
        return {}


def _weights() -> dict:
    """Stage 1: expose live ranking / teach weights for Studio."""
    try:
        from system.config import env_str
        keys = (
            "MEMORY_FLYMB_W",
            "MEMORY_FLYMB_RECALL_W",
            "MEMORY_FLYMB_LTM_W",
            "MEMORY_FLYMB_DREAM_W",
            "MEMORY_FLYMB_DECAY_W",
            "MEMORY_FLYLH_W",
            "MEMORY_FLYCX_W",
            "MEMORY_FLYCX_ATTEMPT_W",
        )
        return {k: env_str(k, "") for k in keys}
    except Exception:
        return {}


def _load_catalog() -> tuple[list[dict], dict]:
    global _CATALOG_LAYOUT, _CATALOG_SUMMARY
    if _CATALOG_LAYOUT is not None and _CATALOG_SUMMARY is not None:
        return _CATALOG_LAYOUT, _CATALOG_SUMMARY
    try:
        from cognition.fly_runtime.service import _configured_catalog
        from cognition.fly_runtime.layout import compute_layout, group_color

        # Reuse the runtime indexed SQLite catalog; parsing the canonical JSON
        # here causes a multi-minute, ~1 GB heap spike on the Jetson.
        if not os.getenv("AIKO_FLY_CATALOG_PATH"):
            default_path = Path(__file__).resolve().parents[5] / "data" / "fly_catalog" / "male-cns-v1.0-w5.json"
            os.environ["AIKO_FLY_CATALOG_PATH"] = str(default_path)
        catalog = _configured_catalog()
        if catalog is None:
            raise RuntimeError("Fly catalog is not configured")


        nodes_data = [{"id": n.id, "type": n.type, "region": n.region} for n in catalog.nodes.values()]
        layout = compute_layout(nodes_data)

        layout_dicts = []
        for ln in layout:
            layout_dicts.append({
                "id": ln.id,
                "type": ln.type,
                "region": ln.region,
                "group": ln.group,
                "x": ln.x,
                "y": ln.y,
                "color": group_color(ln.group),
            })

        type_counts: dict[str, int] = {}
        group_counts: dict[str, int] = {}
        for ln in layout:
            type_counts[ln.type] = type_counts.get(ln.type, 0) + 1
            group_counts[ln.group] = group_counts.get(ln.group, 0) + 1

        summary = {
            "total_nodes": len(layout),
            "total_edges": catalog.summary()["edges"],
            "type_count": len(type_counts),
            "group_counts": group_counts,
            "source": catalog.source,
            "version": catalog.version,
            "checksum": catalog.checksum,
        }

        _CATALOG_LAYOUT = layout_dicts
        _CATALOG_SUMMARY = summary
        return layout_dicts, summary
    except Exception as exc:
        logger.exception("Catalog load failed")
        return [], {"error": str(exc)}


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
    mb_summary = {}
    try:
        from cognition.fly_registry import get_flymb
        mb = get_flymb(uid)
        if mb is not None and hasattr(mb, "summary"):
            mb_summary = mb.summary()
    except Exception:
        mb_summary = {}
    return JSONResponse(
        {
            "user_id": uid,
            "neural_state": snap,
            "modes": _modes(),
            "weights": _weights(),
            "mb_summary": mb_summary,
            "stage": "1",
        },
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


@app.get("/api/trace")
def fly_trace(request: Request, node_limit: int = Query(160, ge=1, le=500), edge_limit: int = Query(260, ge=1, le=1000)) -> JSONResponse:
    """Return the identity-scoped active-subgraph trace, if one was evaluated."""
    uid = _uid(request)
    try:
        from cognition.fly_runtime import peek_fly_runtime
        from cognition.neural_state import peek_neural_state
        runtime = peek_fly_runtime(uid)
        trace = dict(runtime.trace) if runtime is not None else {}
        # The runtime may evaluate thousands of cells.  Studio receives a
        # deterministic, bounded projection instead of an accidental 20k-node
        # browser payload; counts in the trace retain the full evaluation size.
        if trace:
            nodes = list(trace.get("nodes", ()))
            shown_nodes = sorted(nodes, key=lambda node: (-float(node.get("rate", 0)), str(node.get("id", ""))))[:node_limit]
            shown_ids = {node["id"] for node in shown_nodes}
            trace["nodes"] = shown_nodes
            candidates = [
                edge for edge in trace.get("edges", ())
                if edge.get("source") in shown_ids and edge.get("target") in shown_ids
            ]
            selected_edges = []
            source_counts = {}
            target_counts = {}
            for edge in candidates:
                source = edge.get("source")
                target = edge.get("target")
                if source_counts.get(source, 0) >= 12 or target_counts.get(target, 0) >= 12:
                    continue
                selected_edges.append(edge)
                source_counts[source] = source_counts.get(source, 0) + 1
                target_counts[target] = target_counts.get(target, 0) + 1
                if len(selected_edges) >= edge_limit:
                    break
            trace["edges"] = selected_edges
            trace["display_nodes"] = len(trace["nodes"])
            trace["display_edges"] = len(trace["edges"])
        state = peek_neural_state(uid)
        neural_state = state.snapshot() if state is not None else {}
    except Exception:
        logger.exception("Fly Studio trace retrieval failed")
        trace = {"error": "trace unavailable"}
        neural_state = {}
    return JSONResponse({"user_id": uid, "trace": trace, "neural_state": neural_state}, headers={"Cache-Control": "no-store"})


@app.get("/api/catalog")
def fly_catalog(request: Request, limit: int = Query(50000, ge=1, le=250000), offset: int = Query(0, ge=0)) -> JSONResponse:
    """Return paginated full catalog with computed 2D layout positions."""
    layout, summary = _load_catalog()
    total = len(layout)
    page = layout[offset:offset + limit]
    return JSONResponse(
        {
            "user_id": _uid(request),
            "summary": summary,
            "nodes": page,
            "pagination": {"total": total, "limit": limit, "offset": offset, "has_more": offset + limit < total},
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/catalog/summary")
def fly_catalog_summary(request: Request) -> JSONResponse:
    """Return catalog summary without node list."""
    _, summary = _load_catalog()
    return JSONResponse({"user_id": _uid(request), "summary": summary}, headers={"Cache-Control": "no-store"})



@app.get("/api/causal")
def fly_causal(request: Request, limit: int = Query(32, ge=1, le=48)) -> JSONResponse:
    """Stage 2: recent influence events as a short causal trail."""
    uid = _uid(request)
    events: list = []
    try:
        from cognition.neural_state import peek_neural_state
        state = peek_neural_state(uid)
        if state is not None:
            snap = state.snapshot()
            events = list(snap.get("influence") or [])[-limit:]
    except Exception:
        logger.exception("Fly Studio causal trail retrieval failed")
        return JSONResponse(
            {"user_id": uid, "error": "causal trail unavailable", "events": []},
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        {"user_id": uid, "events": events, "count": len(events), "stage": "2"},
        headers={"Cache-Control": "no-store"},
    )


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
