"""Studio backend for Aiko's graph visualiser/editor (n8n-style).

Endpoints
---------
  GET    /api/playbooks                  list (always refreshed from disk)
  GET    /api/playbooks/{id}             one graph
  POST   /api/playbooks                  create new graph
  PUT    /api/playbooks/{id}             save edited nodes/edges/positions
  DELETE /api/playbooks/{id}             delete a user graph (built-ins protected)
  POST   /api/playbooks/{id}/duplicate   clone under a new id
  POST   /api/playbooks/{id}/validate    cycles / deps / unknown tools
  POST   /api/playbooks/{id}/run         bounded dry-run (Jetson-safe)
  GET    /api/nodes                      palette: categories + per-node param forms
  GET    /api/tools                      legacy flat palette (kept for old clients)
  GET    /api/templates                  starter workflow blueprints
  POST   /api/templates/{id}/create      instantiate a template as a new graph

What changed vs. the first version
----------------------------------
  - ``position`` is preserved per node, so the canvas remembers your layout
    instead of re-deriving it from the dependency graph on every load.
  - ``sticky_note`` nodes are stored but never executed.
  - ``disabled`` nodes are stored, skipped at run time, and their dependants
    are rewired to the disabled node's own dependencies (n8n behaviour).
  - ``pinned_data`` replaces a node's tool with a manual trigger at run time,
    so you can iterate on the back half of a workflow without re-hitting an
    API on every click.
  - ``start_node`` runs only that node and its descendants.
  - Unknown tools are warnings on save, errors only on run — a lagging
    palette should never block you from saving work in progress.

Jetson notes: dry-runs cap at 2 parallel workers and 120s wall-clock, so a
browser click can never OOM or wedge the Orin Nano.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Aiko Graph Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STUDIO_DIR = BASE_DIR
FRONTEND_DIR = STUDIO_DIR / "frontend"
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="dag-frontend")
app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="studio-shared")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")

# Nodes that live on the canvas but are never executed.
DECORATIVE_TOOLS = {"sticky_note"}

# Fields the studio may attach to a node and expects to get back verbatim.
_PASSTHROUGH_NODE_FIELDS = (
    "position", "notes", "label", "disabled", "pinned_data", "color",
    "run_if", "when", "loop_to", "loop_condition", "max_visits", "interrupt",
    "timeout_seconds", "max_retries", "retry_backoff_seconds", "fallback_to",
    "needs_approval", "attached_to",
)

# Populate the registry (toolkit @tool decorators + shared workflow nodes +
# the n8n-style flow nodes) so validate/run see exactly what the runtime sees.
# Import failures must never take the studio down — a Jetson can boot with
# optional lanes missing.
for _module in (
    "agentic.tools",
    "agentic.toolkit.flow",
    "agentic.workflows.common.nodes",
):
    try:
        __import__(_module)
    except Exception:  # pragma: no cover
        pass


def _valid_id(value: str) -> bool:
    return bool(value and _ID_RE.match(value))


# ── playbook <-> graph shape ─────────────────────────────────────────────────

def playbook_to_graph(playbook: dict) -> dict:
    """Derive the edge array the canvas draws from the node dependency data."""
    if not isinstance(playbook.get("nodes"), list):
        return playbook
    nodes = playbook["nodes"]
    node_map = {n.get("id"): n for n in nodes if isinstance(n, dict) and n.get("id")}
    edges = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if not nid:
            continue
        src = node_map.get(nid, {})
        tool_call = {"tool": src.get("tool"), "args": src.get("args")}
        for dep in node.get("depends_on") or []:
            edges.append({"source": dep, "target": nid, "type": "depends_on",
                          "tool_call": tool_call, "skill": src.get("tool")})
        if node.get("loop_to"):
            edges.append({"source": nid, "target": node["loop_to"], "type": "loop_to",
                          "tool_call": tool_call, "skill": src.get("tool")})
        if node.get("fallback_to"):
            edges.append({"source": nid, "target": node["fallback_to"], "type": "fallback_to",
                          "tool_call": tool_call, "skill": src.get("tool")})
    return {**playbook, "nodes": nodes, "edges": edges}


def load_playbooks_refresh() -> list:
    """Reload playbooks from graph_engine, resolving registered Spec graphs."""
    from agentic.graph_engine import load_playbooks
    from agentic.workflows.common.graphs import get_graph

    result = []
    for playbook in load_playbooks():
        graph_view = playbook_to_graph(playbook)
        graph_id = graph_view.get("graph_id")
        if graph_id and not graph_view.get("nodes"):
            graph = get_graph(graph_id)
            if graph:
                nodes, edges = [], []
                for node in graph.nodes:
                    nodes.append({
                        "id": node.id,
                        "tool": node.tool,
                        "args": dict(node.args or {}),
                        "depends_on": list(node.depends_on or ()),
                        "loop_to": getattr(node, "loop_to", None),
                        "loop_condition": getattr(node, "loop_condition", None),
                        "max_visits": getattr(node, "max_visits", None),
                        "fallback_to": getattr(node, "fallback_to", None),
                        "run_if": getattr(node, "run_if", None),
                    })
                    for dep in node.depends_on or ():
                        edges.append({"source": dep, "target": node.id, "type": "depends_on"})
                    if getattr(node, "loop_to", None):
                        edges.append({"source": node.id, "target": node.loop_to, "type": "loop_to"})
                    if getattr(node, "fallback_to", None):
                        edges.append({"source": node.id, "target": node.fallback_to, "type": "fallback_to"})
                graph_view = {**graph_view, "nodes": nodes, "edges": edges, "readonly": True}
        result.append(graph_view)
    return result


def _builtin_ids() -> set[str]:
    """IDs shipped in code defaults — protected from DELETE."""
    try:
        from agentic.graph_engine import _default_playbooks

        return {str(p.get("id")) for p in _default_playbooks() if p.get("id")}
    except Exception:
        return set()


def _known_tools() -> set[str]:
    try:
        from agentic.registry import registry

        return set(registry.get_all_tool_names()) | set(DECORATIVE_TOOLS)
    except Exception:
        return set(DECORATIVE_TOOLS)


# ── node validation / cleaning ───────────────────────────────────────────────

def _clean_nodes(nodes_raw: Any) -> tuple[list[dict], list[str], list[str]]:
    """Validate and normalise a node list.

    Returns ``(clean_nodes, errors, warnings)``. Structural problems (missing
    ids, duplicates, dangling deps, cycles) are errors; an unrecognised tool
    name is only a warning so half-finished work still saves.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(nodes_raw, list) or not nodes_raw:
        return [], ["playbook requires a non-empty nodes list"], warnings

    clean_nodes: list[dict] = []
    node_ids: set[str] = set()
    for raw in nodes_raw:
        if not isinstance(raw, dict) or not raw.get("id") or not raw.get("tool"):
            errors.append("each node requires id and tool")
            continue
        node_id = str(raw["id"])[:64]
        if not _valid_id(node_id):
            errors.append(f"invalid node id: {node_id}")
            continue
        if node_id in node_ids:
            errors.append(f"duplicate node id: {node_id}")
            continue
        node_ids.add(node_id)

        node: dict[str, Any] = {"id": node_id, "tool": str(raw["tool"])[:120]}
        args = raw.get("args")
        if args is not None and not isinstance(args, dict):
            errors.append(f"args must be an object for node {node_id}")
            continue
        node["args"] = dict(args or {})
        node["depends_on"] = [str(dep) for dep in (raw.get("depends_on") or [])]

        for field in _PASSTHROUGH_NODE_FIELDS:
            if field in raw and raw[field] not in (None, ""):
                node[field] = raw[field]

        position = node.get("position")
        if isinstance(position, dict):
            try:
                node["position"] = {"x": round(float(position.get("x", 0)), 1),
                                    "y": round(float(position.get("y", 0)), 1)}
            except (TypeError, ValueError):
                node.pop("position", None)
        else:
            node.pop("position", None)

        clean_nodes.append(node)

    for node in clean_nodes:
        for dep in node["depends_on"]:
            if dep not in node_ids:
                errors.append(f"unknown dependency for {node['id']}: {dep}")
        for field in ("loop_to", "fallback_to"):
            if node.get(field) and str(node[field]) not in node_ids:
                errors.append(f"unknown {field} for {node['id']}: {node[field]}")

    # Cycle check over depends_on only (loop_to is intentionally cyclic).
    if not errors:
        visiting: set[str] = set()
        visited: set[str] = set()
        adjacency = {n["id"]: list(n.get("depends_on") or []) for n in clean_nodes}

        def _visit(nid: str, stack: list[str]) -> bool:
            if nid in visiting:
                errors.append(f"cycle detected: {' -> '.join(stack + [nid])}")
                return True
            if nid in visited:
                return False
            visiting.add(nid)
            for dep in adjacency.get(nid, []):
                if _visit(dep, stack + [nid]):
                    return True
            visiting.discard(nid)
            visited.add(nid)
            return False

        for nid in adjacency:
            if _visit(nid, []):
                break

    known = _known_tools()
    if known:
        for node in clean_nodes:
            if node["tool"] not in known:
                warnings.append(f"unknown tool for {node['id']}: {node['tool']}")

    return clean_nodes, errors, warnings


def _write_playbook_entry(playbook_id: str, clean: dict) -> Path:
    from agentic.graph_engine import _playbook_file, _playbook_write_guard

    path = _playbook_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _playbook_write_guard(path):
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"failed to read playbook source: {exc}") from exc
        if not isinstance(existing, list):
            existing = []
        for index, item in enumerate(existing):
            if isinstance(item, dict) and item.get("id") == playbook_id:
                existing[index] = clean
                break
        else:
            existing.append(clean)
        tmp = path.with_suffix(path.suffix + ".studio.tmp")
        try:
            tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(status_code=500, detail=f"failed to write playbook source: {exc}") from exc
    return path


# ── run-time graph preparation ───────────────────────────────────────────────

def _strip_decorative(nodes: list[dict]) -> list[dict]:
    keep = [n for n in nodes if n.get("tool") not in DECORATIVE_TOOLS]
    ids = {n["id"] for n in keep}
    for node in keep:
        node["depends_on"] = [d for d in node.get("depends_on") or [] if d in ids]
        for field in ("loop_to", "fallback_to"):
            if node.get(field) and node[field] not in ids:
                node.pop(field, None)
    return keep


def _drop_disabled(nodes: list[dict]) -> list[dict]:
    """Remove disabled nodes and rewire their dependants to their own deps."""
    disabled = {n["id"]: list(n.get("depends_on") or []) for n in nodes if n.get("disabled")}
    if not disabled:
        return nodes
    kept = [n for n in nodes if not n.get("disabled")]
    for node in kept:
        resolved: list[str] = []
        frontier = list(node.get("depends_on") or [])
        seen: set[str] = set()
        while frontier:
            dep = frontier.pop(0)
            if dep in seen:
                continue
            seen.add(dep)
            if dep in disabled:
                frontier.extend(disabled[dep])
            elif dep not in resolved:
                resolved.append(dep)
        node["depends_on"] = resolved
        for field in ("loop_to", "fallback_to"):
            if node.get(field) in disabled:
                node.pop(field, None)
    return kept


def _apply_pins(nodes: list[dict]) -> list[dict]:
    """Swap pinned nodes for a manual trigger carrying the pinned payload."""
    for node in nodes:
        pinned = node.get("pinned_data")
        if pinned in (None, "", [], {}):
            continue
        payload = pinned if isinstance(pinned, str) else json.dumps(pinned, ensure_ascii=False)
        node["tool"] = "trigger_manual"
        node["args"] = {
            "items_json": payload,
            "to_state": (node.get("args") or {}).get("to_state") or "items",
        }
        node["depends_on"] = []
        node["_pinned"] = True
    return nodes


def _descendants(nodes: list[dict], start: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for node in nodes:
        for dep in node.get("depends_on") or []:
            children.setdefault(dep, []).append(node["id"])
    out = {start}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for child in children.get(current, []):
            if child not in out:
                out.add(child)
                frontier.append(child)
    return out


def _prepare_run_nodes(nodes: list[dict], start_node: str = "") -> list[dict]:
    prepared = _apply_pins(_drop_disabled(_strip_decorative([dict(n) for n in nodes])))
    if start_node:
        keep = _descendants(prepared, start_node)
        prepared = [n for n in prepared if n["id"] in keep]
        ids = {n["id"] for n in prepared}
        for node in prepared:
            node["depends_on"] = [d for d in node.get("depends_on") or [] if d in ids]
    return prepared


# ── startup snapshot ─────────────────────────────────────────────────────────

try:
    PLAYBOOKS = load_playbooks_refresh()
except Exception as _pb_exc:  # pragma: no cover
    import logging as _logging

    _logging.getLogger(__name__).warning("DAG studio: initial playbook load failed: %s", _pb_exc)
    PLAYBOOKS = []


# ── playbook CRUD ────────────────────────────────────────────────────────────

@app.get("/api/playbooks")
def get_playbooks():
    """All playbooks, refreshed from disk."""
    global PLAYBOOKS
    PLAYBOOKS = load_playbooks_refresh()
    return PLAYBOOKS


@app.get("/api/playbooks/{playbook_id}")
def get_playbook(playbook_id: str):
    global PLAYBOOKS
    PLAYBOOKS = load_playbooks_refresh()
    for playbook in PLAYBOOKS:
        if playbook.get("id") == playbook_id:
            return playbook
    raise HTTPException(status_code=404, detail="Playbook not found")


@app.post("/api/playbooks")
async def create_playbook(request: Request):
    """Create a new user graph (Workflow → New)."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="object required")

    playbook_id = str(payload.get("id") or "").strip()
    if not _valid_id(playbook_id):
        raise HTTPException(status_code=400, detail="id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    if playbook_id in {str(p.get("id")) for p in load_playbooks_refresh()}:
        raise HTTPException(status_code=409, detail=f"playbook already exists: {playbook_id}")

    nodes_raw = payload.get("nodes") or [{
        "id": "trigger", "tool": "trigger_manual",
        "args": {"items_json": "[{}]", "to_state": "items"},
        "position": {"x": 120, "y": 160},
    }]
    clean_nodes, errors, warnings = _clean_nodes(nodes_raw)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors[:5]))

    clean = {
        "id": playbook_id,
        "name": str(payload.get("name") or playbook_id)[:120],
        "goal": str(payload.get("goal") or payload.get("name") or playbook_id)[:500],
        "description": str(payload.get("description") or "")[:800],
        "triggers": list(payload.get("triggers") or []),
        "requires_any": list(payload.get("requires_any") or []),
        "capabilities": list(payload.get("capabilities") or []),
        "source": "studio",
        "nodes": clean_nodes,
    }
    if payload.get("from_template"):
        clean["from_template"] = str(payload["from_template"])[:64]
    path = _write_playbook_entry(playbook_id, clean)
    return {"ok": True, "playbook": playbook_to_graph(clean), "path": str(path), "warnings": warnings}


@app.put("/api/playbooks/{playbook_id}")
async def save_playbook(playbook_id: str, request: Request):
    """Persist an edited graph to the user-scoped source the runtime reads."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("id") != playbook_id:
        raise HTTPException(status_code=400, detail="payload id must match playbook id")

    clean_nodes, errors, warnings = _clean_nodes(payload.get("nodes"))
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors[:5]))

    clean = {k: v for k, v in payload.items()
             if k not in {"edges", "nodes", "readonly"} and not k.startswith("_")}
    clean["id"] = playbook_id
    clean["nodes"] = clean_nodes
    path = _write_playbook_entry(playbook_id, clean)
    return {"ok": True, "playbook": playbook_to_graph(clean), "path": str(path), "warnings": warnings}


@app.delete("/api/playbooks/{playbook_id}")
def delete_playbook(playbook_id: str):
    """Delete a user graph. Built-in defaults are protected — duplicate first."""
    from agentic.graph_engine import _playbook_file, _playbook_write_guard

    if playbook_id in _builtin_ids():
        raise HTTPException(status_code=400, detail="built-in playbook — duplicate it first, then edit the copy")
    path = _playbook_file()
    with _playbook_write_guard(path):
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"failed to read playbook source: {exc}") from exc
        if not isinstance(existing, list):
            existing = []
        kept = [p for p in existing if not (isinstance(p, dict) and p.get("id") == playbook_id)]
        if len(kept) == len(existing):
            raise HTTPException(status_code=404, detail="playbook not found in user store")
        tmp = path.with_suffix(path.suffix + ".studio.tmp")
        tmp.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    return {"ok": True, "deleted": playbook_id}


@app.post("/api/playbooks/{playbook_id}/duplicate")
async def duplicate_playbook(playbook_id: str, request: Request):
    """Clone a graph (including built-ins) under a new id."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    new_id = str((body or {}).get("new_id") or f"{playbook_id}_copy").strip()
    if not _valid_id(new_id):
        raise HTTPException(status_code=400, detail="new_id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

    current = {str(p.get("id")): p for p in load_playbooks_refresh()}
    src = current.get(playbook_id)
    if src is None:
        raise HTTPException(status_code=404, detail="Playbook not found")
    if new_id in current:
        raise HTTPException(status_code=409, detail=f"playbook already exists: {new_id}")

    clone = {k: v for k, v in src.items()
             if not k.startswith("_") and k not in {"edges", "readonly"}}
    clone["id"] = new_id
    clone["name"] = f"{src.get('name') or playbook_id} (copy)"
    clone["source"] = "studio"
    if not isinstance(clone.get("nodes"), list) or not clone["nodes"]:
        raise HTTPException(status_code=400, detail="source has no editable nodes")
    clean_nodes, errors, warnings = _clean_nodes(clone["nodes"])
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors[:5]))
    clone["nodes"] = clean_nodes
    path = _write_playbook_entry(new_id, clone)
    return {"ok": True, "playbook": playbook_to_graph(clone), "path": str(path), "warnings": warnings}


# ── palette ──────────────────────────────────────────────────────────────────

@app.get("/api/nodes")
def list_nodes():
    """Palette source: categories, labels and per-node parameter forms."""
    try:
        from agentic.node_catalog import node_catalog

        return node_catalog()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"node catalog failed: {exc}") from exc


@app.get("/api/tools")
def list_tools():
    """Legacy flat palette (registry tools grouped by domain)."""
    try:
        from agentic.registry import registry

        groups: dict[str, list[dict]] = {}
        for spec in sorted(registry.all_specs(), key=lambda s: s.name):
            if not spec.graph and not spec.react:
                continue
            entry = {
                "name": spec.name,
                "description": (spec.description or "")[:160],
                "domain": spec.domain or "general",
                "graph": spec.graph, "react": spec.react,
                "needs_approval": spec.needs_approval,
                "args": list((spec.props or {}).keys())[:8],
            }
            groups.setdefault(entry["domain"], []).append(entry)
        return {"groups": groups, "count": sum(len(v) for v in groups.values())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"tool list failed: {exc}") from exc


# ── templates ────────────────────────────────────────────────────────────────

@app.get("/api/templates")
def get_templates():
    """Starter blueprints for the 'New from template' gallery."""
    try:
        from agentic.workflows.templates import list_templates

        return {"templates": list_templates()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"template list failed: {exc}") from exc


@app.get("/api/templates/{template_id}")
def get_template_detail(template_id: str):
    from agentic.workflows.templates import get_template

    tpl = get_template(template_id)
    if tpl is None:
        raise HTTPException(status_code=404, detail="template not found")
    return playbook_to_graph(tpl)


@app.post("/api/templates/{template_id}/create")
async def create_from_template(template_id: str, request: Request):
    """Instantiate a template as a brand-new, editable user workflow."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    new_id = str((body or {}).get("id") or f"{template_id}_1").strip()
    if not _valid_id(new_id):
        raise HTTPException(status_code=400, detail="id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    if new_id in {str(p.get("id")) for p in load_playbooks_refresh()}:
        raise HTTPException(status_code=409, detail=f"playbook already exists: {new_id}")

    from agentic.workflows.templates import instantiate

    clean = instantiate(template_id, new_id,
                        name=str((body or {}).get("name") or ""),
                        goal=str((body or {}).get("goal") or ""))
    if clean is None:
        raise HTTPException(status_code=404, detail="template not found")

    clean_nodes, errors, warnings = _clean_nodes(clean["nodes"])
    if errors:
        raise HTTPException(status_code=500, detail=f"template is invalid: {'; '.join(errors[:3])}")
    clean["nodes"] = clean_nodes
    path = _write_playbook_entry(new_id, clean)
    return {"ok": True, "playbook": playbook_to_graph(clean), "path": str(path), "warnings": warnings}


# ── validate / run ───────────────────────────────────────────────────────────

@app.post("/api/playbooks/{playbook_id}/validate")
def validate_playbook(playbook_id: str):
    """Structural check without executing: cycles, deps, unknown tools."""
    current = {str(p.get("id")): p for p in load_playbooks_refresh()}
    playbook = current.get(playbook_id)
    if playbook is None:
        raise HTTPException(status_code=404, detail="Playbook not found")

    nodes = [n for n in (playbook.get("nodes") or []) if isinstance(n, dict)]
    clean_nodes, errors, warnings = _clean_nodes(nodes)
    runnable = _prepare_run_nodes(clean_nodes)
    entry_points = [n["id"] for n in runnable if not (n.get("depends_on") or [])]

    if not entry_points and runnable:
        warnings.append("no entry node — every node has dependencies")
    if len(runnable) > 20:
        warnings.append("large graph (>20 nodes) — consider splitting it for a 3B model's context")
    for node in runnable:
        if node.get("loop_to") and not node.get("max_visits"):
            warnings.append(f"{node['id']} loops without max_visits — it will only run once")
        if node.get("loop_to") and not node.get("loop_condition"):
            warnings.append(f"{node['id']} has loop_to but no loop_condition")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "nodes": len(clean_nodes),
        "runnable_nodes": len(runnable),
        "entry_points": entry_points,
    }


@app.post("/api/playbooks/{playbook_id}/run")
async def run_playbook_dry(request: Request, playbook_id: str):
    """Bounded dry-run with Jetson-safe limits.

    Body: ``{"prompt": "...", "timeout_s": 60, "start_node": "draft"}``.
    Approval-gated tools keep their normal refusal — that refusal is still
    useful signal in the studio, so it is reported rather than bypassed.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body or {}
    prompt = str(body.get("prompt") or f"Studio dry-run of {playbook_id}")[:1500]
    timeout_s = max(10, min(int(body.get("timeout_s") or 60), 120))
    start_node = str(body.get("start_node") or "").strip()

    current = {str(p.get("id")): p for p in load_playbooks_refresh()}
    playbook = current.get(playbook_id)
    if playbook is None:
        raise HTTPException(status_code=404, detail="Playbook not found")

    clean_nodes, errors, warnings = _clean_nodes(playbook.get("nodes"))
    if errors:
        raise HTTPException(status_code=400, detail="graph has structural errors — validate first")

    runnable = _prepare_run_nodes(clean_nodes, start_node=start_node)
    if not runnable:
        raise HTTPException(status_code=400, detail="nothing to run (all nodes disabled or decorative)")

    known = _known_tools()
    missing = sorted({n["tool"] for n in runnable if n["tool"] not in known})
    if missing:
        raise HTTPException(status_code=400, detail=f"unknown tool(s): {', '.join(missing[:5])}")

    try:
        from agentic.graph_engine import PlanGraph, PlanNode, execute_graph

        plan_nodes = []
        for node in runnable:
            args = {
                key: (value.replace("$prompt", prompt) if isinstance(value, str) else value)
                for key, value in (node.get("args") or {}).items()
            }
            plan_nodes.append(PlanNode(
                id=node["id"],
                tool=node["tool"],
                args=args,
                depends_on=tuple(node.get("depends_on") or ()),
                run_if=node.get("run_if") if isinstance(node.get("run_if"), dict) else None,
                when=node.get("when") if isinstance(node.get("when"), dict) else None,
                loop_to=node.get("loop_to"),
                loop_condition=node.get("loop_condition") if isinstance(node.get("loop_condition"), dict) else None,
                max_visits=int(node.get("max_visits") or 1),
                interrupt=bool(node.get("interrupt", False)),
                timeout_seconds=float(node["timeout_seconds"]) if node.get("timeout_seconds") else None,
                max_retries=int(node.get("max_retries") or 0),
                retry_backoff_seconds=float(node.get("retry_backoff_seconds") or 1.0),
                fallback_to=node.get("fallback_to"),
                needs_approval=bool(node.get("needs_approval", False)),
            ))

        graph = PlanGraph(
            id=playbook_id,
            name=str(playbook.get("name") or playbook_id),
            goal=str(playbook.get("goal") or prompt[:300]),
            nodes=tuple(plan_nodes),
            source="studio-dryrun",
            reducers=dict(playbook.get("reducers") or {}),
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(execute_graph, graph).result(timeout=timeout_s)

        return {
            "ok": True,
            "graph": playbook_id,
            "started_at_node": start_node or None,
            "warnings": warnings,
            "final_answer": (result.final_answer or "")[:3000],
            "goal_score": result.goal_score,
            "goal_reasons": result.goal_reasons,
            "state_keys": sorted(k for k in (result.final_state or {}) if not str(k).startswith("_"))[:40],
            "nodes": [
                {
                    "id": r.node_id, "tool": r.tool, "ok": r.ok,
                    "error": r.error_type,
                    "content": (r.content or "")[:2000],
                    "args": {k: str(v)[:200] for k, v in (r.args or {}).items()},
                }
                for r in result.results
            ],
        }
    except concurrent.futures.TimeoutError:
        raise HTTPException(status_code=504, detail=f"dry-run exceeded {timeout_s}s — Jetson cap")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"dry-run failed: {exc}") from exc


@app.get("/")
async def serve_studio(request: Request):
    """Serve the studio SPA."""
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
