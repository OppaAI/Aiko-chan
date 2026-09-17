"""
agentic/node_catalog.py

Presentation metadata for the DAG studio's node palette and parameter forms.

The tool *registry* (agentic/registry.py) knows what a tool is and how to call
it. It does not know what a human needs to see to configure one: a friendly
label, a category, which parameters are dropdowns vs. JSON blobs, what a sane
default looks like, or which outputs a branching node produces.

That is this module. Keeping it separate means:
  - the runtime never imports UI concerns;
  - every registered tool still shows up in the palette even if nobody wrote
    curated metadata for it (``_derive_entry`` builds a decent form from the
    tool's own props);
  - adding a curated node is a dict entry, not a code change anywhere else.

Studio contract
---------------
``node_catalog()`` returns::

    {
      "categories": [{"id": "flow", "label": "Flow", "color": "#c651a8", ...}],
      "nodes": {"if_condition": {...}, ...},
      "groups": {"flow": [ {...}, ... ]},
      "count": 87
    }

Each node entry::

    {
      "name": "if_condition",
      "label": "IF",
      "category": "flow",
      "icon": "⑂",
      "summary": "Branch on a condition.",
      "needs_approval": false,
      "graph": true, "react": false,
      "outputs": ["true", "false"],          # gate values for run_if
      "defaults": {...},                     # args for a freshly dropped node
      "params": [
        {"name": "conditions_json", "label": "Conditions", "type": "json",
         "default": "...", "help": "...", "required": false,
         "options": [...]}                   # only for type == "select"
      ]
    }

Param ``type`` is one of: string | textarea | number | boolean | json | code | select.
"""

from __future__ import annotations

from typing import Any

from system.log import get_logger

log = get_logger(__name__)


# ── categories ───────────────────────────────────────────────────────────────

CATEGORIES: dict[str, dict[str, Any]] = {
    "trigger":   {"label": "Triggers",      "color": "#51d4c8", "order": 0},
    "flow":      {"label": "Flow control",  "color": "#c651a8", "order": 1},
    "transform": {"label": "Transform",     "color": "#a888e8", "order": 2},
    "io":        {"label": "HTTP & files",  "color": "#e8a13a", "order": 3},
    "ai":        {"label": "AI & reports",  "color": "#d8bcff", "order": 4},
    "research":  {"label": "Research",      "color": "#7fb3ff", "order": 5},
    "kb":        {"label": "Knowledge",     "color": "#8fd694", "order": 6},
    "workflow":  {"label": "Shared pipeline", "color": "#6fd0c0", "order": 7},
    "it_ops":    {"label": "IT ops",        "color": "#ff9d7a", "order": 8},
    "codebase":  {"label": "Codebase",      "color": "#9ecbff", "order": 9},
    "scheduling": {"label": "Scheduling",   "color": "#ffd479", "order": 10},
    "social":    {"label": "Social & email", "color": "#ff7ab6", "order": 11},
    "jobs":      {"label": "Job hunt",      "color": "#ffc46b", "order": 12},
    "cache":     {"label": "Cache",         "color": "#9aa7b8", "order": 13},
    "everyday":  {"label": "Everyday",      "color": "#b9e3a8", "order": 14},
    "multi_agent": {"label": "Multi-agent", "color": "#cf9bff", "order": 15},
    "note":      {"label": "Annotation",    "color": "#e8d06a", "order": 16},
    "other":     {"label": "Other tools",   "color": "#6b5f85", "order": 99},
}

# registry ToolSpec.domain -> studio category
_DOMAIN_TO_CATEGORY = {
    "flow": "flow",
    "transform": "transform",
    "io": "io",
    "note": "note",
    "research": "research",
    "reports": "ai",
    "kb": "kb",
    "skills": "kb",
    "workflow": "workflow",
    "it_ops": "it_ops",
    "coding": "codebase",
    "codebase": "codebase",
    "self_improve": "codebase",
    "repo": "codebase",
    "scheduling": "scheduling",
    "social": "social",
    "email": "social",
    "photo": "social",
    "jobs": "jobs",
    "job_hunt": "jobs",
    "cache": "cache",
    "everyday": "everyday",
    "multi_agent": "multi_agent",
    "graph": "flow",
    "weather": "research",
}


def _p(name: str, label: str, ptype: str = "string", default: Any = "",
       help_text: str = "", options: list[str] | None = None,
       required: bool = False, advanced: bool = False) -> dict[str, Any]:
    entry = {
        "name": name, "label": label, "type": ptype,
        "default": default, "help": help_text,
        "required": required, "advanced": advanced,
    }
    if options:
        entry["options"] = options
    return entry


# Parameters shared by every flow node, appended automatically.
_STREAM_PARAMS = [
    _p("items_json", "Items (upstream)", "string", "",
       "Usually $result:<node_id>. Leave empty when using a state key instead.", advanced=True),
    _p("from_state", "Read state key", "string", "",
       "Read the item list from GraphState instead of $result — required for anything large.", advanced=True),
    _p("to_state", "Write state key", "string", "items",
       "Where the full (untruncated) output list is stored for the next node. Leave empty to write nothing.", advanced=True),
]


def _with_stream(params: list[dict], to_state_default: str = "items") -> list[dict]:
    tail = []
    for entry in _STREAM_PARAMS:
        copy = dict(entry)
        if copy["name"] == "to_state":
            copy["default"] = to_state_default
        tail.append(copy)
    return params + tail


# ── curated nodes ────────────────────────────────────────────────────────────

NODE_SPECS: dict[str, dict[str, Any]] = {
    # ── triggers ────────────────────────────────────────────────────────────
    "trigger_manual": {
        "label": "Manual Trigger", "category": "trigger", "icon": "▶",
        "summary": "Start the workflow with literal or pinned items.",
        "params": _with_stream([
            _p("items_json", "Test items (JSON)", "json", "[{}]",
               "Paste the data this workflow should start from."),
        ]),
        "defaults": {"items_json": "[{}]", "to_state": "items"},
    },
    "sticky_note": {
        "label": "Sticky Note", "category": "note", "icon": "🗒",
        "summary": "Documentation on the canvas. Never runs.",
        "params": [_p("content", "Note", "textarea", "", "Markdown-ish free text.")],
        "defaults": {"content": "What this part of the workflow does…"},
        "decorative": True,
    },

    # ── flow control ────────────────────────────────────────────────────────
    "if_condition": {
        "label": "IF", "category": "flow", "icon": "⑂",
        "summary": "Branch: returns true/false for downstream run_if gates.",
        "outputs": ["true", "false"],
        "params": _with_stream([
            _p("conditions_json", "Conditions", "json",
               '{"combinator":"and","conditions":[{"field":"status","op":"eq","value":"open"}]}',
               "field + op + value. Ops: eq ne contains starts_with ends_with regex gt gte lt lte in is_empty exists."),
            _p("combinator", "Combine with", "select", "and", options=["and", "or"]),
            _p("mode", "Roll up", "select", "any",
               "any = true when at least one item matches; all = every item; first = the first item.",
               options=["any", "all", "first"]),
            _p("true_state", "Matched → state key", "string", "items_true",
               "Branches gated on 'true' should read this key."),
            _p("false_state", "Unmatched → state key", "string", "items_false",
               "Branches gated on 'false' should read this key."),
        ], to_state_default=""),
        "defaults": {
            "conditions_json": '{"combinator":"and","conditions":[{"field":"status","op":"eq","value":"open"}]}',
            "mode": "any", "true_state": "items_true", "false_state": "items_false",
        },
    },
    "switch_route": {
        "label": "Switch", "category": "flow", "icon": "⇄",
        "summary": "Route to the first matching branch name.",
        "outputs": ["<route>", "default"],
        "params": _with_stream([
            _p("rules_json", "Routes", "json",
               '[{"route":"urgent","field":"priority","op":"gte","value":8},'
               '{"route":"normal","field":"priority","op":"lt","value":8}]',
               "Ordered rules; first match wins."),
            _p("fallback", "Fallback route", "string", "default"),
        ]),
        "defaults": {
            "rules_json": '[{"route":"urgent","field":"priority","op":"gte","value":8}]',
            "fallback": "default",
        },
    },
    "filter_items": {
        "label": "Filter", "category": "flow", "icon": "⌗",
        "summary": "Keep only items that match.",
        "params": _with_stream([
            _p("conditions_json", "Conditions", "json",
               '{"combinator":"and","conditions":[{"field":"title","op":"contains","value":"engineer"}]}'),
            _p("combinator", "Combine with", "select", "and", options=["and", "or"]),
            _p("invert", "Invert (keep non-matching)", "boolean", False),
        ]),
        "defaults": {"conditions_json": '{"combinator":"and","conditions":[]}'},
    },
    "merge_items": {
        "label": "Merge", "category": "flow", "icon": "⋈",
        "summary": "Join several branches back into one stream.",
        "params": [
            _p("mode", "Mode", "select", "append",
               "append stacks the branches; combine_by_key joins on a shared id; "
               "combine_by_position zips them; pick_first_non_empty takes branch 1.",
               options=["append", "combine_by_key", "combine_by_position", "pick_first_non_empty"]),
            _p("key", "Join key", "string", "id", "Used by combine_by_key."),
            _p("from_states", "Branch state keys", "string", "",
               "Comma-separated GraphState keys, e.g. items_true,items_false."),
            _p("input_1", "Branch 1 ($result)", "string", "", advanced=True),
            _p("input_2", "Branch 2 ($result)", "string", "", advanced=True),
            _p("input_3", "Branch 3 ($result)", "string", "", advanced=True),
            _p("input_4", "Branch 4 ($result)", "string", "", advanced=True),
            _p("to_state", "Write state key", "string", "items", advanced=True),
        ],
        "defaults": {"mode": "append", "from_states": "items_true,items_false", "to_state": "items"},
    },
    "split_in_batches": {
        "label": "Split In Batches", "category": "flow", "icon": "⧉",
        "summary": "Walk a long list a chunk at a time (loop node).",
        "outputs": ["done", "next"],
        "params": [
            _p("batch_size", "Batch size", "number", 10),
            _p("from_state", "Read state key", "string", "items"),
            _p("to_state", "Batch state key", "string", "batch"),
            _p("assignments_json", "Per-batch transform", "json", "{}",
               "Set-fields syntax, applied to each chunk as it passes. "
               "Only the loop node runs per pass, so the transform must live here, not downstream."),
            _p("accumulate_to", "Accumulate every pass into", "string", "",
               "State key collecting all transformed batches, e.g. all_items. "
               "A terminal Aggregate node should read this key — without it, downstream only ever sees the last batch."),
            _p("items_json", "Items (upstream)", "string", "", advanced=True),
            _p("reset", "Reset cursor", "boolean", False, advanced=True),
        ],
        "defaults": {"batch_size": 10, "from_state": "items", "to_state": "batch"},
        "loop_hint": {
            "loop_to": "self",
            "loop_condition": {"not": {"contains": '"done": true'}},
            "max_visits": 50,
        },
    },
    "wait_delay": {
        "label": "Wait", "category": "flow", "icon": "⏱",
        "summary": "Pause this branch (bounded).",
        "params": _with_stream([_p("seconds", "Seconds", "number", 1)]),
        "defaults": {"seconds": 1},
    },
    "no_op": {
        "label": "No Op", "category": "flow", "icon": "•",
        "summary": "Pass through. Useful as a join point or placeholder.",
        "params": _with_stream([]),
        "defaults": {},
    },
    "stop_and_error": {
        "label": "Stop and Error", "category": "flow", "icon": "⛔",
        "summary": "Fail deliberately when a precondition is not met.",
        "params": [
            _p("message", "Error message", "string", "workflow stopped"),
            _p("conditions_json", "Only when", "json", "",
               "Leave empty to always fail. Otherwise the same condition shape as IF."),
            _p("combinator", "Combine with", "select", "and", options=["and", "or"]),
            _p("from_state", "Read state key", "string", "items", advanced=True),
            _p("items_json", "Items (upstream)", "string", "", advanced=True),
        ],
        "defaults": {"message": "precondition failed"},
    },

    # ── transform ───────────────────────────────────────────────────────────
    "set_fields": {
        "label": "Set Fields", "category": "transform", "icon": "✎",
        "summary": "Add or overwrite fields on every item.",
        "params": _with_stream([
            _p("assignments_json", "Assignments", "json",
               '{"label":"{{ title }}","doubled":"=price * 2"}',
               '"{{ expr }}" renders a template; a leading "=" evaluates an expression.'),
            _p("keep_only_set", "Keep only these fields", "boolean", False),
        ]),
        "defaults": {"assignments_json": '{"label":"{{ title }}"}'},
    },
    "rename_fields": {
        "label": "Rename Fields", "category": "transform", "icon": "🏷",
        "summary": "Rename or drop fields.",
        "params": _with_stream([
            _p("mapping_json", "Rename map", "json", '{"old_name":"new_name"}'),
            _p("drop", "Drop fields", "string", "", "Comma-separated field names."),
        ]),
        "defaults": {"mapping_json": "{}"},
    },
    "code_transform": {
        "label": "Code", "category": "transform", "icon": "{ }",
        "summary": "Run a sandboxed expression over the items.",
        "params": _with_stream([
            _p("expression", "Expression", "code",
               "{'title': item['title'], 'len': len(item.get('text',''))}",
               "Restricted Python expression. Names: item/json, index, items, state, now."),
            _p("mode", "Run", "select", "each",
               "each = once per item; all = once over the whole list.",
               options=["each", "all"]),
            _p("to_field", "Write to field", "string", "",
               "Leave empty to replace the whole item with the result."),
        ]),
        "defaults": {"expression": "item", "mode": "each"},
    },
    "template_render": {
        "label": "Template", "category": "transform", "icon": "≋",
        "summary": "Render a text template into a field.",
        "params": _with_stream([
            _p("template", "Template", "textarea", "{{ title }} — {{ url }}"),
            _p("to_field", "Write to field", "string", "text"),
        ]),
        "defaults": {"template": "{{ title }}", "to_field": "text"},
    },
    "sort_items": {
        "label": "Sort", "category": "transform", "icon": "↕",
        "summary": "Order items by a field.",
        "params": _with_stream([
            _p("field", "Field", "string", "", "Dotted path, e.g. meta.score."),
            _p("order", "Order", "select", "asc", options=["asc", "desc"]),
        ]),
        "defaults": {"field": "", "order": "desc"},
    },
    "limit_items": {
        "label": "Limit", "category": "transform", "icon": "⊤",
        "summary": "Keep the first or last N items.",
        "params": _with_stream([
            _p("max_items", "Max items", "number", 10),
            _p("keep", "Keep", "select", "first", options=["first", "last"]),
        ]),
        "defaults": {"max_items": 10, "keep": "first"},
    },
    "remove_duplicates": {
        "label": "Remove Duplicates", "category": "transform", "icon": "⧄",
        "summary": "Drop repeats by field or whole item.",
        "params": _with_stream([
            _p("field", "Dedupe on field", "string", "url",
               "Leave empty to hash the entire item."),
        ]),
        "defaults": {"field": "url"},
    },
    "aggregate_items": {
        "label": "Aggregate", "category": "transform", "icon": "Σ",
        "summary": "Count, sum, average, concat or collect.",
        "params": _with_stream([
            _p("mode", "Mode", "select", "count",
               options=["count", "sum", "avg", "min", "max", "concat", "collect"]),
            _p("field", "Field", "string", "", "The value being aggregated."),
            _p("group_by", "Group by", "string", "", "Optional field to group on."),
            _p("to_field", "Write to field", "string", "value"),
            _p("separator", "Concat separator", "string", "\n", advanced=True),
        ]),
        "defaults": {"mode": "count", "to_field": "value"},
    },
    "split_out": {
        "label": "Split Out", "category": "transform", "icon": "⤳",
        "summary": "Explode a list field into separate items.",
        "params": _with_stream([
            _p("field", "List field", "string", "items"),
            _p("keep_parent_fields", "Keep parent fields", "boolean", False),
        ]),
        "defaults": {"field": "items"},
    },
    "items_to_text": {
        "label": "Items → Text", "category": "transform", "icon": "¶",
        "summary": "Flatten items into one prose block for an AI node.",
        "params": [
            _p("template", "Per-item template", "textarea", "{{ title }}\n{{ text }}"),
            _p("separator", "Separator", "string", "\n\n---\n\n"),
            _p("max_items", "Max items", "number", 25),
            _p("max_chars", "Max characters", "number", 8000),
            _p("from_state", "Read state key", "string", "items", advanced=True),
            _p("items_json", "Items (upstream)", "string", "", advanced=True),
            _p("to_state", "Write state key", "string", "text", advanced=True),
        ],
        "defaults": {"template": "{{ title }}\n{{ text }}", "from_state": "items", "to_state": "text"},
    },

    # ── I/O ─────────────────────────────────────────────────────────────────
    "http_request": {
        "label": "HTTP Request", "category": "io", "icon": "🌐",
        "summary": "Call a public JSON or text API.",
        "params": _with_stream([
            _p("url", "URL", "string", "https://api.example.com/v1/items",
               "Supports {{ templates }} from the first incoming item.", required=True),
            _p("method", "Method", "select", "GET",
               options=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]),
            _p("query_json", "Query params", "json", "{}"),
            _p("headers_json", "Headers", "json", "{}"),
            _p("body_json", "Body", "json", "", "JSON body for POST/PUT/PATCH."),
            _p("response_path", "Response path", "string", "",
               "Dotted path into the response, e.g. data.results."),
            _p("timeout", "Timeout (s)", "number", 15, advanced=True),
            _p("max_items", "Max items", "number", 50, advanced=True),
        ]),
        "defaults": {"method": "GET", "url": "https://", "max_items": 50},
    },
}

# Nodes worth pinning to the top of the palette — the ones a new workflow
# almost always starts from.
FEATURED = (
    "trigger_manual", "http_request", "if_condition", "switch_route",
    "filter_items", "set_fields", "merge_items", "split_in_batches",
    "aggregate_items", "items_to_text", "synthesize_report", "write_report",
)


# ── derivation for uncurated registry tools ──────────────────────────────────

_TYPE_MAP = {
    "string": "string", "integer": "number", "number": "number",
    "boolean": "boolean", "array": "json", "object": "json",
}


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _params_from_props(props: dict[str, Any] | None, required: list[str] | None) -> list[dict]:
    params: list[dict] = []
    required = set(required or [])
    for key, meta in (props or {}).items():
        meta = meta if isinstance(meta, dict) else {}
        ptype = _TYPE_MAP.get(str(meta.get("type") or "string"), "string")
        if meta.get("enum"):
            ptype = "select"
        description = str(meta.get("description") or "")
        if ptype == "string" and len(description) > 90:
            ptype = "textarea"
        params.append(_p(
            key, _humanize(key), ptype,
            meta.get("default", "" if ptype != "boolean" else False),
            description,
            options=[str(o) for o in (meta.get("enum") or [])] or None,
            required=key in required,
        ))
    return params


def _params_from_model(model: Any) -> list[dict]:
    try:
        schema = model.model_json_schema()
    except Exception:
        return []
    return _params_from_props(schema.get("properties"), schema.get("required"))


def _derive_entry(spec: Any) -> dict[str, Any]:
    """Build a usable palette entry for a tool nobody curated."""
    category = _DOMAIN_TO_CATEGORY.get(str(spec.domain or ""), "other")
    params = _params_from_model(spec.args_model) if spec.args_model else []
    if not params:
        params = _params_from_props(spec.props, spec.required)
    return {
        "name": spec.name,
        "label": _humanize(spec.name),
        "category": category,
        "icon": "◆",
        "summary": (spec.description or "")[:180],
        "params": params,
        "defaults": {p["name"]: p["default"] for p in params if p.get("required")},
        "curated": False,
    }


# ── public API ───────────────────────────────────────────────────────────────

def node_entry(name: str, spec: Any = None) -> dict[str, Any]:
    """Return the palette entry for one tool (curated first, derived second)."""
    curated = NODE_SPECS.get(name)
    if curated is None:
        if spec is None:
            return {"name": name, "label": _humanize(name), "category": "other",
                    "icon": "◆", "summary": "", "params": [], "defaults": {}, "curated": False}
        return _derive_entry(spec)
    entry = {
        "name": name,
        "label": curated.get("label") or _humanize(name),
        "category": curated.get("category") or "other",
        "icon": curated.get("icon") or "◆",
        "summary": curated.get("summary") or (getattr(spec, "description", "") or "")[:180],
        "params": [dict(p) for p in curated.get("params") or []],
        "defaults": dict(curated.get("defaults") or {}),
        "curated": True,
    }
    for optional in ("outputs", "loop_hint", "decorative"):
        if optional in curated:
            entry[optional] = curated[optional]
    if spec is not None:
        entry["needs_approval"] = bool(getattr(spec, "needs_approval", False))
        entry["graph"] = bool(getattr(spec, "graph", True))
        entry["react"] = bool(getattr(spec, "react", False))
    return entry


def default_args(name: str) -> dict[str, Any]:
    """Args for a freshly dropped node — so it is runnable, not blank."""
    curated = NODE_SPECS.get(name)
    if curated:
        defaults = dict(curated.get("defaults") or {})
        for param in curated.get("params") or []:
            defaults.setdefault(param["name"], param.get("default"))
        return {k: v for k, v in defaults.items() if v not in (None, "")}
    return {}


def node_catalog(include_react_only: bool = True) -> dict[str, Any]:
    """Full catalog for the studio: categories + per-node forms + groups."""
    try:
        import agentic.tools  # noqa: F401 — populates the registry
    except Exception as exc:  # pragma: no cover - optional lanes may be missing
        log.warning("node_catalog: agentic.tools import failed: %s", exc)
    try:
        from agentic.toolkit import flow  # noqa: F401 — core flow nodes
    except Exception as exc:
        log.warning("node_catalog: flow nodes unavailable: %s", exc)
    try:
        import agentic.workflows.common.nodes  # noqa: F401 — shared Layer-1 nodes
    except Exception as exc:
        log.debug("node_catalog: shared workflow nodes unavailable: %s", exc)

    from agentic.registry import registry

    nodes: dict[str, dict[str, Any]] = {}
    for spec in sorted(registry.all_specs(), key=lambda s: s.name):
        if not spec.graph and not (include_react_only and spec.react):
            continue
        nodes[spec.name] = node_entry(spec.name, spec)

    # Curated entries whose handler failed to import still belong in the
    # palette — the studio shows them greyed out rather than silently missing.
    for name in NODE_SPECS:
        nodes.setdefault(name, node_entry(name, None) | {"unavailable": True})

    groups: dict[str, list[dict]] = {}
    for entry in nodes.values():
        groups.setdefault(entry["category"], []).append(entry)
    for bucket in groups.values():
        bucket.sort(key=lambda e: (not e.get("curated"), e["label"]))

    categories = [
        {"id": cid, **meta}
        for cid, meta in sorted(CATEGORIES.items(), key=lambda kv: kv[1]["order"])
        if cid in groups
    ]
    return {
        "categories": categories,
        "groups": groups,
        "nodes": nodes,
        "featured": [n for n in FEATURED if n in nodes],
        "count": len(nodes),
    }


__all__ = ["CATEGORIES", "NODE_SPECS", "FEATURED", "node_catalog", "node_entry", "default_args"]
