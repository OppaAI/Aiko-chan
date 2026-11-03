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
    "trigger": "trigger",
    "ai": "ai",
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

    # ── studio phase 2: triggers ────────────────────────────────────────────
    "trigger_chat": {
        "label": "Chat Trigger", "subtitle": "Chat message", "category": "trigger", "icon": "💬",
        "summary": "Start the workflow from a chat message (n8n Chat Trigger).",
        "params": [
            _p("message", "Message", "textarea", "$prompt",
               "The incoming message. $prompt is replaced by the run prompt."),
            _p("from_user", "From", "string", "user"),
            _p("to_state", "Write state key", "string", "items", advanced=True),
        ],
        "defaults": {"message": "$prompt", "from_user": "user", "to_state": "items"},
    },
    "trigger_webhook": {
        "label": "Webhook Trigger", "subtitle": "Webhook", "category": "trigger", "icon": "🔗",
        "summary": "Start the workflow from an HTTP call. Emits the test payload.",
        "params": [
            _p("path", "Path", "string", "/webhook/studio"),
            _p("method", "Method", "select", "POST", options=["GET", "POST", "PUT", "PATCH", "DELETE"]),
            _p("payload_json", "Test payload (JSON)", "json", "[{}]",
               "Items emitted when the workflow runs from the studio."),
            _p("to_state", "Write state key", "string", "items", advanced=True),
        ],
        "defaults": {"path": "/webhook/studio", "method": "POST", "payload_json": "[{}]"},
    },
    "trigger_schedule": {
        "label": "Schedule Trigger", "subtitle": "Cron", "category": "trigger", "icon": "⏰",
        "summary": "Start the workflow on a cron timetable. Emits one timestamped item.",
        "params": [
            _p("cron", "Cron expression", "string", "0 9 * * *",
               "Minute hour day month weekday, e.g. 0 9 * * * = daily 09:00.", required=True),
            _p("timezone", "Timezone", "string", "America/Los_Angeles"),
            _p("to_state", "Write state key", "string", "items", advanced=True),
        ],
        "defaults": {"cron": "0 9 * * *", "timezone": "America/Los_Angeles"},
    },

    # ── studio phase 2: agentic ─────────────────────────────────────────────
    "ai_agent": {
        "label": "AI Agent", "subtitle": "Tools Agent", "category": "ai", "icon": "🤖",
        "summary": "Composite agent: runs the chat model over your prompt with memory and tools (n8n Tools Agent).",
        "composite": True,
        "params": _with_stream([
            _p("prompt", "Prompt", "textarea", "$prompt",
               "What to ask. $prompt is replaced by the run prompt.", required=True),
            _p("system_prompt", "System prompt", "textarea", "",
               "Persona and instructions for the agent."),
            _p("chat_model_json", "Chat model config", "string", "$result:chat_model",
               "Usually $result:<chat_model node id> — wire the Chat Model sub-node here."),
            _p("tools_json", "Tools (JSON list)", "json", "[]",
               "Tool names the agent may call, e.g. [\"calendar_create\"]."),
            _p("memory_key", "Memory key", "string", "agent_memory",
               "GraphState key holding the conversation window."),
            _p("memory_window", "Memory window", "number", 10, advanced=True),
            _p("temperature", "Temperature", "number", 0.7, advanced=True),
            _p("max_tokens", "Max tokens", "number", 1024, advanced=True),
        ]),
        "defaults": {"prompt": "$prompt", "chat_model_json": "$result:chat_model",
                     "tools_json": "[]", "memory_key": "agent_memory"},
    },
    "chat_model": {
        "label": "Chat Model", "subtitle": "Model", "category": "ai", "icon": "🧠",
        "summary": "Sub-node for the AI Agent: model, temperature, max tokens.",
        "sub_node": True, "sub_kind": "chat_model",
        "params": [
            _p("model", "Model", "select", "ministral-3b",
               options=["ministral-3b", "qwen2.5-7b", "llama3.1-8b", "gpt-4o-mini"]),
            _p("temperature", "Temperature", "number", 0.7),
            _p("max_tokens", "Max tokens", "number", 1024),
            _p("to_state", "Write state key", "string", "items", advanced=True),
        ],
        "defaults": {"model": "ministral-3b", "temperature": 0.7, "max_tokens": 1024},
    },
    "memory_buffer": {
        "label": "Memory", "subtitle": "Buffer", "category": "ai", "icon": "💾",
        "summary": "Sub-node for the AI Agent: sliding-window buffer over a state key.",
        "sub_node": True, "sub_kind": "memory",
        "params": _with_stream([
            _p("memory_key", "Memory key", "string", "agent_memory", required=True),
            _p("window_size", "Window size", "number", 10),
            _p("mode", "Mode", "select", "read_write",
               options=["read_write", "write", "read"]),
        ]),
        "defaults": {"memory_key": "agent_memory", "window_size": 10, "mode": "read_write"},
    },

    # ── studio phase 2: email ───────────────────────────────────────────────
    "email_check": {
        "label": "Email: Check Inbox", "subtitle": "Read email", "category": "social", "icon": "📥",
        "summary": "Poll the owner's inbox via Aiko's email bridge. One item per new message.",
        "params": _with_stream([
            _p("max_results", "Max messages", "number", 5),
        ]),
        "defaults": {"max_results": 5},
    },
    "email_reply": {
        "label": "Email: Reply", "subtitle": "Send reply", "category": "social", "icon": "📧",
        "summary": "Generate replies with the LLM and send them via the email bridge.",
        "params": _with_stream([
            _p("report_json", "Batch (JSON)", "string", "$result:email_check",
               "Usually $result:<email_check node id>, or read from state."),
        ]),
        "defaults": {"report_json": "$result:email_check"},
    },
    "email_draft": {
        "label": "Email: Draft", "subtitle": "Compose draft", "category": "social", "icon": "✉️",
        "summary": "Compose a draft and park it under a state key. Nothing is sent.",
        "params": _with_stream([
            _p("to", "To", "string", ""),
            _p("subject", "Subject", "string", ""),
            _p("body", "Body", "textarea", "$prompt"),
            _p("drafts_key", "Drafts state key", "string", "email_drafts", advanced=True),
        ]),
        "defaults": {"body": "$prompt", "drafts_key": "email_drafts"},
    },
    "gmail_search": {
        "label": "Gmail: Search", "subtitle": "Not connected", "category": "social", "icon": "🔍",
        "summary": "Search Gmail messages. No Gmail adapter configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("query", "Query", "string", "is:unread", "Gmail search query."),
            _p("max_results", "Max messages", "number", 5),
        ]),
        "defaults": {"query": "is:unread", "max_results": 5},
    },
    "gmail_send": {
        "label": "Gmail: Send", "subtitle": "Not connected", "category": "social", "icon": "📤",
        "summary": "Send a Gmail message. No Gmail adapter configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("to", "To", "string", "", required=True),
            _p("subject", "Subject", "string", ""),
            _p("body", "Body", "textarea", "$prompt", required=True),
        ]),
        "defaults": {"body": "$prompt"},
    },
    "gmail_draft": {
        "label": "Gmail: Draft", "subtitle": "Compose draft", "category": "social", "icon": "📝",
        "summary": "Compose a Gmail draft and park it under a state key. Nothing is sent.",
        "params": _with_stream([
            _p("to", "To", "string", ""),
            _p("subject", "Subject", "string", ""),
            _p("body", "Body", "textarea", "$prompt"),
            _p("drafts_key", "Drafts state key", "string", "gmail_drafts", advanced=True),
        ]),
        "defaults": {"body": "$prompt", "drafts_key": "gmail_drafts"},
    },

    # ── studio phase 2: social ──────────────────────────────────────────────
    "threads_post": {
        "label": "Threads: Post", "subtitle": "Post", "category": "social", "icon": "@",
        "summary": "Post text (and an optional image) to Threads via the social bridge.",
        "params": _with_stream([
            _p("text", "Text", "textarea", "$prompt", required=True),
            _p("image_path", "Image path", "string", "", "Workspace-relative image, optional."),
        ]),
        "defaults": {"text": "$prompt"},
    },
    "instagram_post": {
        "label": "Instagram: Post", "subtitle": "Not connected", "category": "social", "icon": "📸",
        "summary": "Post to Instagram. No backend configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("caption", "Caption", "textarea", "$prompt"),
            _p("image_path", "Image path", "string", ""),
        ]),
        "defaults": {"caption": "$prompt"},
    },
    "messenger_send": {
        "label": "Messenger: Send", "subtitle": "Not connected", "category": "social", "icon": "💭",
        "summary": "Send a Messenger message. No backend configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("recipient", "Recipient", "string", ""),
            _p("message", "Message", "textarea", "$prompt", required=True),
        ]),
        "defaults": {"message": "$prompt"},
    },
    "social_read": {
        "label": "Social: Read", "subtitle": "Not connected", "category": "social", "icon": "📡",
        "summary": "Read recent social mentions. No read backend configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("services", "Services", "string", "threads", "Comma-separated, e.g. threads,mastodon."),
            _p("max_results", "Max results", "number", 10),
        ]),
        "defaults": {"services": "threads", "max_results": 10},
    },
    "instagram_read": {
        "label": "Instagram: Read", "subtitle": "Not connected", "category": "social", "icon": "📷",
        "summary": "Read recent Instagram posts or mentions. No backend configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("max_results", "Max results", "number", 10),
        ]),
        "defaults": {"max_results": 10},
    },
    "threads_read": {
        "label": "Threads: Read", "subtitle": "Not connected", "category": "social", "icon": "🧵",
        "summary": "Read recent Threads posts or mentions. No backend configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("max_results", "Max results", "number", 10),
        ]),
        "defaults": {"max_results": 10},
    },
    "messenger_read": {
        "label": "Messenger: Read", "subtitle": "Not connected", "category": "social", "icon": "💬",
        "summary": "Read recent Messenger messages. No backend configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("max_results", "Max results", "number", 10),
        ]),
        "defaults": {"max_results": 10},
    },
    "whatsapp_send": {
        "label": "WhatsApp: Send", "subtitle": "Not connected", "category": "social", "icon": "📱",
        "summary": "Send a WhatsApp message. No backend configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("recipient", "Recipient", "string", "", "Phone number or chat id."),
            _p("message", "Message", "textarea", "$prompt", required=True),
        ]),
        "defaults": {"message": "$prompt"},
    },
    "telegram_send": {
        "label": "Telegram: Send", "subtitle": "Not connected", "category": "social", "icon": "✈️",
        "summary": "Send a Telegram message. No backend configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("chat_id", "Chat id", "string", ""),
            _p("message", "Message", "textarea", "$prompt", required=True),
        ]),
        "defaults": {"message": "$prompt"},
    },

    # ── studio phase 2: utilities ───────────────────────────────────────────
    "calendar_create": {
        "label": "Calendar: Create Event", "subtitle": "Not connected", "category": "scheduling", "icon": "📅",
        "summary": "Create a Google Calendar event. No backend configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("title", "Title", "string", "$prompt", required=True),
            _p("start", "Start", "string", "", "ISO datetime, e.g. 2026-09-24T09:00:00."),
            _p("end", "End", "string", "", "ISO datetime."),
            _p("description", "Description", "textarea", ""),
        ]),
        "defaults": {"title": "$prompt"},
    },
    "calendar_list": {
        "label": "Calendar: List Events", "subtitle": "Not connected", "category": "scheduling", "icon": "🗓",
        "summary": "List upcoming Google Calendar events. No backend configured — returns not-connected gracefully.",
        "params": _with_stream([
            _p("max_results", "Max events", "number", 10),
            _p("time_min", "From", "string", "", "ISO datetime; empty = now."),
        ]),
        "defaults": {"max_results": 10},
    },
    "reminder": {
        "label": "Reminder", "subtitle": "Schedule", "category": "scheduling", "icon": "⏰",
        "summary": "Schedule a local reminder (title + message at a time of day, optionally repeating).",
        "params": _with_stream([
            _p("title", "Title", "string", "$prompt", required=True),
            _p("message", "Message", "textarea", ""),
            _p("time_of_day", "Time of day", "string", "09:00", "HH:MM, 24h."),
            _p("repeat", "Repeat", "select", "once", options=["once", "daily", "weekly"]),
        ]),
        "defaults": {"title": "$prompt", "time_of_day": "09:00", "repeat": "once"},
    },
    "tool_router": {
        "label": "Tool Router", "subtitle": "Route", "category": "flow", "icon": "🔀",
        "summary": "Send items down the branch whose keyword (or /regex/) matches the input text. First match wins; otherwise the default branch.",
        "params": _with_stream([
            _p("input_text", "Input text", "textarea", "$prompt", "Text to match routes against."),
            _p("routes_json", "Routes (JSON)", "textarea",
               '[{"name": "social", "match": "threads"}, {"name": "email", "match": "email"}]',
               'List of {"name", "match"}; match is a keyword or /regex/.'),
            _p("default_route", "Default route", "string", "default"),
            _p("case_sensitive", "Case sensitive", "boolean", False, advanced=True),
        ]),
        "defaults": {
            "input_text": "$prompt",
            "routes_json": '[{"name": "social", "match": "threads"}, {"name": "email", "match": "email"}]',
            "default_route": "default",
        },
    },
    "rss_read": {
        "label": "RSS Reader", "subtitle": "Feed", "category": "research", "icon": "📰",
        "summary": "Fetch an RSS/Atom feed (stdlib only). One item per entry.",
        "params": _with_stream([
            _p("url", "Feed URL", "string", "https://feeds.bbci.co.uk/news/rss.xml", required=True),
            _p("max_items", "Max entries", "number", 20),
        ]),
        "defaults": {"url": "https://feeds.bbci.co.uk/news/rss.xml", "max_items": 20},
    },
    "file_write": {
        "label": "File: Write", "subtitle": "Write file", "category": "io", "icon": "📝",
        "summary": "Write or append a file under Aiko's workspace (jailed, no traversal).",
        "params": _with_stream([
            _p("path", "Path", "string", "notes/draft.txt",
               "Workspace-relative. Parent folders are created.", required=True),
            _p("content", "Content", "textarea", "$prompt"),
            _p("mode", "Mode", "select", "write", options=["write", "append"]),
        ]),
        "defaults": {"path": "notes/draft.txt", "content": "$prompt", "mode": "write"},
    },
    "file_read": {
        "label": "File: Read", "subtitle": "Read file", "category": "io", "icon": "📖",
        "summary": "Read a file under Aiko's workspace (jailed, no traversal). Pairs with File: Write.",
        "params": _with_stream([
            _p("path", "Path", "string", "notes/draft.txt", "Workspace-relative.", required=True),
            _p("max_chars", "Max chars", "number", 20000, advanced=True),
        ]),
        "defaults": {"path": "notes/draft.txt", "max_chars": 20000},
    },
    "notify_user": {
        "label": "Notify", "subtitle": "Alert owner", "category": "everyday", "icon": "🔔",
        "summary": "Surface a notification to the owner. Best-effort email delivery; the item is always emitted.",
        "params": _with_stream([
            _p("title", "Title", "string", "Aiko notification"),
            _p("message", "Message", "textarea", "$prompt", required=True),
            _p("channel", "Channel", "select", "app", options=["app", "email"]),
        ]),
        "defaults": {"title": "Aiko notification", "message": "$prompt", "channel": "app"},
    },
}

# Nodes worth pinning to the top of the palette — the ones a new workflow
# almost always starts from.
FEATURED = (
    "trigger_manual", "trigger_chat", "trigger_schedule", "ai_agent", "chat_model",
    "memory_buffer", "compose_response", "email_check", "email_reply",
    "http_request", "if_condition", "switch_route", "tool_router", "filter_items",
    "set_fields", "merge_items", "rss_read", "notify_user", "threads_post",
    "gmail_search", "reminder", "file_read", "file_write",
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
    for optional in ("outputs", "loop_hint", "decorative", "composite",
                       "sub_node", "sub_kind", "subtitle"):
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
