"""
agentic/workflows/templates.py

Starter workflow blueprints for the DAG studio.

These are the shapes engineers actually build in n8n/Airflow/Prefect, wired
from Aiko's own nodes: fetch → shape → branch → join → summarize → deliver.
Each one is runnable the moment it is instantiated, so "New workflow from
template" gives you something that works and then you edit it — rather than
an empty canvas and a blank stare.

IMPORTANT — templates are NOT playbooks
---------------------------------------
They are deliberately kept OUT of ``graph_engine.load_playbooks()``. A
playbook can be auto-selected by ``plan_from_master`` for a user prompt; a
template must never be. Templates carry no ``triggers``, no
``semantic_triggers`` and no ``requires_any``, and the studio copies them
into the user's own playbook.json (with a fresh id) on instantiation.

Node positions are included so a freshly created workflow lands on the canvas
already laid out left-to-right instead of in a pile.
"""

from __future__ import annotations

import copy
from typing import Any

# Canvas grid — keeps generated layouts consistent with the studio's spacing.
_COL = 260
_ROW = 130
_X0 = 80
_Y0 = 120


def _pos(col: int, row: int = 0) -> dict[str, int]:
    return {"x": _X0 + col * _COL, "y": _Y0 + row * _ROW}


def _node(
    node_id: str,
    tool: str,
    args: dict[str, Any] | None = None,
    depends_on: list[str] | None = None,
    col: int = 0,
    row: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    node = {
        "id": node_id,
        "tool": tool,
        "args": args or {},
        "depends_on": depends_on or [],
        "position": _pos(col, row),
    }
    node.update(extra)
    return node


def _sticky(node_id: str, content: str, col: int, row: int) -> dict[str, Any]:
    return _node(node_id, "sticky_note", {"content": content}, [], col, row)


# ═════════════════════════════════════════════════════════════════════════════
# TEMPLATES
# ═════════════════════════════════════════════════════════════════════════════

TEMPLATES: list[dict[str, Any]] = [

    # 1 ── the empty-but-runnable starting point ─────────────────────────────
    {
        "id": "tpl_blank",
        "name": "Blank — trigger → set",
        "category": "Starter",
        "description": "Two nodes: a manual trigger with test data and a Set node. The cleanest place to start a new workflow.",
        "goal": "Starter workflow",
        "nodes": [
            _node("trigger", "trigger_manual",
                  {"items_json": '[{"name":"Aiko","value":1}]', "to_state": "items"}, [], 0, 0),
            _node("shape", "set_fields",
                  {"from_state": "items", "to_state": "items",
                   "assignments_json": '{"greeting":"hello {{ name }}","doubled":"=value * 2"}'},
                  ["trigger"], 1, 0),
        ],
    },

    # 2 ── the single most common real pipeline ──────────────────────────────
    {
        "id": "tpl_api_to_report",
        "name": "API → filter → summarize → report",
        "category": "ETL",
        "description": "Call a JSON API, shape and filter the results, rank them, hand the top rows to the LLM and save a markdown report. The classic 'fetch, clean, summarize, deliver' pipeline.",
        "goal": "Fetch an API, summarize the interesting rows, and write a report",
        "capabilities": ["research", "reports"],
        "nodes": [
            _sticky("note_1", "Point the HTTP node at any public JSON API.\nresponse_path drills into a nested list.", 0, -1),
            _node("fetch", "http_request",
                  {"url": "https://api.github.com/repos/OppaAI/Aiko-chan/issues",
                   "method": "GET", "query_json": '{"state":"open","per_page":"20"}',
                   "response_path": "", "max_items": 20, "to_state": "items"}, [], 0, 0),
            _node("shape", "set_fields",
                  {"from_state": "items", "to_state": "items",
                   "assignments_json": '{"title":"{{ title }}","url":"{{ html_url }}","text":"{{ body }}"}'},
                  ["fetch"], 1, 0),
            _node("keep", "filter_items",
                  {"from_state": "items", "to_state": "items", "combinator": "and",
                   "conditions_json": '{"combinator":"and","conditions":[{"field":"title","op":"is_not_empty"}]}'},
                  ["shape"], 2, 0),
            _node("top", "limit_items",
                  {"from_state": "items", "to_state": "items", "max_items": 10, "keep": "first"},
                  ["keep"], 3, 0),
            _node("to_text", "items_to_text",
                  {"from_state": "items", "to_state": "text", "max_items": 10, "max_chars": 3500,
                   "template": "- {{ title }}\n  {{ url }}\n  {{ text }}"},
                  ["top"], 4, 0),
            _node("draft", "synthesize_report",
                  {"evidence": "$result:to_text", "prompt": "Summarize these items and flag anything urgent.",
                   "style": "plain"}, ["to_text"], 5, 0),
            _node("report", "write_report",
                  {"title": "$title", "content": "$result:draft", "report_dir": "reports"},
                  ["draft"], 6, 0),
        ],
    },

    # 3 ── branching ─────────────────────────────────────────────────────────
    {
        "id": "tpl_branch_merge",
        "name": "IF branch → two paths → merge",
        "category": "Flow control",
        "description": "Split the stream on a condition, handle each side differently, then merge the branches back together. The canonical conditional pattern.",
        "goal": "Branch on a condition and rejoin",
        "nodes": [
            _sticky("note_1", "IF returns 'true'/'false'.\nEach branch gates with run_if on the IF node.", 1, -1),
            _node("trigger", "trigger_manual",
                  {"items_json": '[{"priority":9,"title":"disk full"},{"priority":2,"title":"typo"}]',
                   "to_state": "items"}, [], 0, 0),
            _node("check", "if_condition",
                  {"from_state": "items", "true_state": "items_true", "false_state": "items_false", "mode": "any",
                   "conditions_json": '{"combinator":"and","conditions":[{"field":"priority","op":"gte","value":5}]}'},
                  ["trigger"], 1, 0),
            _node("urgent", "set_fields",
                  {"from_state": "items_true", "to_state": "items_true",
                   "assignments_json": '{"lane":"urgent","label":"🚨 {{ title }}"}'},
                  ["check"], 2, -1, run_if={"node": "check", "equals": "true"}),
            _node("routine", "set_fields",
                  {"from_state": "items_false", "to_state": "items_false",
                   "assignments_json": '{"lane":"routine","label":"{{ title }}"}'},
                  ["check"], 2, 1, run_if={"node": "check", "equals": "false"}),
            _node("join", "merge_items",
                  {"mode": "append", "from_states": "items_true,items_false", "to_state": "items"},
                  ["urgent", "routine"], 3, 0),
        ],
    },

    # 4 ── multi-way routing ─────────────────────────────────────────────────
    {
        "id": "tpl_switch_router",
        "name": "Switch → three lanes → merge",
        "category": "Flow control",
        "description": "Route items down one of several named lanes with a Switch node, process each lane independently, then collect the results.",
        "goal": "Route work to the right lane",
        "nodes": [
            _node("trigger", "trigger_manual",
                  {"items_json": '[{"kind":"bug","title":"crash on boot"}]', "to_state": "items"}, [], 0, 0),
            _node("router", "switch_route",
                  {"from_state": "items", "to_state": "items", "fallback": "other",
                   "rules_json": '[{"route":"bug","field":"kind","op":"eq","value":"bug"},'
                                 '{"route":"feature","field":"kind","op":"eq","value":"feature"}]'},
                  ["trigger"], 1, 0),
            _node("lane_bug", "set_fields",
                  {"from_state": "items", "to_state": "lane_bug",
                   "assignments_json": '{"queue":"triage","label":"🐛 {{ title }}"}'},
                  ["router"], 2, -1, run_if={"node": "router", "equals": "bug"}),
            _node("lane_feature", "set_fields",
                  {"from_state": "items", "to_state": "lane_feature",
                   "assignments_json": '{"queue":"backlog","label":"✨ {{ title }}"}'},
                  ["router"], 2, 0, run_if={"node": "router", "equals": "feature"}),
            _node("lane_other", "set_fields",
                  {"from_state": "items", "to_state": "lane_other",
                   "assignments_json": '{"queue":"inbox","label":"{{ title }}"}'},
                  ["router"], 2, 1, run_if={"node": "router", "equals": "other"}),
            _node("join", "merge_items",
                  {"mode": "append", "from_states": "lane_bug,lane_feature,lane_other", "to_state": "items"},
                  ["lane_bug", "lane_feature", "lane_other"], 3, 0),
        ],
    },

    # 5 ── looping over a big list ───────────────────────────────────────────
    {
        "id": "tpl_batch_loop",
        "name": "Batch loop → process → aggregate",
        "category": "Flow control",
        "description": "Walk a long list in chunks so nothing blows the context or memory budget. The batch node transforms each chunk as it passes (assignments) and accumulates every pass into all_items; downstream nodes run once, after the loop, so total really sums the whole run. Built for the Jetson.",
        "goal": "Process a long list in bounded batches",
        "nodes": [
            _sticky("note_1", "The loop lives on the batch node itself.\nmax_visits caps the number of passes.\nOnly the batch node runs per pass — downstream\nruns once, after the loop — so the transform\n(assignments) and accumulation (accumulate_to)\nlive on the batch node, not downstream.", 1, -1),
            _node("trigger", "trigger_manual",
                  {"items_json": '[{"n":1},{"n":2},{"n":3},{"n":4},{"n":5}]', "to_state": "items"}, [], 0, 0),
            _node("batch", "split_in_batches",
                  {"from_state": "items", "to_state": "batch", "batch_size": 2,
                   "assignments_json": '{"n":"{{ n }}","squared":"=n * n"}',
                   "accumulate_to": "all_items"},
                  ["trigger"], 1, 0,
                  loop_to="batch",
                  loop_condition={"not": {"contains": '"done": true'}},
                  max_visits=25),
            _node("total", "aggregate_items",
                  {"from_state": "all_items", "to_state": "summary",
                   "mode": "sum", "field": "squared", "to_field": "total"},
                  ["batch"], 2, 0),
        ],
    },

    # 6 ── resilience ────────────────────────────────────────────────────────
    {
        "id": "tpl_retry_fallback",
        "name": "Retry + fallback source",
        "category": "Resilience",
        "description": "Try the primary API with retries and a timeout; if it still fails, fall back to a mirror. The pattern every production integration eventually needs.",
        "goal": "Fetch with retry and a fallback source",
        "nodes": [
            _sticky("note_1", "max_retries + retry_backoff_seconds handle flaky networks.\nfallback_to runs only when this node still fails — the mirror is\nskipped entirely on the success path, and on the failure path\nits result stands in for primary so normalize keeps working.", 0, -1),
            _node("primary", "http_request",
                  {"url": "https://api.example.com/v1/items", "method": "GET",
                   "max_items": 25, "to_state": "items", "timeout": 10}, [], 0, 0,
                  max_retries=2, retry_backoff_seconds=1.5, timeout_seconds=45,
                  fallback_to="backup"),
            _node("backup", "http_request",
                  {"url": "https://mirror.example.com/v1/items", "method": "GET",
                   "max_items": 25, "to_state": "items", "timeout": 10}, [], 0, 1),
            _node("normalize", "set_fields",
                  {"from_state": "items", "to_state": "items",
                   "assignments_json": '{"id":"{{ id }}","title":"{{ title }}"}'},
                  ["primary"], 1, 0),
            _node("dedupe", "remove_duplicates",
                  {"from_state": "items", "to_state": "items", "field": "id"},
                  ["normalize"], 2, 0),
        ],
    },

    # 7 ── parallel fan-out / fan-in ─────────────────────────────────────────
    {
        "id": "tpl_fanout_fanin",
        "name": "Parallel fan-out → fan-in",
        "category": "Performance",
        "description": "Hit three sources at once, merge the responses on a shared key, then summarize. Uses the engine's parallel scheduler — keep max_workers at 2 on a Jetson.",
        "goal": "Query several sources in parallel and combine them",
        "nodes": [
            _node("source_a", "http_request",
                  {"url": "https://api.example.com/a", "to_state": "src_a", "max_items": 20}, [], 0, -1),
            _node("source_b", "http_request",
                  {"url": "https://api.example.com/b", "to_state": "src_b", "max_items": 20}, [], 0, 0),
            _node("source_c", "http_request",
                  {"url": "https://api.example.com/c", "to_state": "src_c", "max_items": 20}, [], 0, 1),
            _node("join", "merge_items",
                  {"mode": "combine_by_key", "key": "id", "from_states": "src_a,src_b,src_c",
                   "to_state": "items"},
                  ["source_a", "source_b", "source_c"], 1, 0),
            _node("rank", "sort_items",
                  {"from_state": "items", "to_state": "items", "field": "score", "order": "desc"},
                  ["join"], 2, 0),
            _node("to_text", "items_to_text",
                  {"from_state": "items", "to_state": "text", "max_items": 15, "max_chars": 3500,
                   "template": "{{ id }}: {{ title }} ({{ score }})"},
                  ["rank"], 3, 0),
            _node("draft", "synthesize_report",
                  {"evidence": "$result:to_text", "prompt": "Combine these sources into one briefing.",
                   "style": "plain"}, ["to_text"], 4, 0),
        ],
    },

    # 8 ── human in the loop ─────────────────────────────────────────────────
    {
        "id": "tpl_human_review",
        "name": "Draft → human review → deliver",
        "category": "Approval",
        "description": "Generate content, park it for human approval through the shared verify node, and only deliver what a person signed off on. Mirrors Lane D's job-post flow.",
        "goal": "Draft content and hold it for review before delivery",
        "nodes": [
            _sticky("note_1", "verify_results with human_in_the_loop=true marks rows pending_approval.\noutput_user_results skips anything still pending.", 2, -1),
            _node("trigger", "trigger_manual",
                  {"items_json": '[{"title":"Weekly update","text":"…"}]', "to_state": "items"}, [], 0, 0),
            _node("compose", "template_render",
                  {"from_state": "items", "to_state": "items", "to_field": "text",
                   "template": "{{ title }}\n\n{{ text }}"},
                  ["trigger"], 1, 0),
            _node("verify", "verify_results",
                  {"results_json": "$result:compose", "human_in_the_loop": "true",
                   "config_json": "{}"}, ["compose"], 2, 0),
            _node("deliver", "output_user_results",
                  {"results_json": "$result:verify", "email_json": '{"enabled":false}',
                   "social_json": "[]", "config_json": "{}"}, ["verify"], 3, 0),
        ],
    },

    # 9 ── poll, dedupe, persist ─────────────────────────────────────────────
    {
        "id": "tpl_poll_dedupe_store",
        "name": "Poll → dedupe → store (TTL)",
        "category": "ETL",
        "description": "A scheduled poller: fetch, drop anything already seen, and persist into the workflow TTL store with retention. Attach it to schedule_graphs.json to run it hourly.",
        "goal": "Poll a source and persist new rows only",
        "nodes": [
            _sticky("note_1", "Register this graph id in schedule_graphs.json\nto run it on a timer.", 0, -1),
            _node("fetch", "http_request",
                  {"url": "https://api.example.com/v1/events", "method": "GET",
                   "max_items": 50, "to_state": "items"}, [], 0, 0),
            _node("dedupe", "remove_duplicates",
                  {"from_state": "items", "to_state": "items", "field": "id"},
                  ["fetch"], 1, 0),
            _node("shape", "set_fields",
                  {"from_state": "items", "to_state": "items",
                   "assignments_json": '{"id":"{{ id }}","title":"{{ title }}","seen_at":"={{ now }}"}'},
                  ["dedupe"], 2, 0),
            _node("store", "store_data",
                  {"workflow_id": "custom_poller", "items_json": "$result:shape",
                   "mode": "append", "retain_days": "7", "config_json": "{}"},
                  ["shape"], 3, 0),
        ],
    },

    # 10 ── research → knowledge base ────────────────────────────────────────
    {
        "id": "tpl_research_to_kb",
        "name": "Research → synthesize → learn",
        "category": "AI",
        "description": "Deep research on a topic, blend it with what Aiko already knows, write a report and file the result into the RAG knowledge store so she keeps it.",
        "goal": "Research a topic and retain the findings",
        "capabilities": ["research", "kb"],
        "nodes": [
            _node("web", "deep_research", {"query": "$prompt"}, [], 0, -1),
            _node("kb", "kb_search", {"query": "$prompt"}, [], 0, 1),
            _node("merge", "combine_evidence",
                  {"parts": ["$result:web", "$result:kb"], "separator": "\n\n---\n\n"},
                  ["web", "kb"], 1, 0),
            _node("draft", "synthesize_report",
                  {"evidence": "$result:merge", "prompt": "$prompt", "style": "auto"},
                  ["merge"], 2, 0),
            _node("report", "write_report",
                  {"title": "$title", "content": "$result:draft", "report_dir": "reports"},
                  ["draft"], 3, 0),
            _node("learn", "learn_report",
                  {"title": "$title", "text": "$result:draft", "kind": "self_learned"},
                  ["report"], 4, 0),
        ],
    },

    # 11 ── ops monitoring ───────────────────────────────────────────────────
    {
        "id": "tpl_health_alert",
        "name": "Health check → threshold → alert",
        "category": "IT ops",
        "description": "Sample system health, branch on a threshold, and only write an alert report when something is actually wrong. A quiet monitor instead of a noisy one.",
        "goal": "Alert only when the box is unhealthy",
        "capabilities": ["it_ops"],
        "nodes": [
            _node("health", "sys_health", {}, [], 0, 0),
            _node("parse", "code_transform",
                  {"items_json": "$result:health", "mode": "each", "to_state": "items",
                   "expression": "item"}, ["health"], 1, 0),
            _node("check", "if_condition",
                  {"from_state": "items", "mode": "any",
                   "conditions_json": '{"combinator":"or","conditions":['
                                      '{"field":"cpu_percent","op":"gte","value":85},'
                                      '{"field":"ram_percent","op":"gte","value":85},'
                                      '{"field":"disk_percent","op":"gte","value":90}]}'},
                  ["parse"], 2, 0),
            _node("alert", "write_report",
                  {"title": "System alert", "content": "$result:health", "report_dir": "reports/it"},
                  ["check"], 3, 0, run_if={"node": "check", "equals": "true"}),
        ],
    },

    # 12 ── log triage ───────────────────────────────────────────────────────
    {
        "id": "tpl_log_triage",
        "name": "Log triage → summarize → note",
        "category": "IT ops",
        "description": "Tail the log, pull out anomalies, have the LLM explain the most likely cause, and save the verdict as a note you can read tomorrow.",
        "goal": "Triage recent log anomalies",
        "capabilities": ["it_ops"],
        "nodes": [
            _node("triage", "log_triage", {"log_file": "logs/aiko.log", "lines": 150}, [], 0, 0),
            _node("draft", "synthesize_report",
                  {"evidence": "$result:triage",
                   "prompt": "Summarize these anomalies and name the single most likely next check.",
                   "style": "plain"}, ["triage"], 1, 0),
            _node("save", "save_note",
                  {"title": "Log triage", "content": "$result:draft", "folder": "notes"},
                  ["draft"], 2, 0),
        ],
    },

    # ── studio phase 2 templates ────────────────────────────────────────────

    # 8 ── agentic chat assistant (n8n-style composite agent) ─────────────────
    {
        "id": "tpl_agent_chat",
        "name": "Agentic chat assistant",
        "category": "Agentic",
        "description": "Chat trigger → AI Agent with attached Chat Model, Memory and Calendar tool sub-nodes — the n8n Tools Agent pattern.",
        "goal": "Conversational agent with memory and tools",
        "capabilities": ["ai"],
        "nodes": [
            _node("chat", "trigger_chat",
                  {"message": "$prompt", "from_user": "user", "to_state": "items"}, [], 0, 0),
            _node("agent", "ai_agent",
                  {"prompt": "$prompt",
                   "system_prompt": "You are Aiko, a helpful on-device assistant. Answer concisely.",
                   "chat_model_json": "$result:chat_model",
                   "tools_json": '["calendar_create"]',
                   "memory_key": "chat_memory",
                   "items_json": "$result:chat", "from_state": "", "to_state": "items"},
                  ["chat", "chat_model", "memory"], 1, 0),
            _node("chat_model", "chat_model",
                  {"model": "ministral-3b", "temperature": 0.7, "max_tokens": 1024},
                  [], 1, 1, attached_to="agent"),
            _node("memory", "memory_buffer",
                  {"memory_key": "chat_memory", "window_size": 10, "mode": "read",
                   "to_state": "memory_window"},
                  [], 1, 2, attached_to="agent"),
            _node("calendar", "calendar_create",
                  {"title": "$prompt", "start": "", "end": "", "description": ""},
                  ["agent"], 1, 3, attached_to="agent", sub_kind="tool"),
            _sticky("note_agent",
                    "Composite AI Agent: Chat Model, Memory and the Calendar tool ride along as attached sub-nodes. "
                    "The agent reads the model config via $result:chat_model and keeps context in the chat_memory window.",
                    2, 1),
        ],
    },

    # 9 ── social inbox triage ────────────────────────────────────────────────
    {
        "id": "tpl_social_triage",
        "name": "Social inbox triage",
        "category": "Social",
        "description": "On a schedule, check email and social mentions, summarize with the AI Agent, and send yourself a digest.",
        "goal": "Scheduled inbox + social digest",
        "capabilities": ["social", "scheduling"],
        "nodes": [
            _node("schedule", "trigger_schedule",
                  {"cron": "0 8 * * *", "timezone": "America/Los_Angeles", "to_state": "items"}, [], 0, 0),
            _node("email", "email_check",
                  {"max_results": 10, "to_state": "email_items"}, ["schedule"], 1, 0),
            _node("social", "social_read",
                  {"services": "threads", "max_results": 10, "to_state": "social_items"}, ["schedule"], 1, 1),
            # $result: only substitutes whole-string references, so fan-in goes
            # through merge_items with distinct state keys rather than a
            # literal "[..., ...]" string (which would never be substituted).
            _node("merge", "merge_items",
                  {"mode": "append", "from_states": "email_items,social_items", "to_state": "triage_items"},
                  ["email", "social"], 2, 0),
            _node("summarize", "ai_agent",
                  {"prompt": "Summarize the new emails and social mentions below. Lead with anything urgent, then a 5-bullet digest.",
                   "chat_model_json": "$result:chat_model",
                   "tools_json": "[]", "memory_key": "",
                   "items_json": "$result:merge", "from_state": "", "to_state": "items"},
                  ["merge", "chat_model"], 3, 0),
            _node("chat_model", "chat_model",
                  {"model": "ministral-3b", "temperature": 0.5, "max_tokens": 1024},
                  [], 2, 1, attached_to="summarize"),
            _node("digest", "notify_user",
                  {"title": "Inbox triage", "message": "$result:summarize", "channel": "app",
                   "items_json": "$result:summarize", "from_state": "", "to_state": "items"},
                  ["summarize"], 4, 0),
        ],
    },

    # 10 ── morning briefing ──────────────────────────────────────────────────
    {
        "id": "tpl_morning_briefing",
        "name": "Morning briefing",
        "category": "Everyday",
        "description": "Every morning: pull your calendar and the news, compose a briefing with the AI Agent, and notify yourself.",
        "goal": "Daily morning briefing",
        "capabilities": ["scheduling", "research"],
        "nodes": [
            _node("schedule", "trigger_schedule",
                  {"cron": "0 7 * * *", "timezone": "America/Los_Angeles", "to_state": "items"}, [], 0, 0),
            _node("cal", "calendar_list",
                  {"max_results": 10, "time_min": "", "to_state": "cal_items"}, ["schedule"], 1, 0),
            _node("news", "rss_read",
                  {"url": "https://feeds.bbci.co.uk/news/rss.xml", "max_items": 15, "to_state": "news_items"},
                  ["schedule"], 1, 1),
            # $result: only substitutes whole-string references, so fan-in goes
            # through merge_items with distinct state keys rather than a
            # literal "[..., ...]" string (which would never be substituted).
            _node("merge", "merge_items",
                  {"mode": "append", "from_states": "cal_items,news_items", "to_state": "brief_items"},
                  ["cal", "news"], 2, 0),
            _node("brief", "ai_agent",
                  {"prompt": "Write my morning briefing: today's calendar events first, then the 5 most important headlines with one line each. Keep it skimmable.",
                   "chat_model_json": "$result:chat_model",
                   "tools_json": "[]", "memory_key": "",
                   "items_json": "$result:merge", "from_state": "", "to_state": "items"},
                  ["merge", "chat_model"], 3, 0),
            _node("chat_model", "chat_model",
                  {"model": "ministral-3b", "temperature": 0.5, "max_tokens": 1500},
                  [], 2, 1, attached_to="brief"),
            _node("send", "notify_user",
                  {"title": "Morning briefing", "message": "$result:brief", "channel": "email",
                   "items_json": "$result:brief", "from_state": "", "to_state": "items"},
                  ["brief"], 4, 0),
        ],
    },

    # 11 ── email auto-drafter ────────────────────────────────────────────────
    {
        "id": "tpl_email_autodrafter",
        "name": "Email auto-drafter",
        "category": "Email",
        "description": "Check the inbox, draft replies with the AI Agent, and park them for review — nothing is sent.",
        "goal": "Draft email replies automatically",
        "capabilities": ["email"],
        "nodes": [
            _node("inbox", "email_check",
                  {"max_results": 5, "to_state": "items"}, [], 0, 0),
            _node("drafter", "ai_agent",
                  {"prompt": "For each email below, draft a short, polite reply in my voice. Return one draft per email with the original subject.",
                   "system_prompt": "You are Aiko drafting email replies. Be brief, warm, and professional.",
                   "chat_model_json": "$result:chat_model",
                   "tools_json": "[]", "memory_key": "",
                   "items_json": "$result:inbox", "from_state": "", "to_state": "items"},
                  ["inbox", "chat_model"], 1, 0),
            _node("chat_model", "chat_model",
                  {"model": "ministral-3b", "temperature": 0.7, "max_tokens": 1024},
                  [], 1, 1, attached_to="drafter"),
            _node("drafts", "email_draft",
                  {"to": "", "subject": "", "body": "$result:drafter", "drafts_key": "email_drafts",
                   "items_json": "$result:drafter", "from_state": "", "to_state": "items"},
                  ["drafter"], 2, 0),
            _sticky("note_drafts",
                    "Drafts are parked under the email_drafts state key — nothing is sent. "
                    "Review them, then send via the Email: Reply node or by hand.",
                    3, 1),
        ],
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# API
# ═════════════════════════════════════════════════════════════════════════════

def list_templates() -> list[dict[str, Any]]:
    """Template summaries for the studio gallery (no node payloads)."""
    return [
        {
            "id": tpl["id"],
            "name": tpl["name"],
            "category": tpl.get("category") or "Other",
            "description": tpl.get("description") or "",
            "node_count": len([n for n in tpl["nodes"] if n.get("tool") != "sticky_note"]),
            "tools": sorted({n["tool"] for n in tpl["nodes"] if n.get("tool") != "sticky_note"}),
        }
        for tpl in TEMPLATES
    ]


def get_template(template_id: str) -> dict[str, Any] | None:
    for tpl in TEMPLATES:
        if tpl["id"] == template_id:
            return copy.deepcopy(tpl)
    return None


def instantiate(template_id: str, new_id: str, name: str = "", goal: str = "") -> dict[str, Any] | None:
    """Clone a template into a user playbook dict ready to be persisted.

    Triggers stay empty on purpose: a user-created graph is run explicitly
    from the studio or from schedule_graphs.json, never auto-matched against
    a chat prompt until the owner adds triggers themselves.
    """
    tpl = get_template(template_id)
    if tpl is None:
        return None
    return {
        "id": new_id,
        "name": name or tpl["name"],
        "goal": goal or tpl.get("goal") or tpl["name"],
        "description": tpl.get("description", ""),
        "triggers": [],
        "requires_any": [],
        "capabilities": list(tpl.get("capabilities") or []),
        "source": "studio",
        "from_template": template_id,
        "nodes": copy.deepcopy(tpl["nodes"]),
    }


__all__ = ["TEMPLATES", "list_templates", "get_template", "instantiate"]
