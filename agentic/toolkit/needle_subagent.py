"""
agentic/toolkit/needle_subagent.py

Needle 3 as a true coding subagent: task in, compact summary out.

The problem this solves: Aiko (Ministral-3B) has a small context window.
Loading file contents, test output, and search results directly into her
context kills it in 2-3 operations. Claude Code / Codex solve this with
subagents that have their own isolated context — the parent only ever
sees a compact summary.

Needle 3 runs as a separate local server process, so it already has an
isolated context. This module runs a bounded ReAct loop against a Needle
worker:

    task -> Needle proposes tools -> Aiko validates + executes ->
    results back to Needle -> repeat -> Needle returns summary

The caller (Aiko) only ever receives the final summary (<= SUMMARY_BUDGET
chars). The full trace — file contents, test output, intermediate steps —
never touches Aiko's context.

Design rules:
  - Needle proposes, Aiko disposes. Every tool call is validated against
    the capability-filtered subset and executed through Aiko's registry
    + approval gates. Needle never runs tools directly.
  - Bounded: MAX_TURNS ReAct iterations, then force-summarize.
  - Budgets are brutal on purpose. Small model, small window.
"""

from __future__ import annotations

import json
import os
from collections import deque

from agentic.registry import TOOLS, tool
from agentic.toolkit.common import json_block
from system.log import get_logger

log = get_logger(__name__)


def _spec(name: str, description: str):
    return TOOLS[name] if name in TOOLS else name


# ---- budgets (chars) ----
TASK_BUDGET = 1500          # max task description sent to worker
SUMMARY_BUDGET = 800        # max summary returned to Aiko
TOOL_RESULT_BUDGET = 1200   # max per tool-result fed back to Needle
TOOL_HISTORY_BUDGET = 4800  # max combined tool-call/result history
MAX_TURNS = 6               # ReAct iterations before force-summarize

# Tools a subagent is allowed to use. Deliberately narrow:
# read + search + test. No writes — the subagent investigates,
# Aiko decides and patches.
# adaptive_search (web) lets the subagent find the best approach/
# library/docs for a goal, like Claude Code does.
SUBAGENT_TOOLS = frozenset({
    "repo_read_file",
    "repo_search_text",
    "codebase_search",
    "code_run_tests",
    "sandbox_run",
    "adaptive_search",
})


@tool(
    _spec("needle_subagent", "Run a coding subtask on an isolated Needle 3 worker; returns a compact summary."),
    description="Run a coding subtask on an isolated Needle 3 worker; returns a compact summary.",
    graph=True,
    react=True,
    domain="multi_agent",
)
def needle_subagent(task: str = "", tools_hint: str = "") -> str:
    """Run a bounded ReAct loop on a Needle 3 worker.

    task: what to investigate/do (capped at TASK_BUDGET chars).
    tools_hint: optional comma-separated tool names to further restrict
        the subset (intersected with SUBAGENT_TOOLS).

    Returns a compact summary (<= SUMMARY_BUDGET chars). The full trace
    never touches the caller's context.
    """
    try:
        from agentic.needle import NeedleClient, NeedleError, NeedleLowConfidence
        from agentic.needle_orchestrator import load_needle_workers
        from agentic.registry import registry
    except ImportError as e:
        return json_block("needle_subagent", {"ok": False, "error": f"import failed: {e}"})

    task = (task or "").strip()[:TASK_BUDGET]
    if not task:
        return json_block("needle_subagent", {"ok": False, "error": "task required"})

    raw = os.getenv("NEEDLE_WORKERS", "")
    if not raw.strip():
        return json_block("needle_subagent", {
            "ok": False, "error": "no Needle workers configured",
            "hint": "Set NEEDLE_WORKERS or call needle_team_status for the format.",
            "fallback": "Do the subtask inline with tight budgets (repo_read_file <= 3000 chars).",
        })

    try:
        workers = load_needle_workers(
            raw,
            default_timeout=float(os.getenv("NEEDLE_TIMEOUT", "20")),
            default_confidence_threshold=float(os.getenv("NEEDLE_CONFIDENCE_THRESHOLD", "0.85")),
            max_workers=int(os.getenv("NEEDLE_MAX_WORKERS", "4")),
        )
    except Exception as e:
        return json_block("needle_subagent", {"ok": False, "error": f"worker load failed: {e}"})
    if not workers:
        return json_block("needle_subagent", {"ok": False, "error": "no workers loaded"})

    worker = workers[0]  # one worker per call; fan-out via needle_team_run
    allowed = set(SUBAGENT_TOOLS) & set(worker.allowed_tools)
    if tools_hint.strip():
        allowed &= {t.strip() for t in tools_hint.split(",") if t.strip()}
    tool_schemas = []
    for name in sorted(allowed):
        spec = registry.get(name)
        if spec:
            tool_schemas.append(spec.to_openai_schema())
    allowed = {schema["function"]["name"] for schema in tool_schemas}
    if not allowed:
        return json_block("needle_subagent", {"ok": False, "error": "no allowed tools resolved"})

    client = NeedleClient(
        worker.base_url,
        timeout=worker.timeout,
        confidence_threshold=worker.confidence_threshold,
    )

    # ---- bounded ReAct loop ----
    transcript = deque(maxlen=4)
    history = ""
    base_prompt = (
        f"You are a coding subagent. Task: {task}\n\n"
        "Use the provided tools to investigate. You may search the web "
        "(adaptive_search) to find the best library, docs, or approach "
        "for the task — like checking Stack Overflow or official docs "
        "before writing code. "
        "When you have the answer, respond with type 'answer' and put "
        "the complete findings in the response field, under 800 characters. "
        "Be specific: file paths, line numbers, exact error text, and "
        "URLs for any web sources you used. Do not dump whole files."
    )
    prompt = base_prompt
    try:
        from agentic.agentic import TaskState, execute_tool_with_policy

        state = TaskState(goal=task)
        for turn in range(MAX_TURNS):
            try:
                resp = client.complete(prompt, tool_schemas)
            except NeedleLowConfidence as e:
                log.warning("needle_subagent low confidence on turn %d: %s", turn, e)
                return json_block("needle_subagent", {
                    "ok": False, "error": f"low confidence: {e}"[:300], "turns": turn + 1,
                })
            except NeedleError as e:
                return json_block("needle_subagent", {"ok": False, "error": str(e)[:300]})

            if resp.kind == "answer" or not resp.calls:
                summary = (resp.content or "").strip()[:SUMMARY_BUDGET]
                return json_block("needle_subagent", {
                    "ok": True, "summary": summary, "turns": turn + 1,
                })

            # Execute proposed calls through Aiko's registry (validate + run)
            for call in resp.calls:
                arguments = json.dumps(call.arguments, ensure_ascii=False)[:400]
                if call.name not in allowed:
                    out_str = "DENIED: outside subagent toolset"
                else:
                    try:
                        result = execute_tool_with_policy(call.name, call.arguments, state)
                        out_str = result.observation()[:TOOL_RESULT_BUDGET]
                    except Exception as e:
                        out_str = f"ERROR: {e}"[:TOOL_RESULT_BUDGET]
                entry = f"turn {turn + 1}: {call.name[:100]}({arguments}) -> {out_str}"
                history = (history + "\n" + entry)[-TOOL_HISTORY_BUDGET:]
                transcript.append(f"turn {turn + 1}: {call.name[:100]} -> {out_str[:200]}")

            prompt = (
                base_prompt + "\n\nRecent tool calls and results:\n" + history +
                "\n\nContinue investigating, or respond with type 'answer' "
                "and your findings under 800 characters."
            )

        # Force-summarize from transcript if we hit the turn cap
        fallback = " | ".join(transcript)[:SUMMARY_BUDGET]
        return json_block("needle_subagent", {
            "ok": True, "summary": fallback or "max turns reached, no findings",
            "turns": MAX_TURNS, "truncated": True,
        })
    except Exception as e:
        return json_block("needle_subagent", {"ok": False, "error": f"subagent loop failed: {e}"[:300]})
