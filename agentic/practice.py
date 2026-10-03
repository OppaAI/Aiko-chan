#!/usr/bin/env python3
"""Practice and promote schema-driven graph workflows without booting the chat LLM.

Examples:
  uv run python -m agentic.practice --task "make a deployment checklist and save it" \
    --tools create_checklist save_note --promote

  uv run python -m agentic.practice --task "research X and save a note" \
    --steps '[{"tool":"deep_search","ok":true,"args":{"query":"$prompt"}},{"tool":"save_note","ok":true,"args":{"title":"$title","content":"$result:step_1"}}]' \
    --promote
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.config import load_config
load_config()

from agentic import experience
from agentic import graph_engine as schema


# ── Autonomous practice task picker ─────────────────────────────────────────
# Used by the practice scheduler job (system/practice.py). Picks ONE task per
# tick, in priority order:
#   1. retry   — a recent failed workflow ("practice what failed")
#   2. rehearse — a successful but rarely-used workflow (reinforce it)
#   3. curriculum — a safe, idempotent built-in exercise (round-robin by day)

_PRACTICE_CURRICULUM: list[dict[str, Any]] = [
    {
        "goal": "Practice: search your knowledge for 'practice routines' and condense the top hit into one paragraph.",
        "tools": ["kb_search", "condense_text"],
    },
    {
        "goal": "Practice: draft a 3-item checklist for a tidy workspace and save it as a note titled 'Practice checklist'.",
        "tools": ["create_checklist", "save_note"],
    },
    {
        "goal": "Practice: make a short plan for learning one agentic tool you rarely use, then summarize the plan in two sentences.",
        "tools": ["make_plan", "condense_text"],
    },
    {
        "goal": "Practice: search your knowledge for 'playbooks', then write a two-sentence note titled 'Practice note' about when a playbook helps.",
        "tools": ["kb_search", "save_note"],
    },
]


def pick_practice_task(user_id: str | None = None) -> dict[str, Any]:
    """Pick one practice task for this tick. Never raises — falls back to curriculum."""
    try:
        from datetime import datetime, timedelta, timezone
        from system.userspace import current_user_id
        from agentic.experience.schema import connect, ensure_experience_schema_migrated

        uid = user_id or current_user_id()
        conn = connect(uid)
        try:
            ensure_experience_schema_migrated(conn)
            week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            # 1. Retry a recent failure.
            row = conn.execute(
                "SELECT id, goal, steps_json FROM experiences "
                "WHERE user_id = ? AND status = 'active' AND outcome = 'failed' "
                "AND created_at > ? ORDER BY created_at DESC LIMIT 1",
                (uid, week_ago),
            ).fetchone()
            if row:
                steps = _step_tools(row["steps_json"])
                return {
                    "source": "retry",
                    "experience_id": row["id"],
                    "goal": f"Practice (retry): {row['goal']}",
                    "suggested_tools": steps,
                    "note": "This workflow failed recently. Try it again with corrected arguments.",
                }
            # 2. Rehearse a rarely-used success.
            row = conn.execute(
                "SELECT id, goal, steps_json FROM experiences "
                "WHERE user_id = ? AND status = 'active' AND outcome = 'ok' "
                "AND use_count < 2 ORDER BY use_count ASC, created_at DESC LIMIT 1",
                (uid,),
            ).fetchone()
            if row:
                return {
                    "source": "rehearse",
                    "experience_id": row["id"],
                    "goal": f"Practice (rehearse): {row['goal']}",
                    "suggested_tools": _step_tools(row["steps_json"]),
                    "note": "This workflow succeeded but is rarely used. Rehearse it to reinforce.",
                }
        finally:
            conn.close()
    except Exception:
        pass
    # 3. Curriculum fallback (round-robin by day of year).
    from datetime import datetime, timezone
    idx = datetime.now(timezone.utc).timetuple().tm_yday % len(_PRACTICE_CURRICULUM)
    item = _PRACTICE_CURRICULUM[idx]
    return {
        "source": "curriculum",
        "experience_id": None,
        "goal": item["goal"],
        "suggested_tools": list(item["tools"]),
        "note": "Built-in safe exercise (read-mostly tools, idempotent).",
    }


def _step_tools(steps_json: str | None) -> list[str]:
    try:
        steps = json.loads(steps_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    tools: list[str] = []
    for s in steps if isinstance(steps, list) else []:
        t = str((s or {}).get("tool") or "").strip()
        if t and t not in {"final_answer", "llm_call"} and t not in tools:
            tools.append(t)
    return tools[:8]


def _steps_from_tools(tools: list[str]) -> list[dict[str, Any]]:
    return [{"tool": t, "ok": True, "args": {}} for t in tools]


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Aiko experience and playbook workflows from practice examples.")
    parser.add_argument("--task", required=True, help="Example task prompt Aiko should learn/practice.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tools", nargs="+", help="Ordered tool names for the workflow.")
    group.add_argument("--steps", help="JSON list of step objects with tool/ok/args fields.")
    parser.add_argument("--answer", default="practice workflow recorded", help="Outcome excerpt to store with the experience.")
    parser.add_argument("--promote", action="store_true", help="Append the practiced sequence to the graph playbook JSON.")
    parser.add_argument("--name", help="Human-readable name for the promoted playbook.")
    args = parser.parse_args()

    if args.steps:
        steps = json.loads(args.steps)
        if not isinstance(steps, list):
            raise SystemExit("--steps must decode to a JSON list")
    else:
        steps = _steps_from_tools(args.tools or [])

    exp_id = experience.record_practice_experience(args.task, steps, args.answer, verified_ok=True, score=1.0)
    print(f"recorded_experience={exp_id}")

    if args.promote:
        path, plan_id = schema.append_playbook_from_experience(
            args.task, steps, name=args.name, source_experience_id=exp_id)
        print(f"promoted_playbook={path} plan_id={plan_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
