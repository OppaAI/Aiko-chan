"""Aiko practice sessions — autonomous trial runs that build experience.

While the user is genuinely idle, Aiko picks ONE practice task per tick and
runs it: retrying a recently failed workflow, rehearsing a rarely-used one,
or working a safe built-in exercise. Outcomes are recorded as experience;
workflows used often enough auto-promote into playbook DAGs
(:func:`agentic.graph_engine.maybe_autopromote_experiences`).

This module owns the scheduler side of that loop:

- :func:`ensure_practice_job` seeds one idempotent ``interval``/``agentic``
  schedule record ("Aiko practice session"). Because it is an ordinary
  schedule record it shows up in Calendar Studio, where it can be edited,
  paused (disabled), or deleted like any other scheduled task.
- The record is ``requires_idle`` so the tick only turns into real work when
  the user has actually been away. Each fire runs through Aiko's normal
  agentic loop (``think.handle_scheduled_job``) with the worker instructions
  below passed as the job's ``skill``.

Safety contract (also repeated in the worker instructions):

- Practice ONLY with the suggested tools plus read-only tools. Never send
  email, post, purchase, delete, or run anything with irreversible side
  effects during practice.
- Never modify Aiko-chan source, config, or any other repo — practice writes
  notes/summaries only.
- One task per tick, bounded by AIKO_PRACTICE_MAX_MINUTES (default 15).
- Instant kill switch: disabling the schedule record (e.g. in Calendar
  Studio) stops future ticks; ``AIKO_PRACTICE_ENABLED=0`` disables the loop.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("aiko.practice")

PRACTICE_JOB_TITLE = "Aiko practice session"
PRACTICE_TICK_SECONDS = 7200  # scheduler wakes every 2 hours
PRACTICE_IDLE_THRESHOLD_SECONDS = 600  # ...but work only starts after 10 idle minutes
PRACTICE_MAX_SESSION_MINUTES = 15  # hard bound per practice session


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default)).strip() or default))
    except (ValueError, TypeError):
        return default


def practice_enabled() -> bool:
    return os.getenv("AIKO_PRACTICE_ENABLED", "1").strip() != "0"


PRACTICE_WORKER_SKILL = """\
AIKO PRACTICE SESSION (one bounded session per tick)

You are Aiko, doing one short practice session while your human is idle.
A scheduler tick started this. Practice keeps your agentic skills sharp and
turns repeated workflows into reusable playbooks.

RULES
1. If your human is active, wrap up immediately.
2. Call suggest_practice_task ONCE and take the task it returns.
3. Attempt the task with the suggested tools (plus read-only tools only).
   Prefer run_playbook when a playbook matches; otherwise do it step by step.
4. NEVER during practice: send email/messages, post, purchase, delete files,
   modify code or config, or run anything with irreversible side effects.
   Notes and summaries are fine.
5. Call record_practice_result ONCE with what you did and whether it worked,
   passing the experience_id from suggest_practice_task unchanged when the
   task came from an existing workflow.
6. Call practice_sweep ONCE at the end (promotes frequently-used workflows).
7. ONE task per tick, at most {max_minutes} minutes. Then stop. Report what
   you practiced and the outcome in one short paragraph.

If anything can't be done safely, stop and say what blocked you.
"""


def worker_skill_text() -> str:
    """Render the worker instructions with the current time box bound in."""
    return PRACTICE_WORKER_SKILL.format(max_minutes=_env_int("AIKO_PRACTICE_MAX_MINUTES", PRACTICE_MAX_SESSION_MINUTES))


def ensure_practice_job(timezone: str | None = None, user_id: str | None = None) -> dict[str, Any] | None:
    """Seed the practice schedule record (idempotent by title).

    Returns the existing record when one is already present, or None when
    the loop is disabled via AIKO_PRACTICE_ENABLED=0.
    """
    if not practice_enabled():
        log.info("Practice: disabled via AIKO_PRACTICE_ENABLED=0; not seeding job.")
        return None
    from system.schedule import _read_all, notify_scheduler_new_job, schedule_job_record

    for job in _read_all(user_id=user_id):
        if job.get("title") == PRACTICE_JOB_TITLE:
            return job
    tick = _env_int("AIKO_PRACTICE_TICK_SECONDS", PRACTICE_TICK_SECONDS)
    idle = _env_int("AIKO_PRACTICE_IDLE_SECONDS", PRACTICE_IDLE_THRESHOLD_SECONDS)
    record = schedule_job_record(
        title=PRACTICE_JOB_TITLE,
        task=(
            "Run one bounded Aiko practice session: pick one practice task "
            "and attempt it. Follow the worker instructions exactly."
        ),
        time_of_day="00:00",
        frequency="interval",
        interval_seconds=tick,
        timezone=timezone,
        action="agentic",
        requires_idle=True,
        idle_seconds=idle,
        skill=worker_skill_text(),
        user_id=user_id,
    )
    log.info("Practice: seeded practice job %s (every %ds, requires %ds idle).",
             record["id"], tick, idle)
    notify_scheduler_new_job()
    return record
