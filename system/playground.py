"""Aiko-Playground autonomous build loop — scheduler integration.

The Playground (public repo OppaAI/Aiko-Playground) is Aiko's own workshop:
while the user is genuinely idle she picks up the top todo goal, builds it in
``work/<slug>/``, tests everything through ``sandbox/run.py``, self-reviews
against the goal's acceptance criteria, and sends exactly one completion email
per finished goal.

This module owns the scheduler side of that loop:

- :func:`ensure_playground_job` seeds one idempotent ``interval``/``agentic``
  schedule record ("Aiko-Playground idle build"). Because it is an ordinary
  schedule record it shows up in Calendar Studio, where it can be edited,
  paused (disabled), or deleted like any other scheduled task.
- The record is ``requires_idle`` with a 10-minute threshold, so the tick only
  turns into real work when the user has actually been away. Each fire runs
  through Aiko's normal agentic loop (``think.handle_scheduled_job``) with the
  worker instructions below passed as the job's ``skill``.
- The tick runs in ``worker_mode``: a lean coding-worker profile (minimal
  system prompt, coding tools only, no memory/KB/persona fetch). Fixed
  overhead drops from ~4-6k tokens to ~1k so a 10k/16k window holds real
  think-act-observe cycles instead of overhead.
- The worker itself is stateless across ticks: progress lives in checkpoint
  files inside the Playground repo, so a tick always resumes the active goal
  instead of starting over.

Safety contract (also repeated in the worker instructions):

- Build ONLY inside the Playground directory. Never modify Aiko-chan source,
  config, or any other repo autonomously — findings about Aiko-chan become a
  REPORT.md note or a PR proposal, never a direct edit.
- Execute ALL built code through ``sandbox/run.py`` (path-confined, timeouts).
  Never run goal code directly.
- The public repo must never contain secrets, credentials, private raw logs,
  or sensitive machine data. Completion email goes through Aiko's own
  send_email tool (ProtonMail) to AIKO_EMAIL; loop/notify.py over SMTP
  (PLAYGROUND_SMTP_PASS) is fallback-only.
- Exactly one completion email per finished goal, via ``send_email``
  (fallback: ``loop/notify.py``).
- Instant kill switch: disabling the schedule record (e.g. in Calendar Studio)
  stops future ticks; a ``loop/disabled`` sentinel file in the Playground repo
  makes a running tick stand down immediately.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("aiko.playground")

PLAYGROUND_JOB_TITLE = "Aiko-Playground idle build"
PLAYGROUND_TICK_SECONDS = 900  # scheduler wakes every 15 minutes
PLAYGROUND_IDLE_THRESHOLD_SECONDS = 600  # ...but work only starts after 10 idle minutes
PLAYGROUND_MAX_SESSION_MINUTES = 45  # hard bound per work session


def playground_dir() -> Path:
    """Resolve the Playground checkout; configurable, never silently assumed."""
    configured = os.environ.get("AIKO_PLAYGROUND_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Aiko-Playground"


PLAYGROUND_WORKER_SKILL = """\
AIKO-PLAYGROUND BUILD SESSION (one bounded session per tick)

You are Aiko, doing one self-coding work session in your workshop.
Repo: {playground_dir} | Time box: {max_minutes} min.

PROCEDURE (on disk, not in this prompt): read loop/WORKER.md first and
follow it exactly. It holds the 14 build rules — goal picking, sandbox
discipline, checkpointing, review, email, and safety rules.

PER TICK:
1. If loop/disabled exists, do nothing. If your human is active, wrap up now.
2. Read GOALS.md; pick the top goal with no work/<slug>/REPORT.md yet.
   Read its goal file + work/<slug>/CHECKPOINT.md if present.
3. ONE goal, build ONLY in work/<slug>/. Keep context small: read only the
   files you need, keep tool outputs short, checkpoint often.
4. On interruption or ANY error: write work/<slug>/CHECKPOINT.md with the
   blocker, then stop. Next tick resumes from the checkpoint.

If anything can't be done safely, stop and leave CHECKPOINT.md explaining why.
"""


def worker_skill_text() -> str:
    """Render the worker instructions with the current repo path bound in."""
    return PLAYGROUND_WORKER_SKILL.format(
        playground_dir=playground_dir(),
        max_minutes=PLAYGROUND_MAX_SESSION_MINUTES,
    )


def ensure_playground_job(timezone: str | None = None, user_id: str | None = None) -> dict[str, Any]:
    """Seed the Playground idle-build schedule record (idempotent by title).

    Returns the existing record when one is already present, so every boot
    seeds at most one tick job. The record is an ordinary schedule entry: it
    appears in Calendar Studio and disabling it there pauses the loop.

    Idempotent by title, but NOT frozen. Every boot re-syncs the fields this
    module owns (skill text, failure-note dir, lean profile) so a record seeded
    by an older revision cannot keep serving stale instructions. Records whose
    owned fields already match are left completely untouched, so a user's own
    edits (enabled, cadence, title) always survive.
    """
    from system.schedule import (
        _read_all,
        notify_scheduler_new_job,
        schedule_job_record,
        update_schedule_record,
    )

    desired_skill = worker_skill_text().strip()
    # str(), not Path: schedule records are JSON, and a PosixPath made
    # _write_all raise TypeError — so creating this job failed outright on any
    # fresh install, and no failure note was ever recorded.
    desired_note_dir = str(playground_dir())

    for job in _read_all(user_id=user_id):
        if job.get("title") != PLAYGROUND_JOB_TITLE:
            continue
        drift: dict[str, Any] = {}
        # Compare stripped: schedule records store skill stripped, so comparing
        # against the raw template (which ends in a newline) would report drift
        # on every single boot and rewrite the record forever.
        if (job.get("skill") or "").strip() != desired_skill:
            drift["skill"] = desired_skill
        if job.get("failure_note_dir") != desired_note_dir:
            drift["failure_note_dir"] = desired_note_dir
        if job.get("lean_context") is not True:
            drift["lean_context"] = True
        if job.get("worker_mode") is not True:
            drift["worker_mode"] = True
        if not drift:
            return job
        updated = update_schedule_record(job["id"], drift, user_id=user_id)
        log.info(
            "Playground: re-synced job %s (%s).",
            job.get("id"), ", ".join(sorted(drift)),
        )
        return updated or job

    record = schedule_job_record(
        title=PLAYGROUND_JOB_TITLE,
        task=(
            "Run one bounded Aiko-Playground self-coding session: pick the top "
            "open goal and advance it. Follow the worker instructions exactly."
        ),
        time_of_day="00:00",
        frequency="interval",
        interval_seconds=PLAYGROUND_TICK_SECONDS,
        timezone=timezone,
        action="agentic",
        requires_idle=True,
        idle_seconds=PLAYGROUND_IDLE_THRESHOLD_SECONDS,
        skill=desired_skill,
        failure_note_dir=desired_note_dir,
        lean_context=True,
        worker_mode=True,
        user_id=user_id,
    )
    log.info("Playground: seeded idle-build job %s (every %ds, requires %ds idle).",
             record["id"], PLAYGROUND_TICK_SECONDS, PLAYGROUND_IDLE_THRESHOLD_SECONDS)
    notify_scheduler_new_job()
    return record
