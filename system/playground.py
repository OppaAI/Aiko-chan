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
  or sensitive machine data. SMTP password comes from the ``PLAYGROUND_SMTP_PASS``
  environment variable, never from a file in the repo.
- Exactly one completion email per finished goal, via ``loop/notify.py``.
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
AIKO-PLAYGROUND AUTONOMOUS BUILD SESSION (one bounded session per tick)

You are Aiko, doing one bounded self-coding work session in your own workshop,
the Aiko-Playground repo. A scheduler tick started this session because your
human has been idle for a while. Another tick will resume where you stop, so
work in small checkpointed steps.

REPO: {playground_dir}  (override with the AIKO_PLAYGROUND_DIR env var)
GOALS: GOALS.md + goals/<slug>.md in that repo.

SESSION RULES
1. First check: if <repo>/loop/disabled exists, do nothing and end the session.
   If a conversation with your human is active right now, wrap up immediately.
2. Read GOALS.md. Pick the highest-priority goal whose work/<slug>/REPORT.md
   does not exist yet. Read its goal file and any work/<slug>/CHECKPOINT.md.
3. Work for at most {max_minutes} minutes on that ONE goal, in small steps.
   Build only inside work/<slug>/ — never touch Aiko-chan source, config, or
   any other repository. If you find something worth changing in Aiko-chan,
   write it up in the goal's REPORT.md as a proposal instead.
4. Run EVERYTHING you build through sandbox/run.py (it is path-confined and
   enforces timeouts). Never execute goal code directly, and never run
   anything outside the Playground directory.
5. Checkpoint as you go: keep work/<slug>/CHECKPOINT.md current (what works,
   what's next, exact commands to resume). If the session ends mid-goal, the
   next tick resumes from this file.
6. When the goal's acceptance criteria are met: self-review the diff against
   each criterion, write work/<slug>/REPORT.md (what was built, test evidence,
   how to run it), commit locally with a clear message.
7. Push only if git credentials work non-interactively; never print or store
   tokens. If push fails, leave the commit local and note it in REPORT.md.
8. Exactly ONE completion email per finished goal, via loop/notify.py.
   Recipient defaults to oppa.ai.org@proton.me (or loop/config.yaml).
   SMTP password comes ONLY from the PLAYGROUND_SMTP_PASS environment
   variable. If it is unavailable, write the email body to
   work/<slug>/EMAIL_DRAFT.md and note that the email is pending — do not
   invent credentials and do not retry-send later.
9. NEVER put secrets, API keys, private raw logs, or sensitive machine data
   in the public repo. Sanitize benchmark output before committing.
10. End the session cleanly when the time box is up or the goal is done.
    Do not start a second goal in the same session.

If anything above cannot be satisfied safely, stop and leave a CHECKPOINT.md
explaining what blocked you. A stopped session is always better than a
reckless one.
"""


def worker_skill_text() -> str:
    """Render the worker instructions with the current repo path bound in."""
    return PLAYGROUND_WORKER_SKILL.format(
        playground_dir=playground_dir(),
        max_minutes=PLAYGROUND_MAX_SESSION_MINUTES,
    )


def ensure_playground_job(timezone: str | None = None, user_id: str | None = None) -> dict[str, Any]:
    """Seed the Playground idle-build schedule record (idempotent by title).

    Returns the existing record when one is already present, so every
    boot seeds at most one tick job. The record is an ordinary schedule entry:
    it appears in Calendar Studio and disabling it there pauses the loop.
    """
    from system.schedule import _read_all, notify_scheduler_new_job, schedule_job_record

    for job in _read_all(user_id=user_id):
        if job.get("title") == PLAYGROUND_JOB_TITLE:
            return job
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
        skill=worker_skill_text(),
        user_id=user_id,
    )
    log.info("Playground: seeded idle-build job %s (every %ds, requires %ds idle).",
             record["id"], PLAYGROUND_TICK_SECONDS, PLAYGROUND_IDLE_THRESHOLD_SECONDS)
    notify_scheduler_new_job()
    return record
