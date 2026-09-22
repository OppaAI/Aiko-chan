"""Stage 4 turn hooks: eligibility trail + MB consolidate (optional)."""
from __future__ import annotations

import logging

log = logging.getLogger("aiko.fly_behavior.stage4_hooks")


def after_online_teach(user_id, text: str, out: dict) -> None:
    try:
        from cognition.flymemory.eligibility import record_step
        record_step(user_id, text or "")
    except Exception as exc:
        log.debug("eligibility record_step skipped: %s", exc)
    try:
        from cognition.fly_behavior.sleep_sched import maybe_consolidate_mb
        out["mb_consolidate"] = maybe_consolidate_mb(user_id)
    except Exception as exc:
        log.debug("mb_consolidate skipped: %s", exp if False else exc)
