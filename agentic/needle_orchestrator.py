"""Deterministic, bounded orchestration for independent Needle 2 workers.

Workers are configured explicitly and must point at separate Needle servers (or
at server sessions that the Needle deployment documents as isolated).  This
module only aggregates constrained tool proposals; Aiko's normal ReAct loop
continues to validate, approve, and execute every proposed call.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from agentic.needle import NeedleClient, NeedleError, NeedleResponse

try:
    from system.log import get_logger as _get_logger
    log = _get_logger(__name__)
except Exception:
    import logging as _logging
    log = _logging.getLogger("aiko.needle_orchestrator")

try:
    from system.config import env_str as _env_str
except Exception:
    _env_str = None


def _flycx_mode() -> str:
    try:
        if _env_str is None:
            return "off"
        return _env_str("MEMORY_FLYCX_MODE", "off").strip().lower()
    except Exception:
        return "off"


def _flycx_cadence(task: str, user_id: str | None = None) -> str:
    """parallel|sequential crew cadence from compass sleep pressure.

    Shares the identity-scoped compass with cognition.attention via
    fly_registry. Drowsy crews run sequentially (same coverage, calmer
    cadence). Shadow only logs the would-be choice.
    """
    mode = _flycx_mode()
    if mode not in ("shadow", "live"):
        return "parallel"
    try:
        from cognition.fly_registry import get_flycx, get_flycx_lock
        from cognition.flymemory import text_features
        if user_id is None:
            try:
                from system.userspace import current_user_id
                user_id = current_user_id() or None
            except Exception:
                user_id = None
        cx = get_flycx(user_id)
        if cx is None:
            return "parallel"
        with get_flycx_lock(user_id):
            out = cx.step(text_features(task or ""), fatigue=0.0)
    except Exception as exc:
        log.debug("flycx cadence failed: %s", exp if False else exc)
        return "parallel"
    choice = "sequential" if out["sleep_pressure"] > 0.5 else "parallel"
    log.debug("flycx cadence mode=%s sleep=%.2f -> %s", mode, out["sleep_pressure"], choice)
    return choice if mode == "live" else "parallel"
