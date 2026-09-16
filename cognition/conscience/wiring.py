"""Import-time wiring for Conscience Circuit Core.

Import this module once at boot (system.wakeup) to patch:
  - AikoThink.route          (CCC approval + respond gate)
  - AikoThink._finalize_response (speak gate)
  - agentic.execute_tool_with_policy (tool gate)

Patches are idempotent and degrade to no-ops if targets are missing.
"""
from __future__ import annotations

import functools
import logging

log = logging.getLogger(__name__)
_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    try:
        _patch_think()
    except Exception as exc:
        log.warning("[ccc-wiring] think patch failed: %s", exp)
    try:
        _patch_agentic()
    except Exception as exc:
        log.warning("[ccc-wiring] agentic patch failed: %s", exp)
