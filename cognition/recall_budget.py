"""Per-recall token budgets (Mem0 lesson: accuracy under a token budget).

Recall used to be budgeted in *counts* (5 memories + 4 episodic + persona
blob + unbounded codebase/wiki blocks) — token cost unbounded. These caps
bound each source; every trim is logged (debug) with before/after counts so
real distributions can tighten the numbers later.

Estimates use the same word-split heuristic as orchestrate._count_tokens'
fallback (no HTTP): cheap and deterministic. Caps below sum above any sane
turn; a total-budget guard comes once logs show real usage — measure, then
enforce totals.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(key, "") or default))
    except (TypeError, ValueError):
        return default


RECALL_TOKEN_BUDGET_TOTAL = _env_int("RECALL_TOKEN_BUDGET_TOTAL", 1500)
RECALL_TOKEN_CAPS = {
    "persona": _env_int("RECALL_TOKEN_CAP_PERSONA", 300),
    "memory": _env_int("RECALL_TOKEN_CAP_MEMORY", 700),
    "knowledge": _env_int("RECALL_TOKEN_CAP_KNOWLEDGE", 350),
    "lorebook": _env_int("RECALL_TOKEN_CAP_LOREBOOK", 200),
    "codebase": _env_int("RECALL_TOKEN_CAP_CODEBASE", 350),
    "wiki": _env_int("RECALL_TOKEN_CAP_WIKI", 250),
}

TRIM_MARKER = "\n…[trimmed to recall budget]"


def est_tokens(text: str) -> int:
    """Word-split token estimate (matches orchestrate fallback)."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


def fit_block(text: str, label: str) -> str:
    """Cap one recall block to its budget. Returns text unchanged when under."""
    if not text:
        return text
    cap = RECALL_TOKEN_CAPS.get(label)
    if not cap:
        return text
    n = est_tokens(text)
    if n <= cap:
        return text
    # Truncate on word boundary (~4 chars/token), keep a marker so trims
    # are visible in the prompt.
    budget_chars = cap * 4
    cut = text[:budget_chars].rsplit(" ", 1)[0] or text[:budget_chars]
    out = cut + TRIM_MARKER
    log.debug("recall budget: %s trimmed ~%d->~%d tokens", label, n, cap)
    return out


def usage(blocks: dict[str, str]) -> dict[str, int]:
    """Per-source token estimates for tuning (call sites log at debug)."""
    return {label: est_tokens(text) for label, text in blocks.items() if text}
