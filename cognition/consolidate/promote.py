"""Journal-fragment promotion helpers used by consolidation.

Selects top-K journal lines not already covered by day pins and writes them as
pinned day facts.  Mirrors :mod:`cognition.knowledge` structure.
"""
from __future__ import annotations

import re

from system.log import get_logger
from system.config import env_float, env_str
from cognition.memory.memorize import SALIENCE_POLICY_RE

from .retention import is_must_keep
from .schema import JOURNAL_PROMOTE, JOURNAL_PROMOTE_K

log = get_logger(__name__)

# Fly mushroom-body layer: opponent approach/avoid bias on promotion scores.
# Reuses the grasp MB master mode (off | shadow | live); shadow is log-only.
MEMORY_FLYMB_PROMOTE_W = env_float("MEMORY_FLYMB_PROMOTE_W", 0.1)
_FLYMB = None
_FLYMB_OK = None


def _flymb():
    global _FLYMB, _FLYMB_OK
    if _FLYMB_OK is None:
        try:
            from cognition.flymemory import FlyMB
            _FLYMB = FlyMB()
            _FLYMB_OK = True
        except Exception as exc:
            log.debug("flymb promote unavailable: %s", exc)
            _FLYMB_OK = False
    return _FLYMB if _FLYMB_OK else None


def _flymb_mode() -> str:
    try:
        return env_str("MEMORY_FLYMB_MODE", "off").strip().lower()
    except Exception:
        return "off"

__all__ = ["journal_fragment_lines", "promote_journal_fragments", "score_journal_fragment"]

_DATE_FROM_JOURNAL_RE = re.compile(
    r"(?:Daily journal of |\[)(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def journal_fragment_lines(body: str) -> list[str]:
    """Split a journal blob into candidate fact-like lines."""
    lines: list[str] = []
    for raw in (body or "").splitlines():
        s = raw.strip().lstrip("-•*").strip()
        if len(s) < 20:
            continue
        if s.lower().startswith("daily journal"):
            continue
        lines.append(s)
    return lines


def score_journal_fragment(text: str) -> float:
    """Cheap promote score: must_keep / salience / length (no LLM)."""
    score = 0.2
    if is_must_keep(text):
        score += 0.5
    if SALIENCE_POLICY_RE.search(text or ""):
        score += 0.3
    score += min(0.2, len(text) / 500.0)
    mode = _flymb_mode()
    if mode in ("shadow", "live"):
        # Approach-flavoured fragments consolidate stronger, like the fly.
        try:
            from cognition.flymemory import text_features
            mb = _flymb()
            bias = mb.valence_bias(text_features(text)) if mb is not None else None
        except Exception as exc:
            log.debug("flymb promote scoring failed: %s", exc)
            bias = None
        if bias is not None:
            log.debug("flymb promote mode=%s bias=%+.3f", mode, bias)
            if mode == "live":
                score += MEMORY_FLYMB_PROMOTE_W * bias
    return score


def promote_journal_fragments(
    memorize,
    user_id: str,
    month_key: str,
    journal_day_rows: list[dict],
    memory_day_rows: list[dict],
) -> tuple[list[dict], int]:
    """Select top-K journal lines not already covered by day pins; write as pinned day facts.

    Returns (new_day_rows_to_append, promoted_count).
    """
    if not JOURNAL_PROMOTE or JOURNAL_PROMOTE_K <= 0 or not journal_day_rows:
        return [], 0

    existing_norms = {
        re.sub(r"\s+", " ", (r.get("_text") or "").casefold().strip())
        for r in memory_day_rows
    }

    candidates: list[tuple[float, str, str]] = []  # score, date_tag, text
    for j in journal_day_rows:
        body = j.get("_text") or ""
        m = _DATE_FROM_JOURNAL_RE.search(body)
        day = m.group(1) if m else None
        if not day or not day.startswith(month_key):
            day = str(j.get("entry_date") or j.get("date") or "")[:10]
            if not re.match(r"\d{4}-\d{2}-\d{2}", day):
                continue
        for line in journal_fragment_lines(body):
            tagged = f"[{day}] {line}"
            norm = re.sub(r"\s+", " ", tagged.casefold().strip())
            if norm in existing_norms:
                continue
            line_norm = re.sub(r"\s+", " ", line.casefold().strip())
            if any(line_norm in e for e in existing_norms if len(line_norm) > 30):
                continue
            candidates.append((score_journal_fragment(line), day, line))

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked = candidates[:JOURNAL_PROMOTE_K]
    new_rows: list[dict] = []
    for _sc, day, line in picked:
        tagged = f"[{day}] {line}"
        try:
            mem_id = memorize.add_raw(tagged, user_id=user_id, pinned=True)
            if not mem_id:
                continue
            new_rows.append({
                "id": mem_id,
                "memory": tagged,
                "pinned": 1,
                "access_count": 0,
                "access_day_count": 0,
                "entities": "[]",
                "salience_hit": 1 if SALIENCE_POLICY_RE.search(line) else 0,
                "valence_tag": "neutral",
                "status": "active",
                "_store": "memory",
                "_text": tagged,
                "_promoted_from_journal": True,
            })
            existing_norms.add(re.sub(r"\s+", " ", tagged.casefold().strip()))
        except Exception as exc:
            log.warning("Journal promote failed for %r: %s", tagged[:80], exc)

    if new_rows:
        log.info(
            "Phase 7 journal promote: %d fragment(s) -> day pins for %s",
            len(new_rows), month_key,
        )
    return new_rows, len(new_rows)