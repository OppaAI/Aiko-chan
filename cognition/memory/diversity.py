"""Recall diversity filter (MMR-lite) for memory context injection.

Exact-text duplicates are already collapsed upstream
(_MemoryBackend.search, _recent_or_important_memories). This module handles
the remaining case: *near*-duplicates — same motif in different wording —
filling every recall slot on topic-continuous chats (e.g. four episodic
traces + five semantic hits all echoing the same evening).

Greedy, deterministic, dependency-free: walk best-first rows, keep a row
unless its content-token Jaccard similarity against an already-kept row
meets the threshold. Pinned rows and the top hit are always kept, so the
filter can only narrow, never starve, the context.

Pure function over row dicts — no repo imports, safe to use from
attention.prioritize_memories and episode.EpisodicStore.format_for_context.
"""
from __future__ import annotations

import os
import re

_WORD_RE = re.compile(r"[a-z0-9']+")

# Compact stopword set so filler words don't inflate similarity.
_STOP = frozenset("""
a an the and or but if then else when while of at by for with about into
through during before after above below to from up down in out on off over
under again further once here there where why how all any both each few
more most other some such no nor not only own same so than too very can
will just don should now is are was were be been being have has had having
do does did doing would could ought i me my we our you your he him his she
her it its they them their this that these those am as it s t d ll m re ve
""".split())


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP
    )


def _sim_threshold() -> float:
    try:
        value = float(os.getenv("MEMORY_DIVERSITY_SIM", "0.6"))
    except (TypeError, ValueError):
        return 0.6
    return max(0.0, min(1.0, value))


def diversify(
    rows: list[dict] | None,
    *,
    text_of=None,
    sim_threshold: float | None = None,
) -> list[dict]:
    """Drop near-duplicate lower-ranked rows, preserving order.

    rows: pre-sorted best-first. text_of(row) -> str used for comparison
    (defaults to memory/text/trace keys). Returns the kept subset —
    always non-empty when input is non-empty.
    """
    items = list(rows or [])
    if len(items) <= 1:
        return items
    threshold = _sim_threshold() if sim_threshold is None else sim_threshold
    get_text = text_of or (
        lambda r: str(r.get("memory") or r.get("text") or r.get("trace") or "")
    )
    kept: list[dict] = []
    kept_sets: list[frozenset[str]] = []
    for row in items:
        words = _content_tokens(get_text(row))
        if not kept or row.get("pinned"):
            kept.append(row)
            kept_sets.append(words)
            continue
        if words:
            sim = max(
                len(words & prev) / len(words | prev) if (words | prev) else 0.0
                for prev in kept_sets
            )
        else:
            sim = 0.0
        if sim < threshold:
            kept.append(row)
            kept_sets.append(words)
    return kept
