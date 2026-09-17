"""Structured deep-think pathway — evidence, recency, contradictions, self-check.

Used only when chat(deep_think=True) or /think is active. Normal chat stays
on the lighter prioritize_memories + format_for_context path.

Design goals (Jetson-safe):
  - no second LLM, no multi-agent debate
  - widen recall is OK if temporal relevance is scored more strictly
  - surface structured evidence tags so memory ≠ fact
  - emit a short UI/brain-trace summary, not raw chain-of-thought
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from system.config import env_float, env_int
from system.log import get_logger

log = get_logger(__name__)

# Half-life for recency decay under deep-think (days). Older memories still
# surface when similarity is high, but recent explicit statements win ties.
DEEP_THINK_RECENCY_HALF_LIFE_DAYS = env_float("DEEP_THINK_RECENCY_HALF_LIFE_DAYS", 21.0)
# Minimum combined score to keep a memory after deep-think rerank (0 = keep all).
DEEP_THINK_EVIDENCE_FLOOR = env_float("DEEP_THINK_EVIDENCE_FLOOR", 0.0)
# Max memories to keep after deep-think rerank (caps contamination).
DEEP_THINK_EVIDENCE_CAP = env_int("DEEP_THINK_EVIDENCE_CAP", 0)  # 0 = no extra cap beyond fetch limit

_OPPOSE_PAIRS = (
    ("like", "dislike"),
    ("love", "hate"),
    ("prefer", "avoid"),
    ("want", "don't want"),
    ("yes", "no"),
    ("always", "never"),
    ("enable", "disable"),
    ("keep", "delete"),
    ("remember", "forget"),
)

# Prompt scaffold — internal stages; the user-facing reply must not dump them.
DEEP_THINK_GUIDE = (
    "<deep_thinking_mode>\n"
    "The user asked for careful, thorough thinking. Work these stages "
    "internally before answering. Never show the stage labels, never pad "
    "the reply to look thorough, and never treat recalled memory as current "
    "fact without checking recency and contradictions.\n"
    "\n"
    "1. Understand — restate the real question, objective, constraints, "
    "known facts, unknowns, and any subproblems worth splitting out.\n"
    "2. Gather evidence — use conversation, memory, knowledge, and (if "
    "present) web/tool results. Tag each claim mentally as MEMORY, "
    "KNOWLEDGE, OBSERVATION, INFERENCE, ASSUMPTION, or UNKNOWN.\n"
    "3. Filter — prefer recent explicit statements over older ones; "
    "downgrade superseded or low-confidence memories; note contradictions "
    "instead of silently picking a side.\n"
    "4. Deliberate — at least two meaningfully different angles or "
    "candidate answers; pressure-test with supporting evidence and what "
    "would falsify each.\n"
    "5. Verify — if a critical unknown is checkable with a tool, web, or "
    "calculation and you lack it, say what is missing rather than inventing.\n"
    "6. Self-check — did you answer the actual question; confuse memory "
    "with fact; use stale information; leave a contradiction unresolved; "
    "need to mark uncertainty?\n"
    "7. Synthesize — one clear answer with caveats only where earned.\n"
    "The reply should read as a considered answer, not a transcript of "
    "these steps.\n"
    "</deep_thinking_mode>"
)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if len(w) > 2}


def _memory_text(row: dict) -> str:
    return str(row.get("memory") or row.get("text") or row.get("trace") or "")


def _parse_created(row: dict) -> datetime | None:
    raw = row.get("created_at") or row.get("timestamp") or row.get("time") or ""
    if not raw:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    s = str(raw).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _recency_weight(row: dict, now: datetime | None = None) -> float:
    """1.0 for brand-new, ~0.5 at half-life, asymptote toward ~0.15."""
    now = now or datetime.now(timezone.utc)
    created = _parse_created(row)
    if created is None:
        return 0.55  # unknown age: mild middle weight
    try:
        age_days = max(0.0, (now - created.astimezone(timezone.utc)).total_seconds() / 86400.0)
    except Exception:
        return 0.55
    half = max(1.0, DEEP_THINK_RECENCY_HALF_LIFE_DAYS)
    return 0.15 + 0.85 * math.exp(-math.log(2) * age_days / half)


def _similarity_proxy(row: dict, query_words: set[str]) -> float:
    text_words = _tokens(_memory_text(row))
    if not query_words or not text_words:
        return float(row.get("_recall_score") or 0.0)
    overlap = len(text_words & query_words)
    base = overlap / max(1, len(query_words))
    try:
        base = max(base, float(row.get("_recall_score") or 0.0))
    except (TypeError, ValueError):
        pass
    return min(1.0, base)


def _polarity_signature(text: str) -> set[str]:
    t = (text or "").lower()
    sig: set[str] = set()
    for a, b in _OPPOSE_PAIRS:
        if a in t:
            sig.add(f"+{a}")
        if b in t:
            sig.add(f"-{a}")
    return sig


def flag_contradictions(memories: list[dict]) -> list[dict]:
    """Mark pairs that share topical tokens but opposing polarity cues."""
    rows = [dict(m) for m in memories]
    for i, a in enumerate(rows):
        ta = _memory_text(a)
        wa = _tokens(ta)
        sa = _polarity_signature(ta)
        if not sa:
            continue
        conflicts = []
        for j, b in enumerate(rows):
            if i == j:
                continue
            tb = _memory_text(b)
            wb = _tokens(tb)
            if len(wa & wb) < 2:
                continue
            sb = _polarity_signature(tb)
            for stem in list(sa):
                key = stem[1:]
                if stem.startswith("+") and f"-{key}" in sb:
                    conflicts.append(j)
                if stem.startswith("-") and f"+{key}" in sb:
                    conflicts.append(j)
        if conflicts:
            a["_contradiction"] = True
            a["_contradiction_with"] = sorted(set(conflicts))[:4]
            a.setdefault("_reconstruction_basis", [])
            if "contradiction" not in a["_reconstruction_basis"]:
                a["_reconstruction_basis"] = list(a.get("_reconstruction_basis") or []) + ["contradiction"]
    return rows


def rerank_for_deep_think(
    memories: list[dict] | None,
    query: str,
) -> tuple[list[dict], dict[str, Any]]:
    """Broader pool in, temporally stricter ranking out.

    Score ≈ similarity × (0.35 + 0.65 × recency) with small bonuses for
    pinned/salient and penalties for superseded / contradiction.
    """
    rows = list(memories or [])
    query_words = _tokens(query)
    now = datetime.now(timezone.utc)
    rows = flag_contradictions(rows)
    scored: list[tuple[float, int, dict]] = []
    for index, row in enumerate(rows):
        sim = _similarity_proxy(row, query_words)
        rec = _recency_weight(row, now)
        score = sim * (0.35 + 0.65 * rec)
        if row.get("pinned"):
            score += 0.15
        if row.get("salience_hit"):
            score += 0.08
        if str(row.get("status") or "").casefold() == "superseded":
            score -= 0.35
        if row.get("_contradiction"):
            score -= 0.12  # keep visible but demote
        try:
            if str(row.get("_reconstruction_confidence") or "") == "high":
                score += 0.05
        except Exception:
            pass
        out = dict(row)
        out["_deep_think_score"] = round(score, 4)
        out["_deep_think_recency"] = round(rec, 4)
        out["_evidence_kind"] = "MEMORY"
        scored.append((score, -index, out))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    ranked = [r for s, _, r in scored if s >= DEEP_THINK_EVIDENCE_FLOOR]
    cap = DEEP_THINK_EVIDENCE_CAP
    if cap and cap > 0:
        ranked = ranked[:cap]
    meta = {
        "input_count": len(rows),
        "kept_count": len(ranked),
        "contradictions": sum(1 for r in ranked if r.get("_contradiction")),
        "half_life_days": DEEP_THINK_RECENCY_HALF_LIFE_DAYS,
    }
    return ranked, meta


def evidence_preamble(
    memories: list[dict] | None,
    knowledge_block: str = "",
    *,
    web_present: bool = False,
) -> str:
    """Compact typed evidence header for the volatile system prompt."""
    mems = list(memories or [])
    n_mem = len(mems)
    n_contra = sum(1 for m in mems if m.get("_contradiction"))
    n_recent = sum(1 for m in mems if float(m.get("_deep_think_recency") or 0) >= 0.7)
    has_knowledge = bool((knowledge_block or "").strip()) and "No matching" not in (knowledge_block or "")
    lines = [
        "<deep_think_evidence>",
        "Evidence classification for this turn (use silently; do not quote tags):",
        f"- MEMORY: {n_mem} recalled item(s)"
        + (f", {n_recent} relatively recent" if n_mem else "")
        + (f", {n_contra} possible contradiction(s) — prefer recent explicit statements" if n_contra else ""),
        f"- KNOWLEDGE: {'present' if has_knowledge else 'none matched'}",
        f"- WEB: {'present this turn' if web_present else 'not fetched this turn'}",
        "- Treat MEMORY as past observation, not guaranteed current preference.",
        "- INFERENCE needs a hedge; ASSUMPTION must not be stated as fact; UNKNOWN must stay open.",
        "</deep_think_evidence>",
    ]
    return "\n".join(lines)


_TOOL_HINT_RE = re.compile(
    r"\b(?:"
    r"latest|current|today|right now|live|look up|search|calculate|compute|"
    r"how many|what time|weather|price|score|news|url|http|"
    r"check (?:the |online )?(?:web|internet|docs?)"
    r")\b",
    re.IGNORECASE,
)


def tool_need_hint(user_input: str, memories: list[dict] | None, knowledge_block: str = "") -> str:
    """Light heuristic: when deep-think should admit a checkable gap."""
    text = user_input or ""
    if not _TOOL_HINT_RE.search(text):
        return ""
    has_knowledge = bool((knowledge_block or "").strip()) and "No matching" not in (knowledge_block or "")
    if memories or has_knowledge:
        if not re.search(r"\b(latest|current|today|right now|live|weather|price|score|news)\b", text, re.I):
            return ""
    return (
        "<deep_think_verify>\n"
        "This question may need live or calculated information you do not "
        "currently have. Prefer stating what is missing over inventing "
        "time-sensitive facts. If the user wanted a tool/web check, say so "
        "plainly rather than guessing.\n"
        "</deep_think_verify>"
    )


def format_summary(
    *,
    query: str,
    memories: list[dict] | None,
    knowledge_block: str = "",
    meta: dict | None = None,
    web_present: bool = False,
) -> str:
    """Short user-visible / sys summary — not raw CoT."""
    mems = list(memories or [])
    meta = meta or {}
    n_mem = len(mems)
    n_contra = sum(1 for m in mems if m.get("_contradiction"))
    has_knowledge = bool((knowledge_block or "").strip()) and "No matching" not in (knowledge_block or "")
    parts = [
        "🧠 Deep Think",
        f"Question: {(query or '')[:120]}{'…' if len(query or '') > 120 else ''}",
        f"Evidence: {n_mem} memor{'y' if n_mem == 1 else 'ies'}"
        + (f", knowledge={'yes' if has_knowledge else 'no'}")
        + (f", web={'yes' if web_present else 'no'}"),
    ]
    if n_contra:
        parts.append(f"⚠ {n_contra} possible memory contradiction(s) — preferred recent statements")
    if meta.get("input_count") and meta.get("kept_count") is not None:
        parts.append(
            f"Recall filter: kept {meta['kept_count']}/{meta['input_count']} "
            f"(recency half-life {meta.get('half_life_days', '?')}d)"
        )
    return "\n".join(parts)


def trace_payload(
    *,
    query: str,
    memories: list[dict] | None,
    knowledge_block: str = "",
    meta: dict | None = None,
    web_present: bool = False,
) -> dict[str, Any]:
    mems = list(memories or [])
    return {
        "query": (query or "")[:200],
        "memory_count": len(mems),
        "contradictions": sum(1 for m in mems if m.get("_contradiction")),
        "knowledge_present": bool((knowledge_block or "").strip())
        and "No matching" not in (knowledge_block or ""),
        "web_present": web_present,
        "rerank": meta or {},
        "top_preview": [
            {
                "score": m.get("_deep_think_score"),
                "recency": m.get("_deep_think_recency"),
                "contradiction": bool(m.get("_contradiction")),
                "text": _memory_text(m)[:120],
            }
            for m in mems[:5]
        ],
    }
