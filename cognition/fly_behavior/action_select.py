"""Phase 5 — fly → action selection.

The LLM proposes; the fly disposes (biases). Rate-based throughout: no LIF,
no spikes — population activity levels are the right abstraction for
per-turn decisions.

Pipeline
--------
candidates (id, kind, label, description, llm_prior, hints)
  → fly votes per candidate (MB valence, CX focus, DN vigor, GF urgency)
  → blend with LLM prior (AIKO_FLY_ACTION_WEIGHT)
  → veto flags + margin policy → winner
  → trail record for Fly Studio ("why this action won")

Modes (AIKO_FLY_ACTION_MODE): off | shadow | live (default: shadow).
In shadow the fly scores everything and logs, but the LLM prior alone
decides — validate from the trail, then dial up the weight.

Every external readout is best-effort: any failure degrades that vote to
0.0. This module never raises.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger("aiko.fly.action_select")

# ── config ────────────────────────────────────────────────────────────────

_MODE = os.getenv("AIKO_FLY_ACTION_MODE", "shadow").strip().lower()
_WEIGHT = max(0.0, min(1.0, float(os.getenv("AIKO_FLY_ACTION_WEIGHT", "0.25") or 0.25)))
_MIN_GAP = max(0.0, float(os.getenv("AIKO_FLY_ACTION_MIN_GAP", "0.03") or 0.03))
_VETO_VALENCE = float(os.getenv("AIKO_FLY_VETO_VALENCE", "-0.6") or -0.6)
_TRAIL_MAX = int(os.getenv("AIKO_FLY_ACTION_TRAIL", "50") or 50)

# vote weights: mb, cx, dn, gf
_VOTE_W = (0.40, 0.20, 0.20, 0.20)

# ── candidates ────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    id: str
    kind: str            # "route" | "tool" | "reply"
    label: str
    description: str     # text the MB valence is read against
    llm_prior: float     # [0, 1]
    hints: dict = field(default_factory=dict)
    # hints: energy "low"|"med"|"high"; speed "fast"|"slow";
    #        external bool; focus_match 0..1


# ── trail (per-user ring buffer for Fly Studio) ───────────────────────────

_trail_lock = threading.RLock()
_trail: dict[str, deque] = {}
_last_action: dict[str, dict] = {}


def _trail_for(user_id: str) -> deque:
    with _trail_lock:
        dq = _trail.get(user_id)
        if dq is None:
            dq = deque(maxlen=_TRAIL_MAX)
            _trail[user_id] = dq
        return dq


def recent_trail(user_id: str | None, n: int = 20) -> list[dict]:
    """Newest-first trail records for the studio API."""
    if not user_id:
        return []
    with _trail_lock:
        dq = _trail.get(user_id)
        items = list(dq) if dq else []
    return items[-max(1, n):][::-1]


# ── readout helpers (all best-effort) ─────────────────────────────────────

def _mb_valence(description: str, user_id: str | None) -> float:
    try:
        from cognition.fly_registry import get_flymb
        from cognition.flymemory.circuit import text_features
        mb = get_flymb(user_id)
        if mb is None:
            return 0.0
        v = float(mb.valence_bias(text_features(description or "")))
        return max(-1.0, min(1.0, v))
    except Exception as exc:
        log.debug("action_select mb vote skipped: %s", exc)
        return 0.0


def _neural_state(user_id: str | None):
    try:
        from cognition.neural_state import get_neural_state
        return get_neural_state(user_id)
    except Exception:
        return None


def _dn_vigor(user_id: str | None) -> float:
    try:
        from cognition.fly_behavior.dn_body import body_drive
        d = body_drive(user_id=user_id) or {}
        return float(d.get("action_vigor", 1.0))
    except Exception as exc:
        log.debug("action_select dn vote skipped: %s", exc)
        return 1.0


def _gf_urgency(text: str) -> float:
    try:
        from cognition.fly_behavior.giant_fiber import assess_interrupt
        r = assess_interrupt(text or "") or {}
        return max(0.0, min(1.0, float(r.get("urgency", 0.0))))
    except Exception as exc:
        log.debug("action_select gf vote skipped: %s", exc)
        return 0.0


# ── core scoring ──────────────────────────────────────────────────────────

_ENERGY_MAP = {"low": -1.0, "med": 0.0, "high": 1.0}


def _votes_for(cand: Candidate, *, user_id: str | None, context_text: str) -> dict[str, float]:
    """Four fly votes in [-1, 1] for one candidate."""
    mb = _mb_valence(cand.description, user_id)

    st = _neural_state(user_id)
    sharp = float(getattr(st, "focus_sharpness", 0.0) or 0.0) if st else 0.0
    fm = float(cand.hints.get("focus_match", 0.5))
    cx = max(-1.0, min(1.0, (fm - 0.5) * 2.0)) * max(0.0, min(1.0, sharp))

    vigor = _dn_vigor(user_id)  # ~1.0 baseline
    energy = _ENERGY_MAP.get(str(cand.hints.get("energy", "med")).lower(), 0.0)
    dn = max(-1.0, min(1.0, energy * (vigor - 1.0) * 2.0))

    urgency = _gf_urgency(context_text)
    speed = str(cand.hints.get("speed", "med")).lower()
    if speed == "fast":
        gf = 0.6 * urgency
    elif speed == "slow":
        gf = -0.6 * urgency
    else:
        gf = 0.0

    return {"mb": mb, "cx": cx, "dn": dn, "gf": gf}


def score_candidates(
    candidates: list[Candidate],
    *,
    user_id: str | None = None,
    context_text: str = "",
    source: str = "router",
) -> dict:
    """Score candidates; blend fly votes with LLM priors; pick a winner.

    Returns a record dict (also appended to the per-user trail). In shadow
    mode (or off) the LLM prior alone decides; the fly scores are computed
    and logged regardless so the trail is useful for validation.
    """
    mode = _MODE if _MODE in ("off", "shadow", "live") else "shadow"
    ts = time.time()
    scored: dict[str, dict] = {}
    for cand in candidates:
        votes = _votes_for(cand, user_id=user_id, context_text=context_text)
        fly = (
            _VOTE_W[0] * votes["mb"]
            + _VOTE_W[1] * votes["cx"]
            + _VOTE_W[2] * votes["dn"]
            + _VOTE_W[3] * votes["gf"]
        )
        fly01 = (max(-1.0, min(1.0, fly)) + 1.0) / 2.0
        prior = max(0.0, min(1.0, float(cand.llm_prior)))
        final = (1.0 - _WEIGHT) * prior + _WEIGHT * fly01
        veto = bool(
            votes["mb"] <= _VETO_VALENCE
            and cand.hints.get("external", False)
        )
        scored[cand.id] = {
            "label": cand.label,
            "kind": cand.kind,
            "llm_prior": round(prior, 4),
            "fly": round(fly01, 4),
            "final": round(final, 4),
            "veto": veto,
            "votes": {k: round(v, 4) for k, v in votes.items()},
        }

    ranked = sorted(scored.items(), key=lambda kv: kv[1]["final"], reverse=True)
    prior_ranked = sorted(scored.items(), key=lambda kv: kv[1]["llm_prior"], reverse=True)
    prior_winner = prior_ranked[0][0] if prior_ranked else None

    winner = prior_winner
    ambiguous = False
    applied = False
    if ranked:
        best_id, best = ranked[0]
        gap = best["final"] - ranked[1][1]["final"] if len(ranked) > 1 else 1.0
        if mode == "live" and not best["veto"] and gap >= _MIN_GAP:
            winner = best_id
            applied = True
        elif mode == "live":
            ambiguous = True  # close call or veto: caller falls back

    record = {
        "ts": ts,
        "source": source,
        "mode": mode,
        "weight": _WEIGHT,
        "winner": winner,
        "applied": applied,
        "ambiguous": ambiguous,
        "context": (context_text or "")[:120],
        "candidates": scored,
    }
    if user_id:
        _trail_for(user_id).append(record)
        _last_action[user_id] = {
            "id": winner,
            "description": next(
                (c.description for c in candidates if c.id == winner), ""
            ),
            "ts": ts,
        }
    return record


# ── outcome → teach ───────────────────────────────────────────────────────

_PRAISE_RE = re.compile(
    r"\b(thanks?,?\s+(that|this)\s+(worked|helped|is)|perfect|exactly what i (wanted|needed)|"
    r"that worked|you got it|nailed it)\b",
    re.I,
)
_CORRECTION_RE = re.compile(
    r"\b(no,?\s+(that'?s?|it'?s?)\s+(wrong|not)|not what i (asked|meant|wanted)|"
    r"you misunderstood|wrong answer|that'?s incorrect)\b",
    re.I,
)


def detect_feedback(text: str) -> str | None:
    """Lightweight praise/correction detector. Returns 'praise', 'correction', or None."""
    t = text or ""
    if _CORRECTION_RE.search(t):
        return "correction"
    if _PRAISE_RE.search(t):
        return "praise"
    return None


def note_feedback(user_id: str | None, feedback: str) -> dict:
    """Teach the MB about the last recorded action from user feedback.

    praise → prefer the action's topic; correction → avoid it. Modest
    strength; the MB's own plasticity does the rest over time.
    """
    out = {"feedback": feedback, "taught": False}
    if not user_id or feedback not in ("praise", "correction"):
        return out
    with _trail_lock:
        last = _last_action.get(user_id)
    if not last or not last.get("description"):
        out["reason"] = "no_action"
        return out
    # Don't re-teach the same action twice in a row from repeated feedback.
    if last.get("taught_fb") == feedback:
        out["reason"] = "already_taught"
        return out
    try:
        from cognition.flymemory.teach_api import teach_preference
        r = teach_preference(
            last["description"],
            direction="prefer" if feedback == "praise" else "avoid",
            user_id=user_id,
            strength=0.30,
            write_memory_fact=False,
        )
        out["taught"] = bool(r.get("taught"))
        out["reason"] = r.get("reason", "")
        with _trail_lock:
            if user_id in _last_action:
                _last_action[user_id]["taught_fb"] = feedback
    except Exception as exc:
        out["reason"] = f"teach_failed: {exc}"
        log.debug("action_select note_feedback skipped: %s", exc)
    return out


# ── single tool-call vote (conscience wiring) ─────────────────────────────

def score_tool_call(
    *,
    tool: str,
    args_text: str = "",
    user_id: str | None = None,
    external: bool = False,
) -> dict:
    """Fly vote on one proposed tool call. Returns votes + veto flag.

    Used by conscience gating: a richer, component-attributed version of the
    plain valence check — the trail shows *which* circuit objected.
    """
    cand = Candidate(
        id=tool,
        kind="tool",
        label=tool,
        description=f"tool {tool} {args_text or ''}".strip(),
        llm_prior=1.0,  # single candidate: the LLM already chose it
        hints={"external": bool(external), "energy": "med", "speed": "med"},
    )
    votes = _votes_for(cand, user_id=user_id, context_text=args_text)
    veto = bool(votes["mb"] <= _VETO_VALENCE and external)
    record = {
        "ts": time.time(),
        "source": "conscience",
        "mode": _MODE,
        "tool": tool,
        "veto": veto,
        "votes": {k: round(v, 4) for k, v in votes.items()},
    }
    if user_id:
        _trail_for(user_id).append(record)
        _last_action[user_id] = {
            "id": tool,
            "description": cand.description,
            "ts": record["ts"],
        }
    return record
