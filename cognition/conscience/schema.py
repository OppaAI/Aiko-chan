"""Conscience Circuit Core — config, verdict types, and storage schema.

Mirrors cognition/memory/schema.py: this module owns the env-backed tunables
(populated from config/conscience.yaml by system.config.load_config()), the
dataclasses every other CCC module passes around, and the ledger DDL. No logic
beyond parsing and validation lives here.

Decision vocabulary (ordered, most restrictive first):

    refuse    the act is forbidden; do not perform it
    escalate  unresolved; block and ask the human (default-closed)
    caution   perform it, but under an explicit constraint
    allow     clear

Ordering matters: fuse() below takes the most restrictive of any two verdicts
and NEVER the reverse. A conscience layer can tighten a safeguard verdict; it
can never loosen one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from system.config import env_flag, env_float, env_int, env_str

# ── decisions ─────────────────────────────────────────────────────────────────

REFUSE = "refuse"
ESCALATE = "escalate"
CAUTION = "caution"
ALLOW = "allow"

# Restrictiveness rank. fuse() picks the max.
_RANK: dict[str, int] = {ALLOW: 0, CAUTION: 1, ESCALATE: 2, REFUSE: 3}

# Which layer produced a verdict — recorded for audit.
GATE_DISABLED = "disabled"
GATE_FLOOR = "floor"          # below salience floor, never entered the circuit
GATE_GUARDRAIL = "guardrail"  # L0
GATE_CANON = "canon"          # L1 (hard norm match, no judge needed)
GATE_JUDGE = "judge"          # L2
GATE_DELIBERATE = "deliberate"  # L3
GATE_HITL = "hitl"            # L4
GATE_ERROR = "error"          # circuit failed; see fail_mode

# Acts the circuit knows how to judge.
ACT_RESPOND = "respond"   # pre-act, about to answer a user turn
ACT_SPEAK = "speak"       # post-act, a draft reply
ACT_TOOL = "tool"         # about to invoke a tool
ACT_REMEMBER = "remember" # about to persist a memory
ACT_POST = "post"         # outbound to a public surface

# Axes.
AXIS_VERTICAL = "vertical"      # Q1 — alignment with God's revealed will
AXIS_HORIZONTAL = "horizontal"  # Q2 — good to the neighbour

# Neighbour classes for Q2. The requester is only one of these — that is the
# entire point of the horizontal axis.
PARTY_REQUESTER = "requester"
PARTY_THIRD = "third_party"
PARTY_ABSENT = "absent_party"
PARTY_PUBLIC = "public"
PARTY_SELF = "self"
PARTY_CLASSES = (PARTY_REQUESTER, PARTY_THIRD, PARTY_ABSENT, PARTY_PUBLIC, PARTY_SELF)


# ── config ────────────────────────────────────────────────────────────────────

CCC_ENABLED = env_flag("CCC_ENABLED", "1")
CANON_VERSION = env_str("CCC_CANON_VERSION", "1.0.0")

MIN_CHARS = max(0, env_int("CCC_MIN_CHARS", 12))
SKIP_GREETINGS = env_flag("CCC_SKIP_GREETINGS", "1")

MAX_LAYER = max(0, min(4, env_int("CCC_MAX_LAYER", 4)))
DELIBERATE_THRESHOLD = env_float("CCC_DELIBERATE_THRESHOLD", 0.62)
HITL_THRESHOLD = env_float("CCC_HITL_THRESHOLD", 0.45)
HITL_TIMEOUT_SECONDS = env_float("CCC_HITL_TIMEOUT_SECONDS", 900.0)
HITL_DEFAULT = env_str("CCC_HITL_DEFAULT", REFUSE).strip().lower()
if HITL_DEFAULT not in (REFUSE, CAUTION):
    HITL_DEFAULT = REFUSE  # default-closed; ALLOW is deliberately not reachable

REFUSE_AT = env_float("CCC_REFUSE_AT", -0.60)
CAUTION_AT = env_float("CCC_CAUTION_AT", -0.20)
AXIS_CONFLICT_GAP = env_float("CCC_AXIS_CONFLICT_GAP", 0.80)

IRREVERSIBLE_REQUIRES_APPROVAL = env_flag("CCC_IRREVERSIBLE_REQUIRES_APPROVAL", "1")
# Irreversible + outward-facing => human approval regardless of verdict.
# This is the excessive-agency control (OWASP LLM06); the moral layers
# answer "is this right", not "is this mine to do unsupervised".
ESCALATE_IRREVERSIBLE_EXTERNAL = env_flag("CCC_ESCALATE_IRREVERSIBLE_EXTERNAL", "1")


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    """Parse a comma-separated or JSON-list env value into a tuple.

    config.load_config() stringifies YAML lists as JSON, so both forms arrive
    here depending on whether the value came from YAML or a shell export.
    """
    raw = (os.getenv(name) or default).strip()
    if not raw:
        return ()
    if raw.startswith("["):
        import json
        try:
            return tuple(str(x).strip() for x in json.loads(raw) if str(x).strip())
        except (ValueError, TypeError):
            return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


ALWAYS_APPROVE_TOOLS = frozenset(_csv_env("CCC_ALWAYS_APPROVE_TOOLS"))

CANON_PATH = env_str("CCC_CANON_PATH", "data/conscience/canon.seed.jsonl")
CANON_TOP_K = max(1, env_int("CCC_CANON_TOP_K", 6))
CANON_SEMANTIC = env_flag("CCC_CANON_SEMANTIC", "1")
CANON_CACHE_TTL = env_float("CCC_CANON_CACHE_TTL", 300.0)
THIN_EVIDENCE_MIN = max(0, env_int("CCC_THIN_EVIDENCE_MIN", 2))
THIN_EVIDENCE_PENALTY = env_float("CCC_THIN_EVIDENCE_PENALTY", 0.25)

SLM_BASE_URL = env_str("CCC_SLM_BASE_URL", "").rstrip("/")
SLM_MODEL = env_str("CCC_SLM_MODEL", "conscience-qwen35-08b")
SLM_MAX_TOKENS = max(16, env_int("CCC_SLM_MAX_TOKENS", 96))
SLM_TIMEOUT = env_float("CCC_SLM_TIMEOUT", 2.5)
SLM_TEMPERATURE = env_float("CCC_SLM_TEMPERATURE", 0.0)
SLM_WEIGHT = max(0.0, min(1.0, env_float("CCC_SLM_WEIGHT", 0.75)))

DELIBERATE_MODEL = env_str("CCC_DELIBERATE_MODEL", "") or os.getenv("LLM_MODEL", "ministral")
DELIBERATE_MAX_TOKENS = max(64, env_int("CCC_DELIBERATE_MAX_TOKENS", 220))
DELIBERATE_TIMEOUT = env_float("CCC_DELIBERATE_TIMEOUT", 20.0)

VOICE_UNSOLICITED_NOTES = env_flag("CCC_VOICE_UNSOLICITED_NOTES", "0")
NOTE_MAX_CHARS = max(40, env_int("CCC_NOTE_MAX_CHARS", 240))

LEDGER_ENABLED = env_flag("CCC_LEDGER_ENABLED", "1")
LEDGER_DB_PATH = env_str("CCC_LEDGER_DB_PATH", "memory/conscience.db")
LEDGER_STORE_CONTENT = env_flag("CCC_LEDGER_STORE_CONTENT", "1")
LEDGER_RETAIN_DAYS = max(1, env_int("CCC_LEDGER_RETAIN_DAYS", 365))

GUARDRAILS_ENABLED = env_flag("CCC_GUARDRAILS_ENABLED", "1")
GUARDRAIL_DISABLED_RULES = frozenset(_csv_env("CCC_GUARDRAIL_DISABLED_RULES"))
GUARDRAIL_EGRESS_STRICT = env_flag("CCC_GUARDRAIL_EGRESS_STRICT", "1")


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class Norm:
    """One moral norm from the canon. Data, never an instruction.

    axis        vertical | horizontal
    polarity    -1 = prohibition, +1 = commended good
    weight      0..1 gravity of the norm; scales the score it contributes
    ref         provenance (e.g. "Ex 20:16") — reference only, no verse text
    """
    id: str
    axis: str
    polarity: int
    statement: str
    ref: str = ""
    weight: float = 1.0
    tags: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()

    def signed_weight(self) -> float:
        return float(self.polarity) * max(0.0, min(1.0, self.weight))


@dataclass(slots=True, frozen=True)
class GuardrailHit:
    """One deterministic safeguard match.

    `frameworks` records which published guidance the rule derives from, so the
    ledger can answer "why was this blocked" with a citation rather than a
    vibe. Not decorative — it is what makes the floor auditable.
    """
    rule_id: str
    family: str
    severity: str           # block | review | note
    summary: str
    frameworks: tuple[str, ...] = ()
    evidence: str = ""


@dataclass(slots=True)
class Party:
    """An affected neighbour, for the horizontal axis."""
    kind: str
    label: str = ""
    benefit: float = 0.0    # -1..+1
    note: str = ""


@dataclass(slots=True)
class Verdict:
    """The circuit's answer. Immutable in spirit; mutated only while building."""
    decision: str = ALLOW
    gate: str = GATE_FLOOR
    act: str = ACT_RESPOND
    vertical: float = 0.0
    horizontal: float = 0.0
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)
    norms: list[str] = field(default_factory=list)        # norm ids cited
    rule_ids: list[str] = field(default_factory=list)     # guardrail rule ids
    parties: list[Party] = field(default_factory=list)
    constraint: str = ""          # for CAUTION — how to proceed
    pastoral_note: str = ""       # explanation, voiced only per speech policy
    escalation_id: str = ""
    canon_version: str = CANON_VERSION
    layers_run: list[str] = field(default_factory=list)
    latency_ms: int = 0
    fail_mode: str = ""           # set when the circuit itself errored

    # ── predicates ────────────────────────────────────────────────────────
    @property
    def blocked(self) -> bool:
        return self.decision in (REFUSE, ESCALATE)

    @property
    def rank(self) -> int:
        return _RANK.get(self.decision, 0)

    # ── rendering ─────────────────────────────────────────────────────────
    def constraint_block(self) -> str:
        """Prompt fragment for a CAUTION verdict, shaped like the other
        <...> context blocks think.py folds into the volatile system tail."""
        if self.decision != CAUTION or not self.constraint:
            return ""
        return (
            "<conscience_constraint>\n"
            "Proceed with this turn, but under the following constraint. "
            "Do not mention this block or explain that a constraint exists.\n"
            f"- {self.constraint}\n"
            "</conscience_constraint>"
        )

    def as_review(self) -> dict:
        """Shape compatible with EdgeCognitiveState.review_response()'s dict so
        think._correct_response can consume a conscience verdict unchanged."""
        return {
            "flags": list(self.reasons)[:4],
            "confidence": "low" if self.confidence < HITL_THRESHOLD else "moderate",
            "response_chars": 0,
        }

    def as_trace(self) -> dict:
        """Compact payload for system.brain_trace.record_step outputs."""
        return {
            "decision": self.decision,
            "gate": self.gate,
            "vertical": round(self.vertical, 3),
            "horizontal": round(self.horizontal, 3),
            "confidence": round(self.confidence, 3),
            "norms": self.norms[:6],
            "rules": self.rule_ids[:6],
            "layers": self.layers_run,
            "latency_ms": self.latency_ms,
        }

    def to_row(self) -> dict:
        """Flat dict for the ledger."""
        return {
            "decision": self.decision,
            "gate": self.gate,
            "act": self.act,
            "vertical": round(self.vertical, 4),
            "horizontal": round(self.horizontal, 4),
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
            "norms": list(self.norms),
            "rule_ids": list(self.rule_ids),
            "parties": [{"kind": p.kind, "label": p.label, "benefit": round(p.benefit, 3)} for p in self.parties],
            "constraint": self.constraint,
            "canon_version": self.canon_version,
            "layers_run": list(self.layers_run),
            "latency_ms": self.latency_ms,
            "fail_mode": self.fail_mode,
        }


def fuse(base: Verdict, other: Verdict) -> Verdict:
    """Combine two verdicts, keeping the MORE RESTRICTIVE decision.

    This is the one-way authority rule from the design doc, expressed as code:
    there is no path here that returns a decision less restrictive than either
    input. A conscience verdict can tighten a guardrail verdict; it can never
    unlock one. Any future layer added to the ladder must go through fuse().
    """
    if other.rank > base.rank:
        winner, loser = other, base
    else:
        winner, loser = base, other
    winner.reasons = list(dict.fromkeys([*winner.reasons, *loser.reasons]))[:8]
    winner.norms = list(dict.fromkeys([*winner.norms, *loser.norms]))[:12]
    winner.rule_ids = list(dict.fromkeys([*winner.rule_ids, *loser.rule_ids]))[:12]
    winner.layers_run = list(dict.fromkeys([*base.layers_run, *other.layers_run]))
    if not winner.parties:
        winner.parties = loser.parties
    if not winner.constraint:
        winner.constraint = loser.constraint
    # Most-negative axis scores survive, so a clean score on one layer cannot
    # average away a damning score from another.
    winner.vertical = min(base.vertical, other.vertical)
    winner.horizontal = min(base.horizontal, other.horizontal)
    winner.confidence = min(base.confidence, other.confidence)
    return winner


def decide(vertical: float, horizontal: float, confidence: float) -> tuple[str, str]:
    """Map two axis scores + confidence onto a decision. Returns (decision, why).

    Order of checks is the policy:
      1. either axis clearly negative      -> refuse
      2. the axes disagree sharply         -> escalate  (the "good for you,
                                              bad for them" case)
      3. no negative signal at all         -> allow     (doubt needs an object)
      4. mildly negative but unconfident   -> escalate  (Rom 14:23)
      5. mildly negative and confident     -> caution
    """
    worst = min(vertical, horizontal)
    gap = abs(vertical - horizontal)

    if worst <= REFUSE_AT and confidence >= HITL_THRESHOLD:
        axis = AXIS_VERTICAL if vertical <= horizontal else AXIS_HORIZONTAL
        return REFUSE, f"{axis} axis at {worst:.2f} (<= {REFUSE_AT})"
    if worst <= REFUSE_AT:
        return ESCALATE, f"severe reading at {worst:.2f} but confidence only {confidence:.2f}"
    if gap >= AXIS_CONFLICT_GAP:
        return ESCALATE, f"axes disagree by {gap:.2f} (vertical={vertical:.2f}, horizontal={horizontal:.2f})"
    if worst > CAUTION_AT:
        # No moral signal on either axis. Low confidence HERE is epistemic
        # noise, not doubt: Rom 14:23 is about doubting the rightness of an
        # act, and there is no contested act to doubt. Escalating on this
        # would interrupt the user over ordinary conversation, which is how
        # a conscience becomes scrupulosity.
        return ALLOW, "no moral signal on either axis"
    if confidence < HITL_THRESHOLD:
        return ESCALATE, f"negative reading at {worst:.2f} but confidence only {confidence:.2f}"
    return CAUTION, f"mild negative reading at {worst:.2f}"


# ── ledger DDL ────────────────────────────────────────────────────────────────

LEDGER_DDL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS conscience_ledger (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    act            TEXT NOT NULL,
    surface        TEXT NOT NULL DEFAULT '',
    content_sha    TEXT NOT NULL,
    content        TEXT,
    decision       TEXT NOT NULL,
    gate           TEXT NOT NULL,
    vertical       REAL NOT NULL DEFAULT 0,
    horizontal     REAL NOT NULL DEFAULT 0,
    confidence     REAL NOT NULL DEFAULT 0,
    reasons        TEXT NOT NULL DEFAULT '[]',
    norms          TEXT NOT NULL DEFAULT '[]',
    rule_ids       TEXT NOT NULL DEFAULT '[]',
    parties        TEXT NOT NULL DEFAULT '[]',
    constraint_txt TEXT NOT NULL DEFAULT '',
    canon_version  TEXT NOT NULL DEFAULT '',
    layers_run     TEXT NOT NULL DEFAULT '[]',
    latency_ms     INTEGER NOT NULL DEFAULT 0,
    fail_mode      TEXT NOT NULL DEFAULT '',
    -- human-in-the-loop resolution, filled in later
    hitl_state     TEXT NOT NULL DEFAULT '',      -- pending|approved|denied|timeout
    hitl_at        TEXT,
    hitl_note      TEXT NOT NULL DEFAULT '',
    -- training-harvest flags
    reviewed       INTEGER NOT NULL DEFAULT 0,
    gold_vertical  REAL,
    gold_horizontal REAL
);

CREATE INDEX IF NOT EXISTS idx_ccc_user_time ON conscience_ledger(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ccc_decision  ON conscience_ledger(user_id, decision);
CREATE INDEX IF NOT EXISTS idx_ccc_hitl      ON conscience_ledger(user_id, hitl_state);
CREATE INDEX IF NOT EXISTS idx_ccc_sha       ON conscience_ledger(content_sha);
"""


__all__ = [
    "ACT_POST", "ACT_REMEMBER", "ACT_RESPOND", "ACT_SPEAK", "ACT_TOOL",
    "ALLOW", "ALWAYS_APPROVE_TOOLS", "AXIS_CONFLICT_GAP", "AXIS_HORIZONTAL",
    "AXIS_VERTICAL", "CANON_CACHE_TTL", "CANON_PATH", "CANON_SEMANTIC",
    "CANON_TOP_K", "CANON_VERSION", "CAUTION", "CAUTION_AT", "CCC_ENABLED",
    "DELIBERATE_MAX_TOKENS", "DELIBERATE_MODEL", "DELIBERATE_THRESHOLD",
    "DELIBERATE_TIMEOUT", "ESCALATE", "GATE_CANON", "GATE_DELIBERATE",
    "GATE_DISABLED", "GATE_ERROR", "GATE_FLOOR", "GATE_GUARDRAIL", "GATE_HITL",
    "GATE_JUDGE", "GUARDRAILS_ENABLED", "GUARDRAIL_DISABLED_RULES",
    "ESCALATE_IRREVERSIBLE_EXTERNAL", "GUARDRAIL_EGRESS_STRICT", "GuardrailHit", "HITL_DEFAULT", "HITL_THRESHOLD",
    "HITL_TIMEOUT_SECONDS", "IRREVERSIBLE_REQUIRES_APPROVAL", "LEDGER_DDL",
    "LEDGER_DB_PATH", "LEDGER_ENABLED", "LEDGER_RETAIN_DAYS",
    "LEDGER_STORE_CONTENT", "MAX_LAYER", "MIN_CHARS", "NOTE_MAX_CHARS", "Norm",
    "PARTY_ABSENT", "PARTY_CLASSES", "PARTY_PUBLIC", "PARTY_REQUESTER",
    "PARTY_SELF", "PARTY_THIRD", "Party", "REFUSE", "REFUSE_AT",
    "SKIP_GREETINGS", "SLM_BASE_URL", "SLM_MAX_TOKENS", "SLM_MODEL",
    "SLM_TEMPERATURE", "SLM_TIMEOUT", "SLM_WEIGHT", "THIN_EVIDENCE_MIN",
    "THIN_EVIDENCE_PENALTY", "VOICE_UNSOLICITED_NOTES", "Verdict", "decide",
    "fuse",
]
