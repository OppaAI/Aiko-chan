"""Conscience Circuit Core — Aiko's moral gate.

Public surface, mirroring how cognition.knowledge and cognition.consolidate
expose their packages:

    from cognition.conscience import conscience_for, maybe_resolve_approval

    verdict = conscience_for(user_id).evaluate(
        act="respond", content=user_input, context={"surface": "chat"},
        llm_client=self._client, embedder=memorize.embedder(),
    )
    if verdict.decision == "refuse":
        ...

Two questions, one floor, one escape hatch:

  Q1 vertical    does the act align with God's revealed will?
  Q2 horizontal  does it do good to the neighbour — every neighbour, not
                 only the one asking?
  L0 safeguards  the industry floor (OWASP LLM, NIST AI RMF, Google/IBM/
                 Microsoft RAI, EU AI Act). Non-overridable.
  L4 HITL        unresolved doubt goes to the human, and a question nobody
                 answers resolves to refuse — never to allow.

Module layout:
    schema      config, Verdict/Norm/Party types, fuse(), decide(), DDL
    guardrails  L0 deterministic safeguards
    canon       L1 moral-norm store and retrieval
    judge       L2 SLM + lexical scorers, L3 deliberation, party enumeration
    core        the ladder, tool policy, HITL wiring
    ledger      audit trail, escalation state, training harvest

Everything degrades: no SLM falls back to the lexical judge, no embedder falls
back to lexical retrieval, no canon file falls back to guardrails alone, and a
circuit exception falls back to caution. Nothing falls back to allow.
"""
from __future__ import annotations

from .canon import CanonStore, get_canon, reload_canon
from .core import ConscienceCircuitCore, conscience_for, maybe_resolve_approval
from .guardrails import scan as scan_guardrails
from .judge import enumerate_parties
from .ledger import ConscienceLedger, close_all, ledger_for
from .schema import (
    ALLOW,
    CAUTION,
    CANON_VERSION,
    ESCALATE,
    REFUSE,
    GuardrailHit,
    Norm,
    Party,
    Verdict,
    decide,
    fuse,
)

__all__ = [
    "ALLOW",
    "CANON_VERSION",
    "CAUTION",
    "CanonStore",
    "ConscienceCircuitCore",
    "ConscienceLedger",
    "ESCALATE",
    "GuardrailHit",
    "Norm",
    "Party",
    "REFUSE",
    "Verdict",
    "close_all",
    "conscience_for",
    "decide",
    "enumerate_parties",
    "fuse",
    "get_canon",
    "ledger_for",
    "maybe_resolve_approval",
    "reload_canon",
    "scan_guardrails",
]
