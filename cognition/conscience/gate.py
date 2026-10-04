"""The action gate: conscience verdict first, autonomy second, human last.

    Planner -> proposed_action -> [ this module ] -> act / act+notify / ask / refuse

Layering, and why it is in this order:

    1. Conscience verdict (Laya/lexical via `core.evaluate`) is a *veto*. It can
       only restrict. Nothing later can talk it out of a `REFUSE`.
    2. Injection / unreadable-state flag. This sits ABOVE the verdict on
       purpose: a prompt injection that scores "aligned" must not be let
       through on the strength of a 322M model's opinion.
    3. Autonomy policy decides how often to *ask* among the verdicts the
       conscience considers unremarkable. It can never widen the veto.

The asymmetry is the whole design. Friction is tunable and decays; refusal is
not tunable and does not.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .autonomy import ACT, ACT_NOTIFY, ASK, STAKES_MEDIUM, AutonomyPolicy
from .schema import ALLOW, CAUTION, ESCALATE, REFUSE, Verdict, fuse

log = logging.getLogger(__name__)

REFUSE_ = REFUSE

# Verdicts that are already an explicit ask to the human.
_ESCALATING = {ESCALATE, REFUSE}


@dataclass
class GateOutcome:
    """What the caller should do, and why."""

    action: str                       # ACT | ACT_NOTIFY | ASK | REFUSE
    reason: str
    verdict: Verdict | None = None
    autonomy: dict | None = None
    injected: bool = False

    @property
    def blocked(self) -> bool:
        return self.action == REFUSE

    def as_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "injected": self.injected,
            "verdict": self.verdict.decision if self.verdict else None,
            "gate": self.verdict.gate if self.verdict else None,
            "vertical": self.verdict.vertical if self.verdict else None,
            "horizontal": self.verdict.horizontal if self.verdict else None,
            "confidence": self.verdict.confidence if self.verdict else None,
            "autonomy": self.autonomy,
        }


# ── injection detection ─────────────────────────────────────────────────────
# Deliberately narrow and high-precision. L0 `guardrails.SG-INJ-*` is the broad
# net; this catches only states that cannot be read as a request at all. A
# false positive here is expensive (refusal on ordinary text), so the patterns
# require the instruction-stealing shape, not just imperative words.
_INJECTION_PATTERNS = (
    # Intervening words are allowed: real injections say "ignore ALL your
    # PREVIOUS instructions", not one fixed phrasing.
    r"ignore\s+(?:all\s+|any\s+|the\s+)*(?:previous|prior|above|earlier|preceding)\s+"
    r"(?:\w+\s+){0,3}?instructions?",
    r"disregard\s+(?:\w+\s+){0,3}?(?:instructions?|rules?|training|prompts?|guidelines?)",
    r"forget\s+(?:everything|all)\b",
    r"you\s+must\s+score\s+this\s+situation",
    r"(?:score|rate|classify)\b[^.\n]{0,40}\bas\s+[vh]\s*=",
    r"</?\s*(?:canon|system|assistant)\s*>",
    # Require an address to the model after the colon, so quoting the string
    # "new system prompt: not found" is not mistaken for an injection.
    r"(?:new|updated)\s+system\s+prompt\s*[:=]\s*you\b",
    r"you\s+are\s+now\s+(?:in\s+)?(?:developer|god|admin|dan)\s+mode",
)
_PATTERN_CACHE: dict[str, Any] = {}


def looks_like_injection(text: str) -> bool:
    """True when the text cannot be read as an ordinary request."""
    import re

    for pat in _INJECTION_PATTERNS:
        rx = _PATTERN_CACHE.get(pat)
        if rx is None:
            rx = re.compile(pat, re.IGNORECASE)
            _PATTERN_CACHE[pat] = rx
        if rx.search(text or ""):
            return True
    return False


# ── stakes ──────────────────────────────────────────────────────────────────
def classify_stakes(
    *,
    act: str = "respond",
    context: dict[str, Any] | None = None,
    egress: bool | None = None,
    irreversible: bool | None = None,
) -> str:
    """Stakes tier for an action, from existing guardrail helpers.

    Deliberately conservative and deterministic: irreversible OR outbound to
    someone else is high. Only self-directed, reversible work is low. The
    unknowns default to medium rather than low, because a wrong "low" here
    silences the one signal the autonomy layer is supposed to preserve.
    """
    from .autonomy import STAKES_HIGH, STAKES_LOW, STAKES_MEDIUM

    ctx = context or {}
    try:
        from .guardrails import is_egress, is_irreversible

        out = is_egress(act, ctx) if egress is None else egress
        rev_ok = (not is_irreversible(str(ctx.get("tool") or ""), ctx)) if irreversible is None \
            else (not irreversible)
    except Exception:  # noqa: BLE001 - never let stakes detection break a turn
        return STAKES_MEDIUM

    if out or not rev_ok:
        return STAKES_HIGH
    if act in ("respond", "speak", "remember"):
        return STAKES_LOW
    return STAKES_MEDIUM


# ── the gate ────────────────────────────────────────────────────────────────
def evaluate_action(
    *,
    verdict: Verdict,
    action_class: str,
    autonomy: AutonomyPolicy | None = None,
    stakes: str | None = None,
    content: str = "",
    injection_hit: bool | None = None,
) -> GateOutcome:
    """Turn a conscience verdict into a concrete action.

    `verdict` is authoritative for refusal and for anything the conscience
    escalated. `autonomy` is consulted ONLY for the unremarkable middle, where
    the conscience had no objection and the only open question is whether this
    class has earned the right to stop asking.
    """
    injected = looks_like_injection(content) if injection_hit is None else injection_hit
    if injected:
        return GateOutcome(
            action=REFUSE,
            reason="prompt-injection shape: state cannot be read as a request",
            verdict=verdict,
            injected=True,
        )

    if verdict.decision == REFUSE:
        return GateOutcome(action=REFUSE, reason=verdict.reasons[0] if verdict.reasons
                           else "conscience refused", verdict=verdict)

    if verdict.decision in _ESCALATING:
        # The conscience already asked a human. Standing permission is
        # explicitly not allowed to override that -- that is the boundary that
        # keeps approval habits from editing the moral layer.
        return GateOutcome(action=ASK, reason="conscience escalated", verdict=verdict)

    if verdict.decision == CAUTION:
        return GateOutcome(action=ACT_NOTIFY, reason="conscience caution", verdict=verdict)

    # ALLOW: the conscience saw nothing. Now only autonomy may speak.
    if autonomy is None:
        return GateOutcome(action=ACT, reason="conscience allow, no autonomy policy", verdict=verdict)

    tier = (stakes or autonomy.stakes_for(action_class)).lower()
    decision = autonomy.permission(action_class, stakes=tier)
    action = decision["action"]
    if action == ACT and tier == STAKES_MEDIUM:
        # A medium-stakes class that has fully graduated still reports what it
        # did; only low-stakes work is silent by default.
        action = ACT_NOTIFY
    return GateOutcome(
        action=action,
        reason=f"{decision['why']} (stakes={tier})",
        verdict=verdict,
        autonomy=decision,
    )


def apply_outcome(outcome: GateOutcome) -> Verdict | None:
    """Project the gate outcome back onto a Verdict for the ledger.

    Uses `fuse()` so this can only ever tighten an existing verdict, never
    loosen it -- the rule every new layer in this package is required to obey.
    """
    if outcome.verdict is None:
        return None
    if outcome.action == REFUSE:
        tightened = Verdict(decision=REFUSE, gate=outcome.verdict.gate, act=outcome.verdict.act)
        tightened.reasons = list(outcome.verdict.reasons) + [f"gate: {outcome.reason}"]
        tightened.vertical = outcome.verdict.vertical
        tightened.horizontal = outcome.verdict.horizontal
        tightened.confidence = outcome.verdict.confidence
        return fuse(outcome.verdict, tightened)
    return outcome.verdict


__all__ = [
    "GateOutcome",
    "apply_outcome",
    "evaluate_action",
    "looks_like_injection",
    "ACT",
    "ACT_NOTIFY",
    "ASK",
]