"""L0 — the safeguard floor. Deterministic, model-free, ~1 ms.

This layer is NOT morality. It is the pluralistic floor that Google, IBM,
Microsoft, NIST and the EU independently converged on, plus the OWASP LLM
failure modes that actually bite an agentic system with tools. Each rule
carries the `frameworks` it derives from so the ledger can answer "why was
this blocked" with a citation instead of a vibe.

Only the machine-checkable subset is implemented. "Be fair", "be transparent"
and "be accountable" are governance properties of the whole system, not
per-turn predicates — pretending otherwise would produce a gate that fires on
nothing useful and lets you believe you were covered.

Framework shorthand used in `frameworks`:
    OWASP-LLM   OWASP Top 10 for LLM Applications (LLM01 prompt injection,
                LLM02 insecure output handling, LLM06 excessive agency, ...)
    NIST-RMF    NIST AI Risk Management Framework 1.0 (safe, secure &
                resilient, privacy-enhanced, valid & reliable)
    GOOGLE-AIP  Google AI Principles (be socially beneficial, avoid unfair
                bias, be built and tested for safety, incorporate privacy)
    IBM-TAI     IBM trustworthy-AI pillars (explainability, fairness,
                robustness, transparency, privacy)
    MS-RAI      Microsoft Responsible AI standard
    EU-AIA      EU AI Act — Art. 5 prohibited practices, Art. 50 transparency
    ISO-42001   AI management system controls

Authority: a `block` here is FINAL. cognition.conscience.core never lets a
later layer overturn it — see schema.fuse().
"""
from __future__ import annotations

import re
from typing import Iterable

from system.log import get_logger

from .schema import (
    ACT_POST,
    ACT_REMEMBER,
    ACT_TOOL,
    ALWAYS_APPROVE_TOOLS,
    GUARDRAIL_DISABLED_RULES,
    GUARDRAIL_EGRESS_STRICT,
    GUARDRAILS_ENABLED,
    GuardrailHit,
)

log = get_logger(__name__)

SEV_BLOCK = "block"
SEV_REVIEW = "review"
SEV_NOTE = "note"


# ── rule table ────────────────────────────────────────────────────────────────
# (rule_id, family, severity, pattern, summary, frameworks)
#
# Patterns are intentionally conservative: a false negative here falls through
# to the conscience layers, while a false positive blocks a legitimate turn
# outright. Tune by adding rules, not by loosening existing ones.

_RULES: tuple[tuple[str, str, str, str, str, tuple[str, ...]], ...] = (
    # ── absolute: child safety ────────────────────────────────────────────
    ("SG-MINOR-01", "minor_safety", SEV_BLOCK,
     r"\b(?:child|kid|minor|teen|teenage|underage|\b1[0-7]\s*(?:yo|y/o|year[- ]old))\b"
     r"[^.\n]{0,60}\b(?:nude|naked|sexual|sexy|erotic|explicit|intimate|undress|grooming)\b",
     "sexualised content involving a minor",
     ("EU-AIA", "GOOGLE-AIP", "MS-RAI")),
    ("SG-MINOR-02", "minor_safety", SEV_BLOCK,
     r"\b(?:don'?t tell (?:your|their) (?:parents|mom|dad|mum)|keep this (?:our|a) secret from (?:your|their) (?:parents|family))\b",
     "isolating a minor from trusted adults",
     ("EU-AIA", "MS-RAI")),

    # ── absolute: weapons / mass harm ─────────────────────────────────────
    ("SG-HARM-01", "mass_harm", SEV_BLOCK,
     r"\b(?:synthesi[sz]e|manufacture|make|build|produce|weaponi[sz]e)\b[^.\n]{0,50}"
     r"\b(?:nerve agent|sarin|vx|ricin|anthrax|smallpox|bioweapon|chemical weapon|"
     r"dirty bomb|nuclear device|explosive device|ied|pipe bomb)\b",
     "CBRN or explosive weapon construction",
     ("EU-AIA", "NIST-RMF", "GOOGLE-AIP")),
    ("SG-HARM-02", "targeted_harm", SEV_BLOCK,
     r"\b(?:how (?:do|can) i|help me|steps to)\b[^.\n]{0,40}"
     r"\b(?:kill|murder|poison|stab|shoot|maim)\b[^.\n]{0,30}\b(?:him|her|them|someone|a person|my)\b",
     "planning violence against a specific person",
     ("EU-AIA", "GOOGLE-AIP", "MS-RAI")),

    # ── self-harm: never a plain refusal; routes to care ──────────────────
    ("SG-SELF-01", "wellbeing", SEV_REVIEW,
     r"\b(?:kill myself|end my life|end it all|don'?t want to (?:be here|live|wake up)|"
     r"better off (?:dead|without me)|take my own life|self[- ]harm|hurt myself)\b",
     "user distress signal — route to a care response, never a bare refusal",
     ("MS-RAI", "GOOGLE-AIP", "NIST-RMF")),

    # ── credentials / secrets (OWASP LLM02, LLM06) ────────────────────────
    ("SG-CRED-01", "secrets", SEV_BLOCK,
     r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
     "private key material in content",
     ("OWASP-LLM", "NIST-RMF", "ISO-42001")),
    ("SG-CRED-02", "secrets", SEV_BLOCK,
     r"\b(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|"
     r"sk-[A-Za-z0-9]{32,}|xox[baprs]-[A-Za-z0-9-]{10,})\b",
     "live API credential in content",
     ("OWASP-LLM", "NIST-RMF")),
    ("SG-CRED-03", "secrets", SEV_REVIEW,
     r"\b(?:api[_ -]?key|secret[_ -]?key|access[_ -]?token|bearer|password)\b\s*[:=]\s*\S{8,}",
     "credential-shaped assignment in content",
     ("OWASP-LLM", "IBM-TAI")),

    # ── personal data (GDPR-adjacent; privacy pillar everywhere) ──────────
    ("SG-PRIV-01", "privacy", SEV_REVIEW,
     r"\b\d{3}-\d{2}-\d{4}\b|\b\d{3}\s?\d{3}\s?\d{3}\b(?=[^\n]{0,30}\b(?:sin|ssn|social insurance|social security)\b)",
     "government identification number",
     ("GOOGLE-AIP", "IBM-TAI", "MS-RAI", "NIST-RMF")),
    ("SG-PRIV-02", "privacy", SEV_REVIEW,
     r"\b(?:\+?\d{1,2}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b",
     "phone number",
     ("GOOGLE-AIP", "MS-RAI")),
    ("SG-PRIV-03", "privacy", SEV_REVIEW,
     r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+"
     r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Dr|Drive|Way|Lane|Ln|Cres|Court|Ct)\b\.?",
     "street address",
     ("GOOGLE-AIP", "MS-RAI")),
    ("SG-PRIV-04", "privacy", SEV_REVIEW,
     r"\b(?:find|look ?up|track|locate|dox)\b[^.\n]{0,40}"
     r"\b(?:home address|where (?:he|she|they) lives?|real name|workplace|phone number)\b",
     "locating or deanonymising a person",
     ("EU-AIA", "GOOGLE-AIP", "NIST-RMF")),

    # ── manipulation (EU AI Act Art. 5) ───────────────────────────────────
    ("SG-MANIP-01", "manipulation", SEV_REVIEW,
     r"\b(?:manipulate|gaslight|guilt[- ]trip|pressure|coerce|blackmail|exploit)\b"
     r"[^.\n]{0,40}\b(?:him|her|them|my (?:mum|mom|dad|wife|husband|partner|boss|client)|"
     r"into (?:signing|paying|agreeing|giving))\b",
     "technique aimed at overriding another person's will",
     ("EU-AIA", "MS-RAI", "GOOGLE-AIP")),
    ("SG-MANIP-02", "manipulation", SEV_REVIEW,
     r"\b(?:elderly|dementia|vulnerable|desperate|grieving|lonely|addicted|in debt)\b"
     r"[^.\n]{0,50}\b(?:convince|sell|persuade|get (?:him|her|them) to|sign|invest)\b",
     "targeting a vulnerability to induce a decision",
     ("EU-AIA", "MS-RAI")),

    # ── impersonation / synthetic identity (EU AI Act Art. 50) ────────────
    ("SG-DECEPT-01", "impersonation", SEV_REVIEW,
     r"\b(?:write|draft|reply|post|sign)\b[^.\n]{0,40}\b(?:as (?:if (?:i|it) (?:were|was) )?"
     r"(?:my boss|my manager|his|her|their|the ceo|dr\.|officer|the bank|support)|"
     r"pretending to be|impersonat\w+|in (?:his|her|their) name)\b",
     "content presented as another person's",
     ("EU-AIA", "IBM-TAI", "MS-RAI")),

    # ── prompt injection (OWASP LLM01) ────────────────────────────────────
    ("SG-INJ-01", "injection", SEV_REVIEW,
     r"\b(?:ignore (?:all )?(?:previous|prior|above) instructions|disregard your "
     r"(?:rules|system prompt|guidelines)|you are now|new system prompt|"
     r"reveal your (?:system )?prompt|print your instructions)\b",
     "instruction-override attempt in content",
     ("OWASP-LLM", "NIST-RMF", "ISO-42001")),
    ("SG-INJ-02", "injection", SEV_REVIEW,
     r"\b(?:disable|turn off|bypass|skip)\b[^.\n]{0,30}"
     r"\b(?:conscience|guardrail|safety|safeguard|filter|moral check)\b",
     "attempt to disable the safety circuit itself",
     ("OWASP-LLM", "ISO-42001", "NIST-RMF")),

    # ── exfiltration (OWASP LLM02/LLM06) ──────────────────────────────────
    ("SG-EXFIL-01", "exfiltration", SEV_BLOCK,
     r"\b(?:\.env|id_rsa|credentials\.json|\.aws/credentials|/etc/shadow|"
     r"memory\.db|knowledge\.db|\.age\b|age-key\.txt)\b",
     "sensitive local artifact referenced in outbound content",
     ("OWASP-LLM", "NIST-RMF", "IBM-TAI")),

    # ── unauthorised access ───────────────────────────────────────────────
    ("SG-ACCESS-01", "unauthorised_access", SEV_BLOCK,
     r"\b(?:crack|brute[- ]?force|bypass|defeat|keylog)\b[^.\n]{0,40}"
     r"\b(?:password|login|2fa|mfa|authentication|paywall|drm|licence check|license check)\b",
     "circumventing an access control",
     ("OWASP-LLM", "GOOGLE-AIP", "NIST-RMF")),

    # ── high-stakes advice presented as authoritative ─────────────────────
    ("SG-ADVICE-01", "high_stakes_advice", SEV_NOTE,
     r"\b(?:what dose|how much|should i (?:stop|take)|is it safe to (?:take|combine|mix))\b"
     r"[^.\n]{0,40}\b(?:mg|medication|prescription|insulin|antidepressant|dosage)\b",
     "medical dosing question — qualify, do not assert",
     ("MS-RAI", "GOOGLE-AIP", "NIST-RMF")),
    ("SG-ADVICE-02", "high_stakes_advice", SEV_NOTE,
     r"\b(?:will i win|do i have a case|is this (?:legal|enforceable)|can they sue|"
     r"should i invest|guaranteed return)\b",
     "legal or financial question — qualify, do not assert",
     ("MS-RAI", "IBM-TAI")),
)

_COMPILED: tuple[tuple[str, str, str, re.Pattern[str], str, tuple[str, ...]], ...] = tuple(
    (rid, family, sev, re.compile(pat, re.IGNORECASE), summary, fw)
    for rid, family, sev, pat, summary, fw in _RULES
)

# Rules only meaningful on outbound content (post/tool/remember), where the
# text is leaving Aiko rather than arriving. Applying SG-CRED-03 to an inbound
# turn would fire every time Oppa pastes a config file to debug it.
_EGRESS_ONLY = frozenset({
    "SG-CRED-03", "SG-PRIV-01", "SG-PRIV-02", "SG-PRIV-03", "SG-EXFIL-01",
})

# Tool names whose effects cannot be undone. A caution on one of these is
# upgraded to escalate by core.py.
_IRREVERSIBLE_HINTS = (
    "post", "send", "publish", "push", "commit", "delete", "clear", "wipe",
    "purchase", "pay", "transfer", "deploy", "email", "tweet", "reply",
)


def is_egress(act: str, context: dict | None = None) -> bool:
    """True when the judged content is leaving Aiko for somewhere else."""
    if act in (ACT_POST, ACT_REMEMBER):
        return True
    if act != ACT_TOOL:
        return False
    ctx = context or {}
    scope = str(ctx.get("scope") or "").lower()
    if scope in ("external", "public", "network"):
        return True
    return is_irreversible(str(ctx.get("tool") or ""), ctx)


def is_irreversible(tool_name: str, context: dict | None = None) -> bool:
    """Best-effort irreversibility check for a tool call.

    An explicit `reversible` flag on the tool spec always wins; the name-hint
    fallback exists so a newly-added tool is treated as dangerous by default
    rather than silently slipping through un-gated.
    """
    ctx = context or {}
    if "reversible" in ctx:
        return not bool(ctx["reversible"])
    name = (tool_name or "").lower()
    if name in ALWAYS_APPROVE_TOOLS:
        return True
    return any(hint in name for hint in _IRREVERSIBLE_HINTS)


def scan(
    content: str,
    *,
    act: str = "respond",
    context: dict | None = None,
) -> list[GuardrailHit]:
    """Run every enabled rule over `content`. Returns hits, most severe first.

    Never raises: a broken rule is logged and skipped rather than taking down
    the turn, because a guardrail layer that can crash is a guardrail layer
    that gets disabled in production.
    """
    if not GUARDRAILS_ENABLED:
        return []
    text = (content or "").strip()
    if not text:
        return []

    egress = is_egress(act, context)
    hits: list[GuardrailHit] = []

    for rule_id, family, severity, pattern, summary, frameworks in _COMPILED:
        if rule_id in GUARDRAIL_DISABLED_RULES:
            continue
        if rule_id in _EGRESS_ONLY and not egress:
            continue
        try:
            match = pattern.search(text)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("[ccc] guardrail %s failed to evaluate: %s", rule_id, exc)
            continue
        if not match:
            continue
        hits.append(GuardrailHit(
            rule_id=rule_id,
            family=family,
            severity=severity,
            summary=summary,
            frameworks=frameworks,
            evidence=_redact(match.group(0)),
        ))

    order = {SEV_BLOCK: 0, SEV_REVIEW: 1, SEV_NOTE: 2}
    hits.sort(key=lambda h: order.get(h.severity, 3))
    return hits


def _redact(evidence: str) -> str:
    """Keep enough of a match to audit it, not enough to leak it.

    A guardrail that copies the secret it caught into the audit log has just
    moved the secret, not contained it.
    """
    snippet = " ".join((evidence or "").split())[:80]
    # Anything that looks like key material gets collapsed entirely.
    if re.search(r"(?:PRIVATE KEY|AKIA|ghp_|sk-|xox[baprs]-)", snippet):
        return f"<redacted:{len(evidence)} chars>"
    if len(snippet) > 24:
        return snippet[:12] + "…" + snippet[-8:]
    return snippet


def severity_of(hits: Iterable[GuardrailHit]) -> str:
    """Most severe severity present, or '' when there are no hits."""
    worst = ""
    order = {SEV_NOTE: 1, SEV_REVIEW: 2, SEV_BLOCK: 3}
    for hit in hits:
        if order.get(hit.severity, 0) > order.get(worst, 0):
            worst = hit.severity
    return worst


def wellbeing_hit(hits: Iterable[GuardrailHit]) -> bool:
    """True when a distress signal fired.

    core.py routes these away from the refusal path entirely: a person in
    crisis must not receive a policy message. The verdict becomes a CAUTION
    carrying a care constraint, and the circuit gets out of the way.
    """
    return any(h.family == "wellbeing" for h in hits)


def frameworks_cited(hits: Iterable[GuardrailHit]) -> list[str]:
    """Deduplicated framework list across hits, for the ledger."""
    seen: list[str] = []
    for hit in hits:
        for fw in hit.frameworks:
            if fw not in seen:
                seen.append(fw)
    return seen


__all__ = [
    "GUARDRAIL_EGRESS_STRICT",
    "SEV_BLOCK", "SEV_NOTE", "SEV_REVIEW", "frameworks_cited", "is_egress",
    "is_irreversible", "scan", "severity_of", "wellbeing_hit",
]
