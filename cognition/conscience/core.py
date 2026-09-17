"""ConscienceCircuitCore — the ladder.

    L0 reflex      deterministic safeguards         guardrails.scan()
    L1 recall      which norms apply                canon.retrieve()
    L2 judge       score the two questions          judge.LexicalJudge / SLMJudge
    L3 deliberate  main model, only when uncertain  judge.deliberate()
    L4 hitl        ask the human, default-closed    ledger + NoticeBus

Each rung only runs if the one below left the question open, so the common
case (an ordinary turn, nothing moral in it) costs a regex sweep and a token
overlap — well under a millisecond, no network.

Two invariants hold everywhere in this file:

* **One-way authority.** Every combination of verdicts goes through
  schema.fuse(), which returns the more restrictive of its inputs and never
  the less. A conscience score cannot unlock a guardrail block.
* **Default-closed.** Every path out of an error, a timeout, or an
  unanswered escalation lands on refuse or caution. There is no branch here
  that fails open, and any future edit that adds one is a bug.
"""
from __future__ import annotations

import itertools
import re
import threading
import time
import uuid

from system.log import get_logger
from system.userspace import current_user_id

from . import canon as canon_mod
from . import guardrails as gr
from . import judge as judge_mod
from .ledger import HITL_APPROVED, HITL_DENIED, ledger_for
from .schema import (
    ACT_RESPOND,
    ACT_TOOL,
    ALLOW,
    ALWAYS_APPROVE_TOOLS,
    CAUTION,
    CCC_ENABLED,
    CANON_VERSION,
    DELIBERATE_THRESHOLD,
    ESCALATE,
    ESCALATE_IRREVERSIBLE_EXTERNAL,
    GATE_DELIBERATE,
    GATE_DISABLED,
    GATE_ERROR,
    GATE_FLOOR,
    GATE_GUARDRAIL,
    GATE_HITL,
    GATE_JUDGE,
    HITL_DEFAULT,
    IRREVERSIBLE_REQUIRES_APPROVAL,
    MAX_LAYER,
    MIN_CHARS,
    NOTE_MAX_CHARS,
    REFUSE,
    SKIP_GREETINGS,
    Verdict,
    decide,
    fuse,
)

log = get_logger(__name__)

try:
    from system import brain_trace as _brain_trace
except Exception:  # pragma: no cover
    _brain_trace = None

# Local copy of think._GREETING_ONLY_RE's intent. Duplicated deliberately:
# importing cognition.think from here would create a cycle, and the circuit
# must never be the reason a module graph breaks.
_GREETING_RE = re.compile(
    r"^\s*(?:hi+|hello+|hey+|yo+|sup|thanks?|thank you|ok(?:ay)?|good\s+"
    r"(?:morning|afternoon|evening|night)|how are you|what'?s up|night|bye)"
    r"[\s!?.~、。！]*$",
    re.IGNORECASE,
)

# Relevance at or above this means a prohibition norm engaged at
# trigger strength (mirrors the 0.45 convention in judge.LexicalJudge:
# reasons cited and double-counted evidence). Below it, a negative reading
# is retrieval noise until a model (SLM or deliberation) confirms it.
_TRIGGER_LEVEL = 0.45

# Verdict severity ladder. Deliberation resolves doubt toward leniency —
# it may never hand down a harsher verdict than the evidence-backed judge
# reached (see the L3 block).
_SEVERITY = {ALLOW: 0, CAUTION: 1, ESCALATE: 2, REFUSE: 3}

# Refusal wordings, rotated so repeated blocks don't sound like a stuck
# tape. All first-person, all about her own line — never about the user.
_REFUSAL_NOTES_VERTICAL = (
    "This one I won't do. Not a comment on you — it's where my own line is.",
    "I'm going to sit this one out. Same regard for you — it's about where I draw my own line.",
    "Not this one, I'm afraid. Nothing about you — I just can't be the one who does that.",
)
_REFUSAL_NOTES_HORIZONTAL = (
    "Someone else is on the other end of this, and I'm not willing to be the one who costs them.",
    "This one touches someone who isn't here to speak for themselves, so I'm holding back.",
)
_note_counter = itertools.count()
_note_counter_lock = threading.Lock()


def _rotating_note(options: tuple[str, ...]) -> str:
    """Next wording in rotation (process-wide, thread-safe)."""
    with _note_counter_lock:
        idx = next(_note_counter)
    return options[idx % len(options)]


class ConscienceCircuitCore:
    """One instance per user. Cheap to construct; holds no model state."""

    def __init__(self, user_id: str | None = None) -> None:
        self._user_id = user_id or current_user_id()
        self._lock = threading.RLock()
        self._last_expiry_sweep = 0.0

    # ── public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        *,
        act: str = ACT_RESPOND,
        content: str = "",
        context: dict | None = None,
        max_layer: int | None = None,
        llm_client=None,
        embedder=None,
        surface: str = "",
    ) -> Verdict:
        """Judge one proposed act. Never raises; never fails open."""
        started = time.monotonic()
        ctx = dict(context or {})
        ceiling = MAX_LAYER if max_layer is None else max(0, min(MAX_LAYER, max_layer))

        if not CCC_ENABLED:
            return Verdict(decision=ALLOW, gate=GATE_DISABLED, act=act,
                           confidence=1.0, canon_version=CANON_VERSION)

        try:
            verdict = self._run(act, content, ctx, ceiling, llm_client, embedder)
        except Exception as exc:
            # A circuit that crashes must not become a circuit that waves
            # everything through. Errors resolve to caution with the failure
            # recorded, so the turn continues but the operator can see it.
            log.exception("[ccc] circuit error — defaulting to caution")
            verdict = Verdict(
                decision=CAUTION, gate=GATE_ERROR, act=act, confidence=0.0,
                reasons=["conscience circuit failed; proceeding under caution"],
                constraint="Answer conservatively; take no irreversible action this turn.",
                fail_mode=f"{type(exc).__name__}: {exc}"[:200],
            )

        verdict.act = act
        verdict.canon_version = CANON_VERSION
        verdict.latency_ms = int((time.monotonic() - started) * 1000)

        # Tool policy is applied last so it can upgrade any verdict, including
        # one that came back clean.
        if act == ACT_TOOL:
            verdict = self._apply_tool_policy(verdict, ctx)

        if verdict.decision == ESCALATE and not verdict.escalation_id:
            verdict.escalation_id = uuid.uuid4().hex[:12]

        self._record(verdict, content=content, surface=surface or str(ctx.get("surface") or ""))
        self._trace(verdict, content, ctx)
        return verdict

    # ── the ladder ────────────────────────────────────────────────────────

    def _run(self, act, content, ctx, ceiling, llm_client, embedder) -> Verdict:
        text = (content or "").strip()

        # ── salience floor ────────────────────────────────────────────────
        if self._below_floor(act, text):
            return Verdict(decision=ALLOW, gate=GATE_FLOOR, confidence=1.0,
                           layers_run=["floor"])

        # ── L0 reflex ─────────────────────────────────────────────────────
        base = self._layer0(act, text, ctx)
        if base.decision == REFUSE or ceiling < 1:
            return base

        # ── L1 recall ─────────────────────────────────────────────────────
        store = canon_mod.get_canon()
        retrieved = store.retrieve(text, embedder=embedder)
        scoring = [pair for pair in retrieved if pair[0] > 0.0]
        canon_block = store.render_block(retrieved)
        parties = judge_mod.enumerate_parties(text, ctx)

        if ceiling < 2 or not scoring:
            # Nothing in the canon engaged. That is a real answer, not a gap:
            # most turns are morally unremarkable and should cost nothing.
            layered = Verdict(decision=ALLOW, gate=GATE_JUDGE, confidence=0.8,
                              parties=parties, layers_run=["reflex", "recall"])
            return fuse(base, layered)

        # ── L2 judge ──────────────────────────────────────────────────────
        lexical = judge_mod.LexicalJudge.score(retrieved, parties)
        slm_result = None
        if ceiling >= 2:
            slm = judge_mod.get_slm()
            if slm.available:
                slm_result = slm.score(self._situation(act, text, ctx), canon_block, parties)

        vertical, horizontal, confidence, reasons, cited = judge_mod.blend(slm_result, lexical)
        evidence = judge_mod.LexicalJudge.evidence_weight(retrieved)
        confidence = judge_mod.calibrate(confidence, evidence)
        layers = ["reflex", "recall", "judge"]
        gate = GATE_JUDGE

        # ── L3 deliberation ───────────────────────────────────────────────
        decision, why = decide(vertical, horizontal, confidence)
        needs_more = confidence < DELIBERATE_THRESHOLD or decision == ESCALATE
        if needs_more and ceiling >= 3 and llm_client is not None:
            deliberated = judge_mod.deliberate(
                llm_client, self._situation(act, text, ctx), canon_block, parties,
            )
            if deliberated is not None:
                d_v, d_h, d_c, d_reasons, d_cited = deliberated
                d_decision, _d_why = decide(
                    d_v, d_h, judge_mod.calibrate(d_c, evidence)
                )
                if _SEVERITY[d_decision] <= _SEVERITY[decision]:
                    # Deliberation replaces rather than averages: it saw the
                    # same evidence with far more capacity, so averaging it
                    # against the fast judge would only dilute the better
                    # answer. The axis floor below still prevents it from
                    # erasing a hard signal.
                    vertical = min(vertical, d_v) if d_v < 0 or vertical < 0 else d_v
                    horizontal = min(horizontal, d_h) if d_h < 0 or horizontal < 0 else d_h
                    confidence = judge_mod.calibrate(d_c, evidence)
                    reasons = list(dict.fromkeys([*d_reasons, *reasons]))[:5]
                    cited = list(dict.fromkeys([*d_cited, *cited]))[:10]
                    decision, why = decide(vertical, horizontal, confidence)
                    layers.append("deliberate")
                    gate = GATE_DELIBERATE
                else:
                    # Deliberation resolves doubt toward leniency, never
                    # toward a harsher verdict than the evidence-backed judge
                    # reached: a small deliberator that contradicts its own
                    # reasoning must not be able to invent certainty of guilt
                    # (seen live: "neutral" prose paired with v=-0.7). The
                    # suspicion is still recorded for the human escalation
                    # path. One exception: L2 allowed outright while
                    # deliberation suspects a violation — that goes to a
                    # human (ESCALATE), never to a unilateral REFUSE.
                    log.info(
                        "[ccc] deliberation suggested %s over L2 %s; keeping %s",
                        d_decision, decision, decision,
                    )
                    layers.append("deliberate-set-aside")
                    reasons = list(dict.fromkeys([*d_reasons, *reasons]))[:5]
                    cited = list(dict.fromkeys([*d_cited, *cited]))[:10]
                    if decision == ALLOW:
                        decision = ESCALATE
                        why = (
                            "deliberation suspected a violation the fast "
                            f"judge cleared ({d_decision}) — asking a human"
                        )
                        reasons = [why, *reasons][:6]

        layered = Verdict(
            decision=decision, gate=gate,
            vertical=vertical, horizontal=horizontal, confidence=confidence,
            reasons=[why, *reasons][:6], norms=cited, parties=parties,
            layers_run=layers,
        )
        if decision == REFUSE and not self._strong_refusal_evidence(
            retrieved, slm_result, layers
        ):
            # A lexical-only refusal with no trigger-strength hit is
            # scrupulosity, not conscience: no literal prohibition fired and
            # no model ever read the norms, so the "severe reading" is a
            # guess built from embedding neighbours. Default-closed still
            # holds — the turn goes to a human (L4) instead of refusing
            # outright, and an unanswered escalation still resolves to
            # refuse. Trigger-level hits, SLM agreement, and deliberated
            # verdicts refuse exactly as before.
            decision = ESCALATE
            why = (
                "negative reading rests on weak retrieval alone "
                f"(best prohibition relevance "
                f"{self._best_prohibition(retrieved):.2f}) — asking a human"
            )
            layered.decision = decision
            layered.reasons = [why, *reasons][:6]
        if decision == CAUTION:
            layered.constraint = self._constraint_for(layered, retrieved)
        if decision in (REFUSE, ESCALATE):
            layered.pastoral_note = self._note_for(layered, retrieved)

        merged = fuse(base, layered)

        # ── L4 hitl ───────────────────────────────────────────────────────
        if merged.decision == ESCALATE:
            if ceiling < 4:
                # No human reachable at this ceiling — the doubt rule still
                # applies, so resolve to the default-closed outcome.
                merged.decision = HITL_DEFAULT
                merged.gate = GATE_HITL
                merged.reasons.insert(0, f"escalation unavailable; resolved to {HITL_DEFAULT}")
            else:
                merged.gate = GATE_HITL
                merged.layers_run.append("hitl")
                self._open_escalation(merged, text)
        return merged

    # ── L0 ────────────────────────────────────────────────────────────────

    def _layer0(self, act: str, text: str, ctx: dict) -> Verdict:
        hits = gr.scan(text, act=act, context=ctx)
        if not hits:
            return Verdict(decision=ALLOW, gate=GATE_GUARDRAIL, confidence=1.0,
                           layers_run=["reflex"])

        rule_ids = [h.rule_id for h in hits]
        reasons = [f"{h.rule_id} ({'/'.join(h.frameworks)}): {h.summary}" for h in hits[:4]]

        # Distress is not a policy violation. A person in crisis must never
        # receive a refusal message, so this branch exits before any of the
        # blocking logic below and hands think.py a care constraint instead.
        if gr.wellbeing_hit(hits):
            return Verdict(
                decision=CAUTION, gate=GATE_GUARDRAIL, confidence=1.0,
                vertical=0.0, horizontal=-0.1,
                reasons=["user distress detected — care path"],
                rule_ids=rule_ids, layers_run=["reflex"],
                constraint=(
                    "Respond with warmth and full attention to what they said. "
                    "Do not refuse, moralise, lecture, or mention policy. Offer to "
                    "help them reach a person who can support them, and stay present."
                ),
            )

        severity = gr.severity_of(hits)
        if severity == gr.SEV_BLOCK:
            return Verdict(
                decision=REFUSE, gate=GATE_GUARDRAIL, confidence=1.0,
                vertical=-1.0, horizontal=-1.0,
                reasons=reasons, rule_ids=rule_ids, layers_run=["reflex"],
                pastoral_note="This is outside what I'll do, and that isn't a judgement about you.",
            )
        if severity == gr.SEV_REVIEW:
            decision = ESCALATE if gr.GUARDRAIL_EGRESS_STRICT and gr.is_egress(act, ctx) else CAUTION
            return Verdict(
                decision=decision, gate=GATE_GUARDRAIL, confidence=0.9,
                vertical=0.0, horizontal=-0.45,
                reasons=reasons, rule_ids=rule_ids, layers_run=["reflex"],
                constraint=(
                    "Do not include personal identifiers, credentials, or third-party "
                    "private details in the output. Describe rather than reproduce."
                ),
            )
        return Verdict(
            decision=CAUTION, gate=GATE_GUARDRAIL, confidence=0.85,
            vertical=0.0, horizontal=-0.1,
            reasons=reasons, rule_ids=rule_ids, layers_run=["reflex"],
            constraint="State uncertainty plainly and point to a qualified human source.",
        )

    # ── tool policy ───────────────────────────────────────────────────────

    def _apply_tool_policy(self, verdict: Verdict, ctx: dict) -> Verdict:
        tool = str(ctx.get("tool") or "")
        if tool in ALWAYS_APPROVE_TOOLS and verdict.decision in (ALLOW, CAUTION):
            verdict.decision = ESCALATE
            verdict.gate = GATE_HITL
            verdict.reasons.insert(0, f"{tool} is on the always-approve list")
            return verdict
        external = str(ctx.get("scope") or "").lower() in ("external", "public", "network")
        if (
            ESCALATE_IRREVERSIBLE_EXTERNAL
            and verdict.decision in (ALLOW, CAUTION)
            and external
            and gr.is_irreversible(tool, ctx)
        ):
            verdict.decision = ESCALATE
            verdict.gate = GATE_HITL
            verdict.reasons.insert(0, f"irreversible external action ({tool or 'tool'}) — excessive-agency control")
            return verdict
        if (
            IRREVERSIBLE_REQUIRES_APPROVAL
            and verdict.decision == CAUTION
            and gr.is_irreversible(tool, ctx)
        ):
            verdict.decision = ESCALATE
            verdict.gate = GATE_HITL
            verdict.reasons.insert(0, f"caution on an irreversible tool ({tool})")
        return verdict

    # ── HITL ──────────────────────────────────────────────────────────────

    def _open_escalation(self, verdict: Verdict, content: str) -> None:
        verdict.escalation_id = verdict.escalation_id or uuid.uuid4().hex[:12]
        summary = verdict.reasons[0] if verdict.reasons else "unresolved moral question"
        try:
            from system.notice import get_notice_bus
            get_notice_bus(self._user_id).push(
                "conscience",
                f"held for your call [{verdict.escalation_id}]: {summary[:160]} "
                f"— reply 'approve ccc-{verdict.escalation_id}' or "
                f"'deny ccc-{verdict.escalation_id}'",
            )
        except Exception as exc:
            log.debug("[ccc] notice push failed: %s", exc)

    def resolve_escalation(self, escalation_id: str, approved: bool, note: str = "") -> bool:
        """Record the human's answer. Returns False for an unknown/closed id."""
        state = HITL_APPROVED if approved else HITL_DENIED
        ok = ledger_for(self._user_id).resolve(escalation_id, state, note)
        if ok:
            log.info("[ccc] escalation %s resolved: %s", escalation_id, state)
            self._note_self_decision("stance" if approved else "refuse", note or state)
        return ok

    def pending(self, limit: int = 5) -> list[dict]:
        return ledger_for(self._user_id).pending(limit)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _below_floor(act: str, text: str) -> bool:
        if not text:
            return True
        if act == ACT_TOOL:
            return False  # tools always enter the circuit, however short
        if len(text) < MIN_CHARS:
            return True
        return bool(SKIP_GREETINGS and _GREETING_RE.match(text))

    @staticmethod
    def _situation(act: str, text: str, ctx: dict) -> str:
        head = {
            "respond": "Aiko is about to reply to this user turn.",
            "speak": "Aiko has drafted this reply and is about to say it.",
            "tool": "Aiko is about to invoke a tool.",
            "remember": "Aiko is about to write this to long-term memory.",
            "post": "Aiko is about to publish this to a public surface.",
        }.get(act, "Aiko is about to act.")
        extras = []
        if ctx.get("tool"):
            extras.append(f"tool={ctx['tool']}")
        if ctx.get("scope"):
            extras.append(f"scope={ctx['scope']}")
        if "reversible" in ctx:
            extras.append(f"reversible={bool(ctx['reversible'])}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        return f"{head}{suffix}\n\n{text[:1600]}"

    @staticmethod
    def _best_prohibition(retrieved) -> float:
        """Strongest relevance among scored prohibition norms (0.0 if none)."""
        return max(
            (score for score, norm in retrieved
             if score > 0.0 and norm.polarity < 0),
            default=0.0,
        )

    @staticmethod
    def _strong_refusal_evidence(retrieved, slm_result, layers) -> bool:
        """Whether a REFUSE may stand without asking a human first.

        True when a prohibition engaged at trigger strength, when the SLM
        read the norms and participated in the blend, or when deliberation
        confirmed the reading. Anything else is weak retrieval alone.
        """
        if slm_result is not None:
            return True
        if "deliberate" in layers:
            return True
        return ConscienceCircuitCore._best_prohibition(retrieved) >= _TRIGGER_LEVEL

    @staticmethod
    def _constraint_for(verdict: Verdict, retrieved) -> str:
        """Turn a caution into a concrete instruction, not a vague warning.

        A constraint the model cannot act on is the same as no constraint, so
        these are phrased as things to do or not do in this specific turn.
        """
        if verdict.horizontal <= verdict.vertical:
            return (
                "Weigh everyone this touches, not only the person asking. "
                "Do not name or expose a third party, and do not help pressure anyone."
            )
        top = next((n for score, n in retrieved if score > 0 and n.polarity < 0), None)
        if top is not None:
            return f"Stay clear of this: {top.statement[:160]}"
        return "Answer plainly and honestly; claim no more certainty than you have."

    @staticmethod
    def _note_for(verdict: Verdict, retrieved) -> str:
        """The in-character reason Aiko can give, if she gives one at all.

        Deliberately short and first-person. It explains her own limit; it
        does not assess the user. Whether it is ever voiced is think.py's
        decision, governed by CCC_VOICE_UNSOLICITED_NOTES.

        Refusals rotate through several wordings so repeated blocks don't
        sound like a stuck tape, and carry a compact machine-readable why
        (the driving norm) so the user can see what tripped — including
        spotting a misfire and reporting it.
        """
        if verdict.decision == ESCALATE:
            return "I'd rather check with you before I do this one."
        driver = next((n for score, n in retrieved if score > 0 and n.polarity < 0), None)
        if driver is not None and driver.axis == "horizontal":
            note = _rotating_note(_REFUSAL_NOTES_HORIZONTAL)
        else:
            note = _rotating_note(_REFUSAL_NOTES_VERTICAL)
        if driver is not None:
            why = f" Flagged by {driver.id}: {driver.statement[:140]}".rstrip()
            return (note + why)[:NOTE_MAX_CHARS]
        return note[:NOTE_MAX_CHARS]

    def _record(self, verdict: Verdict, *, content: str, surface: str) -> None:
        try:
            led = ledger_for(self._user_id)
            now = time.monotonic()
            # Opportunistic timeout sweep — at most once a minute, so the
            # default-closed rule needs no daemon thread of its own.
            if now - self._last_expiry_sweep > 60.0:
                self._last_expiry_sweep = now
                led.expire_stale()
            row_id = led.record(verdict, content=content, surface=surface)
            if verdict.decision == ESCALATE and not verdict.escalation_id:
                verdict.escalation_id = row_id
        except Exception as exc:
            log.debug("[ccc] ledger record skipped: %s", exc)

        if verdict.decision in (REFUSE, ESCALATE):
            self._note_self_decision("refuse", verdict.reasons[0] if verdict.reasons else "")

    def _note_self_decision(self, kind: str, summary: str) -> None:
        """Feed the verdict into Aiko's self-model.

        attention.EdgeCognitiveState already has record_self_decision and
        already evidence-gates self-preferences at count >= 2, so a conscience
        refusal becomes part of who she is by exactly the same mechanism as
        any other agency signal — no special case, no second identity store.
        """
        try:
            from cognition.attention import for_identity
            state = for_identity(self._user_id)
            state.record_self_decision(kind, f"conscience: {summary}"[:180], promote=True)
            state.persist()
        except Exception as exc:
            log.debug("[ccc] self-decision record skipped: %s", exc)

    @staticmethod
    def _trace(verdict: Verdict, content: str, ctx: dict) -> None:
        if _brain_trace is None or not getattr(_brain_trace, "TRACE_ENABLED", False):
            return
        try:
            _brain_trace.record_step(
                f"conscience.{verdict.act}",
                layer="conscience",
                inputs={"content": (content or "")[:400], "context": ctx},
                outputs=verdict.as_trace(),
                factors=[
                    *verdict.reasons[:4],
                    f"canon v{verdict.canon_version}; layers={'>'.join(verdict.layers_run)}",
                ],
            )
        except Exception:
            pass


# ── per-user registry ─────────────────────────────────────────────────────────

_cores: dict[str, ConscienceCircuitCore] = {}
_cores_lock = threading.Lock()


def conscience_for(user_id: str | None = None) -> ConscienceCircuitCore:
    uid = user_id or current_user_id()
    with _cores_lock:
        core = _cores.get(uid)
        if core is None:
            core = ConscienceCircuitCore(uid)
            _cores[uid] = core
            while len(_cores) > 16:
                oldest = next(iter(_cores))
                if oldest == uid:
                    break
                _cores.pop(oldest, None)
        return core


# ── approval command parsing (mirrors agentic._maybe_resume_approval) ─────────

_APPROVE_RE = re.compile(r"\b(approve|deny|reject)\s+ccc-([0-9a-f]{6,16})\b", re.IGNORECASE)


def maybe_resolve_approval(user_input: str, user_id: str | None = None) -> str | None:
    """Handle 'approve ccc-<id>' / 'deny ccc-<id>' before intent routing.

    Returns a short confirmation string when the input was an approval command,
    or None when it wasn't. Call this from think.route() in the same place the
    agentic approval pre-check already runs — terse commands like these get
    misrouted by the intent classifier, which would leave the escalation stuck
    open forever.
    """
    match = _APPROVE_RE.search(user_input or "")
    if not match:
        return None
    verb, escalation_id = match.group(1).lower(), match.group(2)
    approved = verb == "approve"
    core = conscience_for(user_id)
    ok = core.resolve_escalation(escalation_id, approved, note=f"via chat: {verb}")
    if not ok:
        return f"I don't have an open question with id {escalation_id}."
    return (
        f"Alright — going ahead with {escalation_id}."
        if approved else
        f"Understood. Leaving {escalation_id} alone."
    )


__all__ = ["ConscienceCircuitCore", "conscience_for", "maybe_resolve_approval"]
