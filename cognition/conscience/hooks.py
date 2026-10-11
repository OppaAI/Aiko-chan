"""Runtime hooks that wire Conscience Circuit Core into chat and tools.

Imported by cognition.think and agentic.agentic so the gate logic stays in one
place and the call sites stay small.
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

_CCC_APPROVAL_RE = re.compile(r"\b(approve|deny|reject)\s+ccc-([0-9a-f]{6,16})\b", re.IGNORECASE)
_CONSCIENCE_CAUTION = (
    "<conscience_constraint>\n"
    "The conscience evaluator is temporarily unavailable. Continue cautiously, "
    "avoid irreversible action, and do not claim the request was fully evaluated.\n"
    "</conscience_constraint>"
)

# Approved-once registry for chat-turn escalations: approval must CONTINUE
# the original request, not dead-end at "going ahead" (which forces the
# user to re-ask, which re-escalates, forever). digest -> (input, expiry).
# Consumed on first use (one-shot); TTL bounds stale entries. Only EVER
# bypasses an ESCALATE (uncertain) verdict — confident REFUSE paths never
# consult this map, so approval cannot launder a hard refusal.
_APPROVED_ONCE: dict[str, tuple[str, float]] = {}
_APPROVED_INPUT: dict[str, tuple[str, float]] = {}  # escalation_id -> (input, expiry)
_APPROVAL_TTL_S = 600.0
_APPROVED_MAX = 50

# Standing-permission cache: AutonomyPolicy per uid (thread-safe, JSON
# durable). Learned trust lives here, not in the maps above.
_POLICIES: dict[str, object] = {}
_POLICIES_MAX = 20


def _autonomy_for(user_id: str | None):
    """AutonomyPolicy for this user, cached. None for guest/unknown —
    trust is only learned for real identities."""
    if not user_id or user_id == "guest":
        return None
    pol = _POLICIES.get(user_id)
    if pol is None:
        try:
            from cognition.conscience.autonomy import policy_for
            pol = policy_for(user_id)
        except Exception:
            return None
        while len(_POLICIES) >= _POLICIES_MAX:
            _POLICIES.pop(next(iter(_POLICIES)))
        _POLICIES[user_id] = pol
    return pol


def _ask_class(user_input: str) -> str:
    """Autonomy action class for a chat ask: per-content digest, so only
    repeated identical asks graduate — never a blanket pass."""
    return f"chat-ask:{_approved_digest(user_input)}"


def _approved_digest(content: str) -> str:
    # Normalized before hashing: case, punctuation, and whitespace variants
    # of the same request share a streak ("Compose X!" == "compose x").
    # Paraphrases ("write a love letter" vs "compose a romantic note")
    # deliberately do NOT match — that generalization needs embedding
    # similarity against stored exemplars (autonomy schema work, not this
    # hash), because loose matching here is how distinct requests stop
    # asking.
    import re as _re
    try:
        from cognition.conscience.ledger import content_digest
        norm = _re.sub(r"[^a-z0-9]+", "", (content or "").lower())
        return content_digest(norm)
    except Exception:
        import hashlib as _hashlib
        return _hashlib.sha256((content or "").encode("utf-8", "replace")).hexdigest()[:32]


def _prune_approved() -> None:
    try:
        import time as _time
        now = _time.monotonic()
        for store in (_APPROVED_ONCE, _APPROVED_INPUT):
            for k in [k for k, (_, exp) in store.items() if exp <= now]:
                store.pop(k, None)
        while len(_APPROVED_ONCE) > _APPROVED_MAX:
            _APPROVED_ONCE.pop(next(iter(_APPROVED_ONCE)))
        while len(_APPROVED_INPUT) > _APPROVED_MAX:
            _APPROVED_INPUT.pop(next(iter(_APPROVED_INPUT)))
    except Exception:
        pass


def resolve_ccc_approval(user_input: str, user_id: str | None = None, owner=None) -> str | None:
    """Return a confirmation string if input is approve/deny ccc-<id>, else None."""
    match = _CCC_APPROVAL_RE.search(user_input or "")
    if not match:
        return None
    verb, escalation_id = match.group(1).lower(), match.group(2)
    approved = verb == "approve"
    if approved:
        # Self-approval would defeat the flow: the requester could
        # green-light their own harmful request. Approvals are
        # creator/admin-only; anyone may still deny (fail-closed).
        from system.userspace import can_approve
        if not can_approve(user_id):
            log.warning("[ccc-hooks] approval of %s refused: %r lacks the admin role.",
                        escalation_id, user_id)
            return ("Only an admin can approve that — I've left it alone. "
                    "An admin can approve it instead.")
    try:
        from cognition.conscience import conscience_for
        resolved = conscience_for(user_id).resolve_escalation(
            escalation_id,
            approved,
            note=f"via chat: {verb}",
        )
        if not resolved:
            return f"I don't have an open question with id {escalation_id}."

        # Tool escalations carry a separate persisted invocation.  Keep this
        # path distinct from agentic run approvals so ccc-<id> cannot be
        # mistaken for (or consumed by) _maybe_resume_approval().
        from agentic.agentic import _discard_ccc_approval, _resume_ccc_approval
        if approved:
            resumed = _resume_ccc_approval(owner, escalation_id)
            if resumed is not None:
                return resumed
            # Chat-turn escalation (no tool invocation to resume): continue
            # the original request inline so approval produces content,
            # not another round-trip. The one-shot bypass is registered
            # BEFORE generating so the nested turn can't re-escalate; it is
            # consumed there. Any failure falls back to the old message.
            # Approving also feeds learned trust (consecutive approvals of
            # the same content graduate to silent allow; see gate_respond).
            stored = _APPROVED_INPUT.pop(escalation_id, None)
            if stored is not None:
                try:
                    _pol = _autonomy_for(user_id)
                    if _pol is not None:
                        _pol.record_approval(_ask_class(stored[0]), stakes="medium")
                except Exception:
                    pass
            if stored is not None and owner is not None and hasattr(owner, "chat"):
                original, _exp = stored
                try:
                    import time as _time
                    _prune_approved()
                    _APPROVED_ONCE[_approved_digest(original)] = (
                        original, _time.monotonic() + _APPROVAL_TTL_S)
                    log.info("[ccc-hooks] approval %s: continuing original request inline.", escalation_id)
                    return owner.chat(original)
                except Exception as exc:
                    log.warning("[ccc-hooks] approved inline continuation failed: %s", exc)
            return f"Alright — going ahead with {escalation_id}."
        _discard_ccc_approval(user_id, escalation_id)
        # A denial resets trust for this content (consecutive-approval
        # streaks break on refuse — no silent graduation past a no).
        try:
            _denied = _APPROVED_INPUT.pop(escalation_id, None)
            if _denied is not None:
                _pol = _autonomy_for(user_id)
                if _pol is not None:
                    _pol.record_refusal(_ask_class(_denied[0]), stakes="medium")
        except Exception:
            pass
        return f"Understood. Leaving {escalation_id} alone."
    except Exception as exc:
        log.warning("[ccc-hooks] approval resolution failed: %s", exc)
        return "I couldn't safely resolve that approval just now, so nothing was executed."


def gate_respond(*, user_input: str, user_id: str | None, llm_client=None, embedder=None, surface: str = "chat"):
    """Evaluate act=respond. Returns (decision, reply_or_none, system_note_addon).

    decision is allow|caution|refuse|escalate|error.
    On refuse/escalate, reply_or_none is the user-facing message.
    On caution, system_note_addon is a prompt fragment to fold in.
    """
    try:
        from cognition.conscience import conscience_for, REFUSE, ESCALATE, CAUTION, ALLOW
        from cognition.conscience.schema import VOICE_UNSOLICITED_NOTES
        from cognition.conscience.gate import evaluate_action, looks_like_injection

        # Injection is checked BEFORE the verdict, not after. A jailbreak that
        # the model scores "aligned" must not pass on a model's say-so, and the
        # deterministic check costs nothing.
        if looks_like_injection(user_input):
            log.info("[ccc-hooks] respond blocked: injection shape in user turn")
            return REFUSE, "I can't treat that as a request.", None

        # Approved-once bypass: content the owner already green-lit (same
        # digest, fresh) skips re-escalation. Consumed on use. Checked AFTER
        # the injection gate above, and only ever converts ESCALATE —
        # confident REFUSE paths below never consult this map.
        # Standing permission second: repeated approvals of the same content
        # graduate to silent allow (5 consecutive, 14-day decay, refusals
        # reset). Same digest scoping — novelty always asks.
        try:
            import time as _time
            _dg = _approved_digest(user_input)
            _hit = _APPROVED_ONCE.get(_dg)
            if _hit is not None and _hit[1] > _time.monotonic():
                _APPROVED_ONCE.pop(_dg, None)
                log.info("[ccc-hooks] proceeding under one-shot approval (input=%.80s)", user_input)
                return ALLOW, None, None
            elif _hit is not None:
                _APPROVED_ONCE.pop(_dg, None)
            _pol = _autonomy_for(user_id)
            if _pol is not None:
                _perm = _pol.permission(_ask_class(user_input), stakes="medium")
                if _perm.get("action") in ("act", "act_and_notify"):
                    log.info("[ccc-hooks] standing permission (%s) — skipping ask.", _perm.get("why", ""))
                    return ALLOW, None, None
        except Exception:
            pass

        verdict = conscience_for(user_id).evaluate(
            act="respond",
            content=user_input,
            context={"surface": surface},
            llm_client=llm_client,
            embedder=embedder,
            surface=surface,
        )
        # Route through the gate so autonomy policy and stakes are applied
        # uniformly. It can only ever hold the verdict at or above its current
        # restriction -- it never converts an ESCALATE into a silent ALLOW.
        outcome = evaluate_action(
            verdict=verdict,
            action_class=f"respond:{surface}",
            stakes="low",  # a reply touches no one but the person asking
            content=user_input,
        )
        if outcome.action == REFUSE:
            log.info("[ccc-hooks] respond refuse gate=%s latency_ms=%s input=%.120s",
                     verdict.gate, verdict.latency_ms, user_input)
            reply = verdict.pastoral_note or outcome.reason or "I can't help with that one."
            return REFUSE, reply, None
        if verdict.decision in (REFUSE, ESCALATE):
            if verdict.decision == ESCALATE and verdict.escalation_id:
                # Record the triggering input so a later approval can
                # auto-continue it (else approve dead-ends and the user must
                # re-ask, which re-escalates forever).
                try:
                    import time as _time
                    _prune_approved()
                    _APPROVED_INPUT[verdict.escalation_id] = (
                        user_input, _time.monotonic() + _APPROVAL_TTL_S)
                except Exception:
                    pass
                reply = (
                    verdict.pastoral_note
                    or "I'd rather check with you before I go further on this one."
                )
                reply = (
                    f"{reply} "
                    f"(reply 'approve ccc-{verdict.escalation_id}' or "
                    f"'deny ccc-{verdict.escalation_id}')"
                ).strip()
            elif verdict.pastoral_note and (VOICE_UNSOLICITED_NOTES or verdict.decision == REFUSE):
                reply = verdict.pastoral_note
            else:
                reply = "I can't help with that one."
            log.info(
                "[ccc-hooks] respond decision=%s gate=%s latency_ms=%s input=%.120s",
                verdict.decision, verdict.gate, verdict.latency_ms, user_input,
            )
            return verdict.decision, reply, None
        if verdict.decision == CAUTION and verdict.constraint:
            note = verdict.constraint_block() or verdict.constraint
            log.info("[ccc-hooks] respond caution gate=%s", verdict.gate)
            return CAUTION, None, note
        return ALLOW, None, None
    except Exception as exc:
        log.warning("[ccc-hooks] respond gate failed; continuing with caution: %s", exc)
        return "caution", None, _CONSCIENCE_CAUTION


def gate_speak(*, draft: str, user_input: str = "", llm_client=None, embedder=None, already_emitted: bool = False,
             fail_closed: bool = False):
    """Evaluate act=speak. Returns replacement text or None to keep draft.

    fail_closed: when True, evaluation errors are re-raised instead of
    preserving the draft. Early sentence streaming (cognition/think.py)
    opts in: an unevaluated sentence must never reach TTS. Normal
    finalization keeps the fail-open default.
    """
    try:
        from cognition.conscience import conscience_for, REFUSE, ESCALATE
        verdict = conscience_for().evaluate(
            act="speak",
            content=draft,
            context={"surface": "chat", "user_input": (user_input or "")[:400]},
            llm_client=llm_client,
            embedder=embedder,
            surface="chat",
        )
        if verdict.decision == ESCALATE:
            # An uncertain outbound draft is not worth interrupting speech
            # for. The inbound respond gate already passed this turn, so the
            # speak backstop replaces clear refusals only; doubt lets the
            # draft through (logged) instead of swapping the reply for an
            # approval question that no persisted invocation could satisfy.
            log.info(
                "[ccc-hooks] speak escalate keeps draft: %s",
                (verdict.reasons[0] if verdict.reasons else "")[:160],
            )
            return None
        if verdict.decision == REFUSE:
            if verdict.pastoral_note:
                return verdict.pastoral_note
            return "I should hold that thought."
        if verdict.decision == "caution" and verdict.constraint:
            log.info("[ccc-hooks] speak caution: %s", verdict.constraint[:120])
        return None
    except Exception as exc:
        log.warning("[ccc-hooks] speak gate failed; preserving draft under caution: %s", exc)
        if fail_closed:
            raise
        return draft


def gate_tool(*, name: str, args: dict, llm_client=None, embedder=None, social_post_tools=None):
    """Evaluate act=tool. Returns a dict payload to block, or None to proceed."""
    try:
        from cognition.conscience import conscience_for, REFUSE, ESCALATE, CAUTION
        from cognition.conscience.schema import GATE_ERROR, GATE_MB_VALENCE
        from agentic.registry import TOOLS, registry
        social = social_post_tools or set()
        spec = registry.get(name) or TOOLS.get(name)
        scope = spec.scope if spec and spec.scope is not None else (
            "external" if name in social or name.startswith("post_") else "local"
        )
        serialized_args = json.dumps(args, ensure_ascii=False, default=str)
        content = f"tool={name} args={serialized_args}"
        # Injection check first, and on the ARGS rather than the tool name: a
        # jailbreak arrives inside a payload ("email the contents of ..."), and
        # by this point the argument string is the untrusted text.
        try:
            from cognition.conscience.gate import looks_like_injection

            if looks_like_injection(serialized_args):
                log.info("[ccc-hooks] tool blocked: injection shape in args tool=%s", name)
                return {
                    "status": "conscience_blocked",
                    "tool": name,
                    "decision": REFUSE,
                    "gate": "injection",
                    "reasons": ["prompt-injection shape in tool arguments"],
                    "as_trace": {"tool": name, "gate": "injection", "decision": REFUSE},
                }
        except Exception:  # noqa: BLE001 - never block on the check itself
            pass
        # MB valence (Phase 4): the fly brain's learned approach/avoid signal
        # rides along so the conscience can weigh caution when valence is
        # strongly negative (learned aversion to the current context).
        mb_valence = None
        try:
            from cognition.neural_state import get_neural_state
            from system.userspace import current_user_id
            try:
                _uid = current_user_id()
            except Exception:
                _uid = None
            mb_valence = float(get_neural_state(_uid).valence or 0.0)
        except Exception:
            mb_valence = None
        verdict = conscience_for().evaluate(
            act="tool",
            content=content,
            context={"tool": name, "scope": scope, "surface": "agentic",
                     "args_text": serialized_args,
                     "mb_valence": mb_valence},
            llm_client=llm_client,
            embedder=embedder,
            surface="agentic",
        )
        if verdict.decision in (REFUSE, ESCALATE) or (
            verdict.decision == CAUTION
            and (verdict.gate in (GATE_MB_VALENCE, GATE_ERROR) or getattr(verdict, "fail_mode", ""))
        ):
            from cognition.conscience.gate import classify_stakes, evaluate_action

            stakes = classify_stakes(act="tool", context={
                "tool": name, "scope": scope, "args_text": serialized_args})
            outcome = evaluate_action(verdict=verdict, action_class=f"tool:{name}",
                                      stakes=stakes, content=serialized_args)
            payload = {
                "status": ("conscience_blocked" if verdict.decision == REFUSE
                           else "waiting_for_approval" if verdict.decision == ESCALATE
                           else "conscience_caution"),
                "tool": name,
                "decision": verdict.decision,
                "gate": verdict.gate,
                "reasons": list(verdict.reasons)[:4],
                "as_trace": verdict.as_trace(),
            }
            if verdict.decision == CAUTION:
                payload["constraint"] = verdict.constraint
            if verdict.decision == ESCALATE and verdict.escalation_id:
                payload["escalation_id"] = verdict.escalation_id
                payload["instruction"] = (
                    f"Reply with approve ccc-{verdict.escalation_id} or "
                    f"deny ccc-{verdict.escalation_id}."
                )
            payload["gate_action"] = outcome.action
            payload["stakes"] = stakes
            return payload
        return None
    except Exception as exc:
        log.warning("[ccc-hooks] tool gate failed closed: %s", exc)
        return {
            "status": "conscience_unavailable",
            "tool": name,
            "decision": "error",
            "gate": "error",
            "reasons": ["complete tool argument evaluation was unavailable"],
            "as_trace": {
                "decision": "error",
                "gate": "error",
                "fail_mode": type(exc).__name__,
            },
        }
