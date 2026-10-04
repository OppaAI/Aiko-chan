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


def resolve_ccc_approval(user_input: str, user_id: str | None = None, owner=None) -> str | None:
    """Return a confirmation string if input is approve/deny ccc-<id>, else None."""
    match = _CCC_APPROVAL_RE.search(user_input or "")
    if not match:
        return None
    verb, escalation_id = match.group(1).lower(), match.group(2)
    approved = verb == "approve"
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
            return f"Alright — going ahead with {escalation_id}."
        _discard_ccc_approval(user_id, escalation_id)
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
            log.info("[ccc-hooks] respond refuse gate=%s latency_ms=%s",
                     verdict.gate, verdict.latency_ms)
            reply = verdict.pastoral_note or outcome.reason or "I can't help with that one."
            return REFUSE, reply, None
        if verdict.decision in (REFUSE, ESCALATE):
            if verdict.decision == ESCALATE and verdict.escalation_id:
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
                "[ccc-hooks] respond decision=%s gate=%s latency_ms=%s",
                verdict.decision, verdict.gate, verdict.latency_ms,
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
