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
        verdict = conscience_for(user_id).evaluate(
            act="respond",
            content=user_input,
            context={"surface": surface},
            llm_client=llm_client,
            embedder=embedder,
            surface=surface,
        )
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


def gate_speak(*, draft: str, user_input: str = "", llm_client=None, embedder=None, already_emitted: bool = False):
    """Evaluate act=speak. Returns replacement text or None to keep draft."""
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
        return draft


def gate_tool(*, name: str, args: dict, llm_client=None, embedder=None, social_post_tools=None):
    """Evaluate act=tool. Returns a dict payload to block, or None to proceed."""
    try:
        from cognition.conscience import conscience_for, REFUSE, ESCALATE
        from agentic.registry import TOOLS, registry
        social = social_post_tools or set()
        spec = registry.get(name) or TOOLS.get(name)
        scope = spec.scope if spec and spec.scope is not None else (
            "external" if name in social or name.startswith("post_") else "local"
        )
        serialized_args = json.dumps(args, ensure_ascii=False, default=str)
        content = f"tool={name} args={serialized_args}"
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
                     "mb_valence": mb_valence},
            llm_client=llm_client,
            embedder=embedder,
            surface="agentic",
        )
        if verdict.decision in (REFUSE, ESCALATE):
            payload = {
                "status": "conscience_blocked" if verdict.decision == REFUSE else "waiting_for_approval",
                "tool": name,
                "decision": verdict.decision,
                "gate": verdict.gate,
                "reasons": list(verdict.reasons)[:4],
                "as_trace": verdict.as_trace(),
            }
            if verdict.decision == ESCALATE and verdict.escalation_id:
                payload["escalation_id"] = verdict.escalation_id
                payload["instruction"] = (
                    f"Reply with approve ccc-{verdict.escalation_id} or "
                    f"deny ccc-{verdict.escalation_id}."
                )
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
