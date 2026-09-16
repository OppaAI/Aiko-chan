"""Runtime hooks that wire Conscience Circuit Core into chat and tools.

Imported by cognition.think and agentic.agentic so the gate logic stays in one
place and the call sites stay small.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)


def resolve_ccc_approval(user_input: str, user_id: str | None = None) -> str | None:
    """Return a confirmation string if input is approve/deny ccc-<id>, else None."""
    try:
        from cognition.conscience import maybe_resolve_approval
        return maybe_resolve_approval(user_input, user_id=user_id)
    except Exception as exc:
        log.debug("[ccc-hooks] approval resolve skipped: %s", exc)
        return None


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
        log.debug("[ccc-hooks] respond gate skipped: %s", exc)
        return "error", None, None


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
        if verdict.decision in (REFUSE, ESCALATE) and not already_emitted:
            if verdict.decision == ESCALATE and verdict.escalation_id:
                return (
                    (verdict.pastoral_note or "I'd rather check with you before saying that.")
                    + f" (reply 'approve ccc-{verdict.escalation_id}' or "
                    + f"'deny ccc-{verdict.escalation_id}')"
                )
            if verdict.pastoral_note:
                return verdict.pastoral_note
            return "I should hold that thought."
        if verdict.decision == "caution" and verdict.constraint and not already_emitted:
            log.info("[ccc-hooks] speak caution: %s", verdict.constraint[:120])
        return None
    except Exception as exc:
        log.debug("[ccc-hooks] speak gate skipped: %s", exc)
        return None


def gate_tool(*, name: str, args: dict, llm_client=None, embedder=None, social_post_tools=None):
    """Evaluate act=tool. Returns a dict payload to block, or None to proceed."""
    try:
        from cognition.conscience import conscience_for, REFUSE, ESCALATE
        social = social_post_tools or set()
        scope = "external" if name in social or name.startswith("post_") else "local"
        content = f"tool={name} args={json.dumps(args, ensure_ascii=False, default=str)[:1200]}"
        verdict = conscience_for().evaluate(
            act="tool",
            content=content,
            context={"tool": name, "scope": scope, "surface": "agentic"},
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
        log.debug("[ccc-hooks] tool gate skipped: %s", exc)
        return None
