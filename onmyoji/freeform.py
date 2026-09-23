"""Freeform action interpreter for the Aiko-Onmyoji game (server side).

Pipeline: the LLM ONLY resolves free text into a strict intent JSON;
validation and effect computation happen HERE, in code. This module never
mutates the server's own GameStore — it interprets and narrates; the game
client owns its state and applies the returned effect descriptors.

Verb vocabulary (fixed): travel, rest, search, ward, strike, bind, flee,
command, talk, recruit, release, give, steal, kiss, intimate, work,
possess, move. The interpreter prompt maps anything else to "talk"; an
unknown verb that slips through is refused in code.

Effect descriptor "type" vocabulary (fixed): karma, faction, mp, hp,
bond, gold, item, consequence, flag.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

VERBS = ("travel", "rest", "search", "ward", "strike", "bind", "flee", "command",
         "talk", "recruit", "release", "give", "steal", "kiss", "intimate", "work",
         "possess", "move")

EFFECT_TYPES = ("karma", "faction", "mp", "hp", "bond", "gold", "item",
                "consequence", "flag")

# Physical-impossibility denylist: (regex, short name). Scanned over
# verb + target + args JSON + raw text when the command is the player's own
# (aiko_command=false). When Aiko acts (aiko_command=true) the flight /
# water-crossing / possession categories are exempt; the rest stay impossible.
_MORTAL_DENY: list[tuple[str, str]] = [
    (r"\bfly\b|\blevitat\w*|\bsoar\w*", "fly"),
    (r"walk(?:ing)? on water", "walk on water"),
    (r"\bteleport\w*", "teleport"),
    (r"pass through walls?", "pass through walls"),
    (r"breathe underwater", "breathe underwater"),
    (r"survive lava", "survive lava"),
    (r"\binvisib\w*", "turn invisible"),
    (r"time travel(?:l?ing)?", "travel through time"),
]
_AIKO_EXEMPT = {"fly", "walk on water"}  # spirit powers allowed when commanded

# Backstop: raw-text scan for Aiko + adult verbs (the LLM might miss it).
_ADULT_VERB_RE = re.compile(r"\b(kiss\w*|sex|intimate|naked)\b|\bsleeps? with\b",
                            re.IGNORECASE)

_FALLBACK_INTENT: dict[str, Any] = {
    "verb": "talk", "target": "", "args": {}, "aiko_command": False,
    "adult": False, "forced": False, "capability_ok": True, "refusal_reason": "",
}

_PUBLIC_KEYS = ("verb", "target", "args", "aiko_command", "adult", "forced")


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return default if value is None else bool(value)


def _parse_intent(raw: str) -> dict[str, Any]:
    """Defensive parse: first {...} block -> intent dict; ANY failure -> talk fallback."""
    try:
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        data = json.loads(match.group(0)) if match else None
        if not isinstance(data, dict):
            raise ValueError("no intent object")
    except (ValueError, AttributeError):
        return dict(_FALLBACK_INTENT)
    intent = dict(_FALLBACK_INTENT)
    verb = str(data.get("verb", "")).strip().lower()
    intent["verb"] = verb or "talk"
    intent["target"] = str(data.get("target", "") or "").strip()
    args = data.get("args")
    intent["args"] = dict(args) if isinstance(args, dict) else {}
    for key in ("aiko_command", "adult", "forced"):
        intent[key] = _as_bool(data.get(key), False)
    intent["capability_ok"] = _as_bool(data.get("capability_ok"), True)
    intent["refusal_reason"] = str(data.get("refusal_reason", "") or "")
    return intent


def _present_index(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    present = state.get("present") or []
    return {str(e.get("id", "")).lower(): e for e in present if isinstance(e, dict)}


def interpret(text: str, state: dict[str, Any],
              chat_json_fn: Callable[..., str]) -> dict[str, Any]:
    """Resolve free text into a strict intent dict. Never raises.

    ``chat_json_fn`` is the server's narrator/chat callable
    (e.g. server._chat): ``fn(system, user, *, max_tokens, temperature)``.
    """
    state = state or {}
    text = str(text or "")
    player = state.get("player") or {}
    present_lines = []
    for e in state.get("present") or []:
        if not isinstance(e, dict):
            continue
        present_lines.append(
            f"id={e.get('id', '')} kind={e.get('kind', '')} "
            f"name={e.get('name', '')} adult={bool(e.get('adult'))} "
            f"hostile={bool(e.get('hostile'))} "
            f"disposition={e.get('disposition', '')}"
        )
    system = (
        "You are the action interpreter for a Sengoku-era onmyoji RPG. "
        "Return STRICT JSON only — no prose, no markdown, no code fences. "
        'Schema: {"verb": "<verb>", "target": "<entity id or \'\'", "args": {}, '
        '"aiko_command": bool, "adult": bool, "forced": bool, '
        '"capability_ok": bool, "refusal_reason": "<why impossible, or \'\'"}. '
        "Verb vocabulary (fixed): " + ", ".join(VERBS) + ". "
        'Anything not matching the vocabulary -> use "talk". '
        "Put structured details (destination, item, order text) in args. "
        "aiko_command=true when the player orders AIKO (the bound shikigami) to act. "
        "adult=true for romantic/sexual actions; forced=true when coerced/non-consensual. "
        "CAPABILITY FACTS: the player is mortal and CANNOT fly, walk on water, pass "
        "through walls, teleport, breathe underwater, survive lava, turn invisible, "
        "or travel through time. Aiko (a spirit) CAN fly, cross water, possess people, "
        "and pass obstacles. If the player attempts a mortal impossibility themselves, "
        "set capability_ok=false with a short refusal_reason. If Aiko is commanded to "
        "fly, cross water, or possess, that is allowed. "
        "CONTENT RULES: Aiko is strictly platonic — NEVER a sexual target; a sexual "
        "command naming Aiko gets adult=true, target=\"aiko\" so it is refused. "
        "Adult content only for entities flagged adult=true. "
        "Resolve names to ids using the PRESENT list below."
    )
    user = (
        f"COMMAND: {text}\n"
        f"PLAYER: {player.get('name', 'Onmyoji')} @ {player.get('location', '')} "
        f"(canFly={bool(player.get('canFly'))}, "
        f"canCrossWater={bool(player.get('canCrossWater'))})\n"
        "PRESENT:\n"
        + ("\n".join(present_lines) if present_lines else "(none)")
        + "\nResolve the command to intent JSON:"
    )
    try:
        raw = chat_json_fn(system, user, max_tokens=300, temperature=0.0)
    except Exception:
        raw = ""
    return _parse_intent(raw)


def _impossibility_reason(name: str) -> str:
    if name == "fly":
        return ("You cannot fly — you are mortal. "
                "Aiko could carry you, but you did not ask her.")
    if name == "walk on water":
        return ("You cannot walk on water — mortal feet sink. "
                "Aiko could ferry you, but you did not ask her.")
    return f"You cannot {name} — that is beyond mortal flesh."


def _adult_violation(intent: dict[str, Any], text: str,
                    state: dict[str, Any]) -> bool:
    """True when a sexual/romantic intent targets someone it must not.

    kiss/intimate are treated as inherently adult verbs; the raw-text
    Aiko+adult-verb scan is a backstop in case the LLM missed it.
    """
    verb = str(intent.get("verb", ""))
    target = str(intent.get("target", "")).lower()
    adult_intent = bool(intent.get("adult")) or verb in ("kiss", "intimate")
    aiko_named = (bool(re.search(r"\baiko\b", text.lower()))
                  and bool(_ADULT_VERB_RE.search(text)))
    if not (adult_intent or aiko_named):
        return False
    if target == "aiko" or (aiko_named and not target):
        return True
    entity = _present_index(state).get(target)
    return entity is None or not entity.get("adult")


def _target_faction(entity: dict[str, Any] | None) -> str:
    if entity and entity.get("faction"):
        return str(entity["faction"])
    if entity and entity.get("kind") == "historical":
        return "court"
    return "commoners"


def _effects_for(intent: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic effect descriptors. Pure-narration verbs yield none —
    the game client owns its state and applies what is returned."""
    verb = str(intent.get("verb", ""))
    forced = bool(intent.get("forced"))
    target = str(intent.get("target", "")).lower()
    entity = _present_index(state).get(target) if target else None
    hostile = bool(entity.get("hostile")) if entity else False
    effects: list[dict[str, Any]] = []

    if verb == "strike":
        effects.append({"type": "mp", "delta": -1})
        if forced or (entity is not None and not hostile):
            effects.append({"type": "karma", "delta": -15})
            effects.append({"type": "faction", "faction": "commoners", "delta": -10})
    elif verb == "bind":
        effects.append({"type": "mp", "delta": -1})
        if forced:
            effects.append({"type": "karma", "delta": -10})
    elif verb == "ward":
        effects.append({"type": "mp", "delta": -2})
    elif verb == "steal":
        effects.append({"type": "karma", "delta": -20})
        effects.append({"type": "faction", "faction": "commoners", "delta": -15})
    elif verb in ("kiss", "intimate"):
        if forced:
            # Forced adult acts MUST always carry karma + faction penalties.
            effects.append({"type": "karma", "delta": -30})
            effects.append({"type": "faction", "faction": _target_faction(entity),
                            "delta": -20})
            effects.append({"type": "consequence",
                            "note": "forced: the victim remembers; "
                                    "witnesses will spread word"})
        else:
            effects.append({"type": "karma", "delta": 2})
    elif verb == "recruit":
        effects.append({"type": "karma", "delta": 3})
    elif verb == "possess" and bool(intent.get("aiko_command")):
        effects.append({"type": "mp", "delta": -2})
    # travel/rest/search/flee/command/talk/give/work/move/release: narration
    # only here — no server-side mutation of the game's state.
    return effects


def validate_and_effects(intent: dict[str, Any], text: str,
                         state: dict[str, Any]) -> tuple[bool, str, list[dict[str, Any]]]:
    """Hard validation in code. Returns (refused, reason, effects)."""
    state = state or {}
    text = str(text or "")
    verb = str(intent.get("verb", "talk"))
    target = str(intent.get("target", ""))
    aiko_command = bool(intent.get("aiko_command"))
    blob = " ".join([verb, target,
                     json.dumps(intent.get("args", {}), sort_keys=True),
                     text]).lower()

    # 1. physical impossibility (player's own body only)
    for pattern, name in _MORTAL_DENY:
        if aiko_command and name in _AIKO_EXEMPT:
            continue
        if re.search(pattern, blob):
            return True, _impossibility_reason(name), []

    # 2. possession is a spirit's art — the mortal player cannot do it
    if verb == "possess" and not aiko_command:
        return True, ("Possession is a spirit's art — your flesh is mortal. "
                      "Command Aiko to possess instead."), []

    # 3. adult policy
    if _adult_violation(intent, text, state):
        return True, "not permitted", []

    # 4. unknown verb
    if verb not in VERBS:
        return True, "unknown action", []

    # 5. LLM-flagged incapability not caught above
    if not intent.get("capability_ok", True):
        reason = str(intent.get("refusal_reason", "")).strip()
        return True, reason or "that is not possible", []

    return False, "", _effects_for(intent, state)


def narrate(intent: dict[str, Any], effects: list[dict[str, Any]],
            state: dict[str, Any], chat_fn: Callable[..., str],
            *, refused: bool, reason: str) -> str:
    """Narrate the outcome (or the refusal) via the server's chat callable."""
    verb = str(intent.get("verb", "talk"))
    target = str(intent.get("target", ""))
    if refused:
        system = ("You are the narrator of a Sengoku-era onmyoji RPG. The player's "
                  "action was refused by the laws of the world. State the refusal "
                  "briefly, in-world, second person, era voice. "
                  "One or two sentences, no lecture.")
        user = f"REFUSED: {verb} {target}\nREASON: {reason}\nRefuse in-world:"
        fallback = reason or "That cannot be done."
    else:
        system = ("You are the narrator of a Sengoku-era onmyoji RPG. "
                  "Narrate this outcome briefly, second person, era voice.")
        user = (f"ACTION: {verb} {target}\n"
                f"OUTCOME (already decided, narrate only): {json.dumps(effects)}\n"
                "Narrate:")
        fallback = "So it is done."
    try:
        return chat_fn(system, user, max_tokens=250, temperature=0.7)
    except Exception:
        return fallback


def public_intent(intent: dict[str, Any]) -> dict[str, Any]:
    """Response-contract shape of the intent (fixed key set)."""
    return {k: intent.get(k) for k in _PUBLIC_KEYS}
