"""Context gatekeeper — the spoiler control.

Every narrator/Aiko prompt is built HERE, from exactly four code-owned
sources. The rule is ignorance by construction: unvisited places, unmet
people, future anchors, and un-earned items are filtered BEFORE the LLM
ever sees the prompt, so leaks are structurally impossible, not merely
forbidden by instruction.
"""
from __future__ import annotations

import json
from pathlib import Path

from .state import Entity, GameState

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_json(name: str) -> list[dict]:
    path = _DATA_DIR / name
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def eligible_anchors(state: GameState) -> list[dict]:
    """Anchors at/near the player's date and place — the future excluded.

    An anchor is eligible when its date window has opened relative to the
    player's current date (start <= today) and it is either local or
    famous enough to be rumored (``scope`` == "realm"). Anything starting
    after today is cut: the narrator cannot know it.
    """
    today = state.scene.date
    here = state.scene.location_id
    out = []
    for anchor in _load_json("anchors.json"):
        start = str(anchor.get("date_start", ""))
        if start > today:
            continue  # the future: structurally invisible
        place = str(anchor.get("location_id", ""))
        scope = str(anchor.get("scope", "local"))
        if place == here or scope == "realm" or not place:
            out.append(anchor)
    return out


def eligible_figures(state: GameState) -> list[dict]:
    """Historical figures whose active window covers today AND place matches.

    Nobunaga in Azuchi 1578: eligible. Nobunaga in Edo 1600: excluded —
    wrong place, dead man. No row, no meeting, no leak.
    """
    today = state.scene.date
    here = state.scene.location_id
    out = []
    for figure in _load_json("figures.json"):
        if not (str(figure.get("active_from", "")) <= today <= str(figure.get("active_to", ""))):
            continue
        places = figure.get("places", [])
        if here in places or "traveling" in places:
            out.append(figure)
    return out


def present_entities(state: GameState) -> list[Entity]:
    """Only entities already met AND currently present. Strangers don't exist yet."""
    known = state.entities
    out = []
    for eid in state.scene.present_npc_ids + state.scene.present_spirit_ids:
        entity = known.get(eid)
        if entity is not None:
            out.append(entity)
    return out


def build_narrator_context(state: GameState, task: str = "") -> str:
    """Assemble the narrator prompt context. What isn't returned can't leak."""
    lines = [
        f"DATE: {state.scene.date}",
        f"PLACE: {state.scene.location_id}",
    ]
    if state.scene.visible_detail:
        lines.append(f"SCENE: {state.scene.visible_detail}")
    player = state.player
    lines.append(f"PLAYER: {player.name} | morality={player.morality:+.2f} | bond={player.bond:.2f}")
    if player.inventory:
        lines.append("INVENTORY: " + ", ".join(player.inventory))
    if player.completed_quests:
        lines.append("COMPLETED: " + ", ".join(player.completed_quests))
    if player.reputation:
        lines.append("STANDING: " + ", ".join(f"{k}={v:+.2f}" for k, v in sorted(player.reputation.items())))
    for entity in present_entities(state):
        lines.append(
            f"PRESENT [{entity.kind}] {entity.name} ({entity.role}, {entity.disposition}): "
            + (entity.speech_style or "plain speech") + " | "
            + "; ".join(entity.details[:3])
        )
    for anchor in eligible_anchors(state):
        lines.append(
            f"RUMOR/ACTIVE: {anchor.get('title', '')} @ {anchor.get('location_id', '')} "
            f"({anchor.get('date_start', '')}): {anchor.get('hook', '')}"
        )
    for figure in eligible_figures(state):
        lines.append(
            f"FIGURE IN REACH: {figure.get('name', '')} ({figure.get('role', '')}): "
            f"{figure.get('note', '')}"
        )
    if task:
        lines.append(f"TASK: {task}")
    lines.append(
        "RULES: narrate only what is above. Never invent future events, unlisted "
        "people, or items not in INVENTORY. Stay in era voice."
    )
    return "\n".join(lines)


def build_aiko_context(state: GameState, task: str = "") -> str:
    """Aiko's prompt: same world, her bound-shikigami voice, pact-aware."""
    base = build_narrator_context(state, task=task)
    pact = (
        f"PACT: you are Aiko, bound shikigami of {state.player.name}. Loyal by binding, "
        f"warm within it. Bond={state.player.bond:.2f}. You sense spirits the mortal "
        "cannot; bring real decisions to your master, handle routine business alone."
    )
    return pact + "\n" + base
