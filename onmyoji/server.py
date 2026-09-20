"""Onmyoji game server — vertical slice.

FastAPI service (Jetson-local) exposing the game loop to the Android client:

  GET  /state                 full game state (scene + player + quests)
  POST /look                  scene description from the narrator
  POST /talk   {to, text}     dialogue — to "aiko" or a present entity id
  POST /act    {verb, args}   constrained verbs: travel|rest|search|ward|strike|bind|flee|command

Dialogue/acts that create facts (meetings, items, quest stages, morality)
are applied in code AFTER the LLM replies — state.py is the only writer.
Run: python -m onmyoji.server  (UV_PROJECT_ENVIRONMENT must have fastapi/uvicorn/openai)
"""
from __future__ import annotations

import os
from typing import Any

from .gatekeeper import build_aiko_context, build_narrator_context, eligible_figures
from .state import Entity, GameState, GameStore

LLM_MODEL = os.getenv("ONMYOJI_MODEL", os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral")))
LLM_BASE_URL = os.getenv("ONMYOJI_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"))

_STORE_PATH = os.getenv("ONMYOJI_SAVE_PATH", "data/onmyoji_save.json")

_VERBS = ("travel", "rest", "search", "ward", "strike", "bind", "flee", "command")


def _client():
    from openai import OpenAI

    return OpenAI(base_url=LLM_BASE_URL, api_key=os.getenv("LLM_API_KEY", "") or "not-needed")


def _chat(system: str, user: str, *, max_tokens: int = 400, temperature: float = 0.7) -> str:
    resp = _client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _store() -> GameStore:
    return GameStore(_STORE_PATH)


def look() -> dict[str, Any]:
    """Narrate the current scene. Read-only."""
    store = _store()
    context = build_narrator_context(store.state)
    text = _chat(
        "You are the narrator of a Sengoku-era onmyoji RPG. Describe vividly, second person, era voice.",
        context + "\nDescribe what the player sees, hears, smells. End with 2-3 visible options.",
        max_tokens=350,
    )
    return {"ok": True, "text": text, "scene": store.state.scene.__dict__}


def talk(to: str, text: str) -> dict[str, Any]:
    """Dialogue with Aiko or a present entity. May persist a first meeting."""
    store = _store()
    state: GameState = store.state
    target = (to or "").strip()
    if target.lower() == "aiko":
        context = build_aiko_context(state)
        reply = _chat(
            "You are Aiko, bound shikigami. Warm within the pact, loyal by binding. "
            "First person, concise, era voice. Never break role.",
            context + f"\nMASTER SAYS: {text}\nReply as Aiko:",
            max_tokens=300,
        )
        return {"ok": True, "speaker": "aiko", "text": reply}

    entity = state.entities.get(target)
    if entity is None or target not in (state.scene.present_npc_ids + state.scene.present_spirit_ids):
        # First contact: generate a lightweight persona, then persist it.
        context = build_narrator_context(state)
        card = _chat(
            "You are the narrator. Invent ONE brief NPC/spirit persona card as tight JSON: "
            '{"name":..., "kind":"human|spirit", "role":..., "disposition":..., '
            '"speech_style":..., "details":[...]}. Era-plausible, no famous names.',
            context + f"\nA stranger approaches ({target}). Generate their card:",
            max_tokens=250,
            temperature=0.9,
        )
        entity = _persona_from_card(target, card, state)
        store.meet(entity)
        state = store.state
    context = build_narrator_context(state)
    reply = _chat(
        f"You voice {entity.name} ({entity.role}, {entity.disposition}). Speech: {entity.speech_style}. "
        "Stay in character, era voice. Never narrate the player's actions.",
        context + f"\n{entity.name} is spoken to: {text}\nReply as {entity.name}:",
        max_tokens=300,
    )
    return {"ok": True, "speaker": entity.entity_id, "name": entity.name, "text": reply}


def _persona_from_card(entity_id: str, card_json: str, state: GameState) -> Entity:
    import json as _json
    import re as _re

    try:
        match = _re.search(r"\{.*\}", card_json, _re.DOTALL)
        card = _json.loads(match.group(0)) if match else {}
    except (ValueError, AttributeError):
        card = {}
    if not isinstance(card, dict):
        card = {}
    kind = str(card.get("kind", "human")).lower()
    kind = "spirit" if "spirit" in kind else "human"
    details = card.get("details", [])
    return Entity(
        entity_id=entity_id,
        kind=kind,
        name=str(card.get("name", entity_id)).strip() or entity_id,
        role=str(card.get("role", "traveler")).strip(),
        disposition=str(card.get("disposition", "wary")).strip(),
        speech_style=str(card.get("speech_style", "plain speech")).strip(),
        details=[str(d) for d in details if isinstance(d, str)][:3],
    )


# ── constrained verbs: state transitions computed HERE, narrated after ──

def act(verb: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a movement/action verb. Returns narration + resulting state."""
    args = args or {}
    verb = (verb or "").strip().lower()
    if verb not in _VERBS:
        return {"ok": False, "error": f"unknown verb (use one of {', '.join(_VERBS)})"}
    store = _store()
    state = store.state
    effects: list[str] = []

    if verb == "travel":
        dest = str(args.get("to", "")).strip()
        if not dest:
            return {"ok": False, "error": "travel needs {'to': location_id}"}
        store.move(dest, str(args.get("date", "")))
        effects.append(f"traveled to {dest}")
    elif verb == "rest":
        store.shift_bond(+0.02)
        effects.append("rested; bond +0.02")
    elif verb == "search":
        found = str(args.get("find", "old coin")).strip()
        store.give(found)
        effects.append(f"found: {found}")
    elif verb in ("strike", "bind"):
        target = str(args.get("target", "the spirit")).strip()
        dark = bool(args.get("cruel", False))
        if dark:
            store.shift_morality(-0.15, faction="commoners", amount=-0.1)
            effects.append(f"{verb} on {target} (cruel): morality -0.15, commoners -0.10")
        else:
            store.shift_morality(+0.03)
            effects.append(f"{verb} on {target}: resolved, morality +0.03")
    elif verb == "ward":
        effects.append("ward raised (protection for this scene)")
    elif verb == "flee":
        store.shift_morality(-0.02)
        effects.append("fled: morality -0.02")
    elif verb == "command":
        order = str(args.get("order", "")).strip()
        effects.append(f"ordered Aiko: {order or '(standing orders hold)'}")

    context = build_narrator_context(store.state)
    text = _chat(
        "You are the narrator of a Sengoku-era onmyoji RPG. Narrate this outcome briefly, era voice.",
        context + "\nOUTCOME (already decided, narrate only): " + "; ".join(effects),
        max_tokens=250,
    )
    return {"ok": True, "text": text, "effects": effects,
            "scene": store.state.scene.__dict__, "morality": store.state.player.morality}


def figures_here() -> dict[str, Any]:
    """Historical figures legitimately reachable right now (for map markers)."""
    return {"ok": True, "figures": eligible_figures(_store().state)}


def create_app():
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Aiko Onmyoji — game server (slice)")

    class TalkIn(BaseModel):
        to: str = "aiko"
        text: str = ""

    class ActIn(BaseModel):
        verb: str = ""
        args: dict[str, Any] = {}

    @app.get("/state")
    def get_state() -> dict[str, Any]:
        store = _store()
        return {"ok": True, "scene": store.state.scene.__dict__,
                "player": store.state.player.__dict__,
                "quests": {k: v.__dict__ for k, v in store.state.quests.items()},
                "entities": {k: v.__dict__ for k, v in store.state.entities.items()}}

    @app.post("/look")
    def post_look() -> dict[str, Any]:
        return look()

    @app.post("/talk")
    def post_talk(body: TalkIn) -> dict[str, Any]:
        return talk(body.to, body.text)

    @app.post("/act")
    def post_act(body: ActIn) -> dict[str, Any]:
        return act(body.verb, body.args)

    @app.get("/figures")
    def get_figures() -> dict[str, Any]:
        return figures_here()

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0",
                port=int(os.getenv("ONMYOJI_PORT", "8090")))
