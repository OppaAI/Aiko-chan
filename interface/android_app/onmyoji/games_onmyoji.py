"""
Onmyoji game API for Aiko (Phase 0 scaffold + playable actions).

Endpoints (mounted at /api/onmyoji):
  GET  /health          — backend slot check
  POST /start          — new journey (Kyoto, eve of Honno-ji)
  GET  /state          — deterministic journey snapshot
  POST /act            — travel | rest | search | ritual (all code-computed)

Hard state lives in state.py and is computed in code only; the LLM
narrator/Aiko voices (Phase 0+) will read this state, never write it.
Auth mirrors the games backends (session with owner fallback).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .state import JourneyState, ROADS, RITUALS, ALL_SKILLS, WORK_TOWNS, AIKO_GREETING
from .state import do_rest, do_ritual, do_search, do_talk, do_train, do_work, do_travel, new_journey, push_dialogue

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/onmyoji", tags=["onmyoji"])

# In-memory journeys keyed by user_id (single worker), same as the games.
_journeys: dict[str, JourneyState] = {}


class StartRequest(BaseModel):
    location: str = "Kyoto"
    date: str = "1582-06-01"


class ActRequest(BaseModel):
    action: str = Field(description="travel | rest | search | talk | train | work | ritual")
    to: str = ""  # travel destination
    ritual: str = ""  # ritual name (ward | bind | purify | banish)
    target: str = ""  # entity id for ritual / talk
    skill: str = ""  # art name for train


class ActResponse(BaseModel):
    journey: JourneyState
    events: list[str] = Field(default_factory=list)


class TalkRequest(BaseModel):
    target: str = Field(default="aiko", description="aiko or a known entity id")
    message: str = Field(description="free text, max 500 chars")


class TalkResponse(BaseModel):
    journey: JourneyState
    reply: str = ""
    events: list[str] = Field(default_factory=list)


async def _require_user(request: Request) -> dict:
    """Session auth with owner fallback for companion apps (same as games)."""
    from interface.webui import auth

    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except HTTPException:
        owner = (os.getenv("AIKO_USER_ID") or "").strip()
        if owner:
            log.warning("Onmyoji session auth failed — falling back to app owner")
            return {"user_id": owner, "username": owner}
        raise
    except Exception as e:
        log.warning("Onmyoji auth failed: %s", e)
        owner = (os.getenv("AIKO_USER_ID") or "").strip()
        if owner:
            return {"user_id": owner, "username": owner}
        raise HTTPException(status_code=401, detail="Authentication required")


@router.get("/health")
async def health(session: dict = Depends(_require_user)):
    return {"ok": True, "game": "onmyoji", "phase": 0}


@router.post("/start", response_model=JourneyState)
async def start_journey(body: StartRequest, session: dict = Depends(_require_user)):
    uid = session["user_id"]
    journey = new_journey()
    journey.location = (body.location or "Kyoto").strip() or "Kyoto"
    journey.date = (body.date or "1582-06-01").strip() or "1582-06-01"
    push_dialogue(journey, "aiko", "you", AIKO_GREETING)
    _journeys[uid] = journey
    log.info("Onmyoji journey started for %s at %s %s", uid, journey.date, journey.location)
    return journey


@router.get("/state", response_model=JourneyState)
async def journey_state(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _journeys:
        raise HTTPException(status_code=404, detail="No journey — call POST /start first")
    return _journeys[uid]


@router.post("/talk", response_model=TalkResponse)
async def talk(body: TalkRequest, session: dict = Depends(_require_user)):
    """Free-text dialogue with Aiko or a known entity (LLM voiced)."""
    uid = session["user_id"]
    if uid not in _journeys:
        raise HTTPException(status_code=404, detail="No journey — call POST /start first")
    journey = _journeys[uid]
    text = (body.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")
    text = text[:500]
    target = (body.target or "aiko").strip().lower()
    if target != "aiko" and not any(e.id == body.target.strip() for e in journey.entities):
        raise HTTPException(status_code=400, detail=f"Unknown entity '{body.target.strip()}' — talk to Aiko or someone met")
    if target != "aiko":
        target = body.target.strip()
    push_dialogue(journey, "you", target, text)
    reply = await _voice_line(journey, target, text)
    push_dialogue(journey, target, "you", reply)
    return TalkResponse(journey=journey, reply=reply)


async def _voice_line(journey: JourneyState, target: str, user_text: str) -> str:
    """One LLM-voiced reply, or a graceful offline fallback."""
    fallback = (
        "…The words hang in the night air. (Aiko's voice is quiet — "
        "the mind behind her is offline.)"
    )
    try:
        from interface.webui import auth
        think = auth.aiko_web_instance._think if auth.aiko_web_instance else None
        if think is None:
            return fallback
    except Exception:
        return fallback
    if target == "aiko":
        persona = (
            "You are Aiko, a playful cat-girl bound as a shikigami "
            "(spirit servant) to the player, an onmyoji in Sengoku Japan. "
            "You act on your own initiative but ask your master for orders "
            "on real decisions. Loyal through the pact, warm within it. "
            f"Pact-bond level {journey.bond}."
        )
        who = "Aiko"
    else:
        ent = next(e for e in journey.entities if e.id == target)
        nature = "a spirit" if ent.kind == "spirit" else "a person"
        flavor = f" Dread {ent.dread}/3." if ent.kind == "spirit" else ""
        persona = (
            f"You are {ent.name or target}, {nature} in Sengoku Japan "
            f"({ent.role or 'wanderer'}; {ent.disposition or 'hard to read'})."
            f"{flavor} Stay in character, brief and vivid."
        )
        who = ent.name or target
    recent = journey.dialogue[-6:]
    history = "\n".join(
        f"{'You' if d.get('who') == 'you' else who}: {d.get('text', '')}" for d in recent
    )
    orders = ""
    if journey.standing_orders:
        orders = "Standing orders: " + "; ".join(journey.standing_orders) + "\n"
    try:
        response = await asyncio.to_thread(
            think._client.chat.completions.create,
            model=think._llm_model,
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": (
                    f"Sengoku Japan, {journey.date}, {journey.location}.\n"
                    f"{orders}"
                    f"Conversation so far:\n{history}\n"
                    f"You say to {who}: {user_text}\n"
                    f"Reply as {who} in 1-3 short sentences."
                )},
            ],
            max_tokens=150,
            timeout=30.0,
        )
        msg = response.choices[0].message
        content = msg.content
        if isinstance(content, list):  # content-block style responses
            content = " ".join(
                (p.get("text", "") if isinstance(p, dict)
                 else getattr(p, "text", "") or "")
                for p in content
            )
        line = (content or "").strip()
        if not line:
            # granite/ministral thinking builds sometimes emit the reply as
            # reasoning only (empty content) — speak that instead of silence.
            line = (getattr(msg, "reasoning_content", "") or "").strip()
        return line or fallback
    except Exception as e:
        log.warning("onmyoji talk voice failed: %r", e)
        return fallback


@router.get("/options")
async def act_options(session: dict = Depends(_require_user)):
    """Valid actions for the phone UI (destinations, rituals)."""
    uid = session["user_id"]
    loc = _journeys[uid].location if uid in _journeys else "Kyoto"
    return {
        "actions": ["travel", "rest", "search", "talk", "train", "work", "ritual"],
        "destinations": sorted(ROADS.get(loc, [])),
        "all_places": sorted(ROADS),
        "rituals": sorted(RITUALS),
        "skills": list(ALL_SKILLS),
        "work_towns": sorted(WORK_TOWNS),
    }


@router.post("/act", response_model=ActResponse)
async def do_act(body: ActRequest, session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _journeys:
        raise HTTPException(status_code=404, detail="No journey — call POST /start first")
    journey = _journeys[uid]
    action = (body.action or "").strip().lower()
    if action == "travel":
        ok, note = do_travel(journey, body.to)
    elif action == "rest":
        ok, note = do_rest(journey)
    elif action == "search":
        ok, note = do_search(journey, random.Random())
    elif action == "talk":
        ok, note = do_talk(journey, body.target)
    elif action == "train":
        ok, note = do_train(journey, body.skill)
    elif action == "work":
        ok, note = do_work(journey)
    elif action == "ritual":
        ok, note = do_ritual(journey, body.ritual, body.target)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action '{body.action}' (travel | rest | search | talk | train | work | ritual)",
        )
    if not ok:
        raise HTTPException(status_code=400, detail=note)
    log.info("Onmyoji act for %s: %s -> %s", uid, action, journey.location)
    return ActResponse(journey=journey, events=[note])
