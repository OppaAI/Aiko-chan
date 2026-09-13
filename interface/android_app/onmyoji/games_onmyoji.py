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

import logging
import os
import random

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .state import JourneyState, ROADS, RITUALS, do_rest, do_ritual, do_search, do_travel, new_journey

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/onmyoji", tags=["onmyoji"])

# In-memory journeys keyed by user_id (single worker), same as the games.
_journeys: dict[str, JourneyState] = {}


class StartRequest(BaseModel):
    location: str = "Kyoto"
    date: str = "1582-06-01"


class ActRequest(BaseModel):
    action: str = Field(description="travel | rest | search | ritual")
    to: str = ""  # travel destination
    ritual: str = ""  # ritual name (ward | bind | purify | banish)
    target: str = ""  # entity id for ritual


class ActResponse(BaseModel):
    journey: JourneyState
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
    _journeys[uid] = journey
    log.info("Onmyoji journey started for %s at %s %s", uid, journey.date, journey.location)
    return journey


@router.get("/state", response_model=JourneyState)
async def journey_state(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _journeys:
        raise HTTPException(status_code=404, detail="No journey — call POST /start first")
    return _journeys[uid]


@router.get("/options")
async def act_options(session: dict = Depends(_require_user)):
    """Valid actions for the phone UI (destinations, rituals)."""
    uid = session["user_id"]
    loc = _journeys[uid].location if uid in _journeys else "Kyoto"
    return {
        "actions": ["travel", "rest", "search", "ritual"],
        "destinations": sorted(ROADS.get(loc, [])),
        "all_places": sorted(ROADS),
        "rituals": sorted(RITUALS),
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
    elif action == "ritual":
        ok, note = do_ritual(journey, body.ritual, body.target)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action '{body.action}' (travel | rest | search | ritual)",
        )
    if not ok:
        raise HTTPException(status_code=400, detail=note)
    log.info("Onmyoji act for %s: %s -> %s", uid, action, journey.location)
    return ActResponse(journey=journey, events=[note])
