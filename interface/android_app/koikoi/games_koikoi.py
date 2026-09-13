"""
Koi-Koi (こいこい) game API for Aiko.

Endpoints (mounted at /api/games/koikoi):
  POST /start          — new match vs Aiko (or practice)
  POST /move           — play a hand card / answer flip choice / koi-koi call
  GET  /state          — full table snapshot
  GET  /legal-moves    — playable hand options (+ pending choice, if any)
  GET  /engine         — Aiko heuristic availability
  POST /warmup         — no-op ready check (no engine process)
  POST /resign         — concede the match

Rules engine: interface/android_app/koikoi/cards.py (pure stdlib).
AI: interface/android_app/koikoi/ai.py heuristic ("aiko"), always
available; random legal fallback. No external binary exists for Koi-Koi
comparable to YaneuraOu/KataGo — and none is needed on the Nano.

Match = N months (rounds, 3/6/12). Each round both sides play 8 turns;
completing a yaku offers Koi-koi (continue, stakes ×2) or Stop (bank).
In-memory games keyed by user_id (single worker), same as shogi/go.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from . import ai as _ai
from . import cards as C

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/games/koikoi", tags=["games"])

_games: dict[str, dict] = {}

_MONTH_CHOICES = (3, 6, 12)
_DEFAULT_MONTHS = 6

# Banter configuration
KOIKOI_BANTER_FREQUENCY = float(os.getenv("KOIKOI_BANTER_FREQUENCY", "0.25"))


def _banter_enabled() -> bool:
    """Check if banter is enabled via env var (default on)."""
    return (os.getenv("KOIKOI_BANTER", "1") or "").strip().lower() in {"1", "true", "yes", "on"}


def _should_speak_koikoi(
    game: dict,
    event: str,
    side: str = "aiko",
    yaku_list: Optional[list[dict]] = None,
    cards_taken: int = 0,
    round_month: Optional[int] = None,
) -> bool:
    """Social intelligence gate: decide when Aiko should speak in Koi-Koi.

    Triggers:
    - Always: game end, round end, yaku completion, koi-koi call, tsuki-fuda,
      dealt hand yaku (teshi/kuttsuki), big captures
    - Occasional: personality comments (frequency controlled by env var)
    """
    # Always speak on significant events
    if event in {
        "game_end", "round_end", "yaku_complete", "koi_koi_call",
        "tsuki_fuda", "dealt_yaku", "big_capture", "resign",
    }:
        return True

    # Occasional personality comments
    if event == "turn_start" and random.random() < KOIKOI_BANTER_FREQUENCY:
        return True

    return False


async def _banter_for_koikoi(
    game: dict,
    event: str,
    side: str = "aiko",
    move_usi: Optional[str] = None,
    yaku_list: Optional[list[dict]] = None,
    cards_taken: int = 0,
) -> Optional[str]:
    """Generate Aiko's personality comment for Koi-Koi via LLM.

    Falls back gracefully if LLM unavailable.
    """
    if not _banter_enabled():
        return None

    try:
        from interface.webui import auth
        if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
            return None

        think = auth.aiko_web_instance._think
        difficulty = game.get("difficulty", "medium")
        month = game.get("month", 1)
        months_total = game.get("months", 6)
        you_total = game["totals"].get("you", 0)
        aiko_total = game["totals"].get("aiko", 0)
        koi_count = game.get("koi", 0)
        multiplier = 2 ** koi_count if koi_count > 0 else 1

        # Build context
        yaku_str = ""
        if yaku_list:
            yaku_str = ", ".join(f"{y['name']} ({y['points']}pts)" for y in yaku_list)

        prompt = (
            "You are Aiko, a playful cat-girl AI playing Koi-Koi (hanafuda) "
            "as the opponent. Respond with ONE short, playful line "
            "(under 25 words, English with Japanese flavor). "
            "No analysis, no move notation, just personality.\n\n"
            f"Game: Koi-Koi, Month {month}/{months_total}, Round multiplier ×{multiplier}\n"
            f"Scores: You {you_total} - Aiko {aiko_total}\n"
            f"Event: {event}\n"
            f"Difficulty: {difficulty}\n"
        )
        if yaku_str:
            prompt += f"Yaku completed: {yaku_str}\n"
        if move_usi:
            prompt += f"Move: {move_usi}\n"
        prompt += "\nReact as Aiko:"

        response = await asyncio.to_thread(
            think._client.chat.completions.create,
            model=think._llm_model,
            messages=[
                {"role": "system", "content": "You are Aiko, a playful cat-girl AI playing Koi-Koi."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=60,
            timeout=15.0,
        )
        text = (response.choices[0].message.content or "").strip()
        return text if text else None

    except Exception:
        log.debug("Koi-Koi banter failed, using template", exc_info=True)
        return None


class StartRequest(BaseModel):
    mode: str = Field(default="vs_ai", description="vs_ai | practice")
    difficulty: Optional[str] = Field(default=None, description="easy | medium | hard")
    months: int = Field(default=_DEFAULT_MONTHS, description="3 | 6 | 12")


class MoveRequest(BaseModel):
    hand: Optional[int] = Field(default=None, description="hand card id to play")
    take: list[int] = Field(default_factory=list, description="field ids taken with hand")
    flip_take: list[int] = Field(default_factory=list, description="field ids taken with deck flip")
    decision: Optional[str] = Field(default=None, description="koi | stop")


class CardView(BaseModel):
    id: int
    month: int
    kind: str
    special: str = ""


class YakuView(BaseModel):
    id: str
    name: str
    jp: str
    points: int


class PendingView(BaseModel):
    kind: str  # flip | decision
    flip: Optional[CardView] = None
    options: list[list[CardView]] = Field(default_factory=list)
    new_yaku: list[YakuView] = Field(default_factory=list)
    would_score: int = 0
    multiplier: int = 1


class RoundResult(BaseModel):
    winner: str
    points: int
    base: int
    multiplier: int = 1
    reason: str = "stop"  # stop | exhausted | resigned


class KoiState(BaseModel):
    hand_you: list[CardView] = Field(default_factory=list)
    hand_aiko_count: int = 0
    stock_count: int = 0
    field: list[CardView] = Field(default_factory=list)
    cap_you: list[CardView] = Field(default_factory=list)
    cap_aiko: list[CardView] = Field(default_factory=list)
    yaku_you: list[YakuView] = Field(default_factory=list)
    yaku_aiko: list[YakuView] = Field(default_factory=list)
    totals: dict[str, int] = Field(default_factory=dict)
    month: int = 1
    months: int = _DEFAULT_MONTHS
    oya: str = "you"
    turn: str = "you"
    status: str = "playing"  # playing | finished
    mode: str = "vs_ai"
    pending: Optional[PendingView] = None
    last_play: Optional[dict] = None
    last_flip: Optional[CardView] = None
    round_result: Optional[RoundResult] = None
    winner: Optional[str] = None
    ai_comment: Optional[str] = None
    engine: Optional[str] = None


async def _require_user(request: Request) -> dict:
    """Session auth with owner fallback for the Android app (same as Shogi/Go)."""
    from interface.webui import auth

    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except HTTPException:
        owner = (os.getenv("AIKO_USER_ID") or "").strip()
        if owner:
            log.warning("Koikoi session auth failed — falling back to app owner")
            return {"user_id": owner, "username": owner}
        raise
    except Exception as e:
        log.warning("Koikoi auth failed: %s", e)
        owner = (os.getenv("AIKO_USER_ID") or "").strip()
        if owner:
            return {"user_id": owner, "username": owner}
        raise HTTPException(status_code=401, detail="Authentication required")


def _view(card: int) -> dict:
    m = C.month_of(card)
    return {
        "id": card, "month": m, "kind": C.kind_of(card),
        "special": C.special_of(card),
    }


def _views(cards: list[int]) -> list[dict]:
    return [_view(c) for c in cards]


def _yaku(game: dict, side: str) -> list[dict]:
    return C.detect_yaku(game["cap"][side], game.get("month"))


def _claimed_map(game: dict, side: str) -> dict[str, int]:
    return {y["id"]: y["points"] for y in _yaku(game, side)}


def _new_yaku(game: dict, side: str) -> list[dict]:
    """Completed yaku not yet acknowledged (new ids or grown points)."""
    claimed: dict[str, int] = game["claimed"][side]
    return [y for y in _yaku(game, side) if claimed.get(y["id"]) != y["points"]]


def _ack(game: dict, side: str) -> None:
    game["claimed"][side] = _claimed_map(game, side)


def _mult(game: dict) -> int:
    return 2 ** game["koi"] if game["koi"] > 0 else 1


def _new_round(game: dict) -> None:
    rng: random.Random = game["rng"]
    you, aiko, field, stock = C.deal(rng)
    game["hand"] = {"you": you, "aiko": aiko}
    game["field"] = field
    game["stock"] = stock
    game["cap"] = {"you": [], "aiko": []}
    game["claimed"] = {"you": {}, "aiko": {}}
    game["koi"] = 0
    game["pending"] = None
    game["last_play"] = None
    game["last_flip"] = None
    game["turn"] = game["oya"]


def _advance(game: dict) -> None:
    """After a settled round: next month or match end."""
    if game["month"] >= game["months"]:
        game["status"] = "finished"
        ty, ta = game["totals"]["you"], game["totals"]["aiko"]
        game["winner"] = "you" if ty > ta else ("aiko" if ta > ty else "draw")
        return
    game["month"] += 1
    game["oya"] = game["round_result"]["winner"] if game["round_result"] else game["oya"]
    if game["round_result"] and game["round_result"]["winner"] == "draw":
        pass  # oya keeps deal on exhausted-draw months
    if game.get("mode") == "practice":
        game["oya"] = "you"  # solitaire practice: human always opens
    _new_round(game)
    _auto_dealt(game)


def _check_dealt_yaku(game: dict) -> Optional[tuple[str, dict]]:
    """Teshi/kuttsuki on the just-dealt hands: (side, yaku) or None.

    Teshi outranks kuttsuki; same kind goes to the dealer (oya).
    """
    hits: list[tuple[str, dict, int]] = []
    for side in ("you", "aiko"):
        found = C.detect_hand_yaku(game["hand"][side])
        if found:
            rank = 0 if found[0]["id"] == "teshi" else 1
            hits.append((side, found[0], rank))
    if not hits:
        return None
    hits.sort(key=lambda h: (h[2], 0 if h[0] == game["oya"] else 1))
    side, yaku, _ = hits[0]
    return side, yaku


def _settle_dealt(game: dict, side: str, yaku: dict) -> dict:
    """Bank a dealt yaku (teshi/kuttsuki): no koi-koi possible, ×1."""
    gained = yaku["points"]
    game["totals"][side] += gained
    _ack(game, "you")
    _ack(game, "aiko")
    game["round_result"] = {
        "winner": side, "points": gained, "base": gained,
        "multiplier": 1, "reason": yaku["id"],
    }
    game["pending"] = None
    _advance(game)
    return game["round_result"]


def _auto_dealt(game: dict) -> list[str]:
    """Settle teshi/kuttsuki chains after a deal (recurses via _advance)."""
    notes: list[str] = []
    guard = 0
    while game.get("status") == "playing" and guard < 14:
        guard += 1
        hit = _check_dealt_yaku(game)
        if not hit:
            break
        side, yaku = hit
        res = _settle_dealt(game, side, yaku)
        who = "You are" if side == "you" else "Aiko is"
        notes.append(f"{who} dealt {yaku['name']} ({yaku['jp']}) +{res['points']}! 🎴")
    return notes


def _settle_stop(game: dict, side: str) -> dict:
    base = C.yaku_points(_yaku(game, side))
    mult = _mult(game)
    gained = base * mult
    game["totals"][side] += gained
    _ack(game, "you")
    _ack(game, "aiko")
    game["round_result"] = {
        "winner": side, "points": gained, "base": base,
        "multiplier": mult, "reason": "stop",
    }
    game["pending"] = None
    _advance(game)
    return game["round_result"]


def _settle_exhausted(game: dict) -> dict:
    py = C.yaku_points(_yaku(game, "you"))
    pa = C.yaku_points(_yaku(game, "aiko"))
    if py == pa == 0:
        winner, gained = "draw", 0
    elif py > pa:
        winner, gained = "you", py
    elif pa > py:
        winner, gained = "aiko", pa
    else:
        winner, gained = ("you", py) if game["oya"] == "you" else ("aiko", pa)
    if winner in ("you", "aiko"):
        game["totals"][winner] += gained
    _ack(game, "you")
    _ack(game, "aiko")
    game["round_result"] = {
        "winner": winner, "points": gained, "base": gained,
        "multiplier": 1, "reason": "exhausted",
    }
    game["pending"] = None
    _advance(game)
    return game["round_result"]


def _hands_empty(game: dict) -> bool:
    return not game["hand"]["you"] and not game["hand"]["aiko"]


def _apply_hand_play(game: dict, side: str, hand: int, take: list[int]) -> None:
    game["hand"][side].remove(hand)
    for c in take:
        game["field"].remove(c)
    if take:
        game["cap"][side].extend([hand] + list(take))
    else:
        game["field"].append(hand)
    game["last_play"] = {"by": side, "hand": _view(hand), "take": _views(list(take))}


def _apply_flip(game: dict, side: str, flip: int, take: list[int]) -> None:
    for c in take:
        game["field"].remove(c)
    if take:
        game["cap"][side].extend([flip] + list(take))
    else:
        game["field"].append(flip)
    game["last_flip"] = _view(flip)


def _pending_view(game: dict) -> Optional[dict]:
    p = game.get("pending")
    if not p:
        return None
    if p["kind"] == "flip":
        return {
            "kind": "flip", "flip": _view(p["flip"]),
            "options": [[_view(c) for c in opt] for opt in p["options"]],
            "new_yaku": [], "would_score": 0, "multiplier": _mult(game),
        }
    new = _new_yaku(game, p["side"])
    base = C.yaku_points(_yaku(game, p["side"]))
    return {
        "kind": "decision", "flip": None, "options": [],
        "new_yaku": new, "would_score": base * _mult(game), "multiplier": _mult(game),
    }


def _state_response(uid: str, ai_comment: Optional[str] = None) -> KoiState:
    game = _games[uid]
    return KoiState(
        hand_you=_views(game["hand"]["you"]),
        hand_aiko_count=len(game["hand"]["aiko"]),
        stock_count=len(game["stock"]),
        field=_views(game["field"]),
        cap_you=_views(game["cap"]["you"]),
        cap_aiko=_views(game["cap"]["aiko"]),
        yaku_you=_yaku(game, "you"),
        yaku_aiko=_yaku(game, "aiko"),
        totals=dict(game["totals"]),
        month=game["month"],
        months=game["months"],
        oya=game["oya"],
        turn=game["turn"],
        status=game["status"],
        mode=game.get("mode", "vs_ai"),
        pending=_pending_view(game),
        last_play=game.get("last_play"),
        last_flip=game.get("last_flip"),
        round_result=game.get("round_result"),
        winner=game.get("winner"),
        ai_comment=ai_comment,
        engine=game.get("engine"),
    )


async def _ai_turn(game: dict) -> str:
    """Run Aiko's whole turn (play + flip + koi decision). Returns a note."""
    rng: random.Random = game["rng"]
    diff = game.get("difficulty")
    notes: list[str] = []
    guard = 0
    while game["turn"] == "aiko" and game["status"] == "playing" and guard < 40:
        guard += 1
        if not game["hand"]["aiko"]:
            _settle_exhausted(game)
            notes.append("cards exhausted")
            break
        hand, take = _ai.choose_play(
            game["hand"]["aiko"], game["field"], game["cap"]["aiko"],
            diff, rng, game.get("month"))
        _apply_hand_play(game, "aiko", hand, take)
        if take:
            notes.append(f"Aiko takes {len(take) + 1} with {_card_name(hand)}")
        if game["stock"]:
            flip = game["stock"].pop(0)
            opts = C.match_options(flip, game["field"])
            if not opts:
                _apply_flip(game, "aiko", flip, [])
            else:
                chosen = _ai.choose_flip(
                    flip, opts, game["cap"]["aiko"], diff, rng, game.get("month"))
                _apply_flip(game, "aiko", flip, chosen)
                if chosen:
                    notes.append(f"flip {_card_name(flip)} takes {len(chosen)}")
        new = _new_yaku(game, "aiko")
        if new and game["status"] == "playing":
            cards_left = len(game["hand"]["you"]) + len(game["hand"]["aiko"]) + len(game["stock"])
            call = _ai.choose_decision(
                game["cap"]["aiko"], game["cap"]["you"], game["koi"], cards_left,
                diff, rng, game.get("month"))
            if call == "koi":
                game["koi"] += 1
                _ack(game, "aiko")
                names = ", ".join(y["name"] for y in new)
                notes.append(f"Aiko calls koi-koi on {names}! 🌸 (stakes ×{_mult(game)})")
                if _should_speak_koikoi(game, "koi_koi_call", "aiko", new):
                    line = await _banter_for_koikoi(game, "koi_koi_call", "aiko", yaku_list=new)
                    if line:
                        notes.append(f"🐱 {line}")
            else:
                res = _settle_stop(game, "aiko")
                names = ", ".join(y["name"] for y in new)
                notes.append(f"Aiko stops with {names} — +{res['points']} 🌸")
                if _should_speak_koikoi(game, "yaku_complete", "aiko", new):
                    line = await _banter_for_koikoi(game, "yaku_complete", "aiko", yaku_list=new)
                    if line:
                        notes.append(f"🐱 {line}")
                break
        if _hands_empty(game) and game["status"] == "playing":
            _settle_exhausted(game)
            notes.append("cards exhausted")
            break
        game["turn"] = "you"
    return " · ".join(notes) if notes else "Aiko plays"


async def _drain_aiko(game: dict) -> str:
    """Run Aiko until it is the human's turn (or the match ends).

    Needed whenever a human action leaves turn == "aiko": after her reply,
    and crucially when a new month starts with Aiko dealing — otherwise the
    table would stall with nobody to move.
    """
    notes: list[str] = []
    guard = 0
    while (
        game.get("turn") == "aiko"
        and game.get("status") == "playing"
        and not game.get("pending")
        and guard < 40
    ):
        guard += 1
        note = await _ai_turn(game)
        if note:
            notes.append(note)
    return " · ".join(notes)


def _card_name(card: int) -> str:
    m, (kind, special) = C.month_of(card), C.DECK[C.month_of(card)][card % 4]
    flower = C.MONTHS[m][1]
    return f"{flower} {special or kind}".strip()


@router.get("/engine")
async def engine_status(session: dict = Depends(_require_user)):
    return {
        "aiko": True,
        "kind": "heuristic",
        "fallback": "random",
        "difficulty": _ai.difficulty_name(None),
        "difficulties": sorted(["easy", "medium", "hard"]),
        "months": list(_MONTH_CHOICES),
    }


@router.post("/warmup")
async def warmup_engine(session: dict = Depends(_require_user)):
    # No engine process: the heuristic is always ready.
    return {"warmed": True}


@router.post("/start", response_model=KoiState)
async def start_game(body: StartRequest, session: dict = Depends(_require_user)):
    uid = session["user_id"]
    mode = body.mode if body.mode in ("vs_ai", "practice") else "vs_ai"
    diff = _ai.difficulty_name(body.difficulty)
    months = body.months if body.months in _MONTH_CHOICES else _DEFAULT_MONTHS
    game = {
        "mode": mode, "difficulty": diff, "months": months,
        "month": 1, "oya": "you", "turn": "you", "status": "playing",
        "totals": {"you": 0, "aiko": 0}, "winner": None,
        "engine": "aiko", "round_result": None,
        "rng": random.Random(),
    }
    _games[uid] = game
    _new_round(game)
    notes = _auto_dealt(game)
    comment = (
        f"New {months}-month match — you deal first 🌸 "
        f"(Aiko plays {diff}; her heuristic is always ready)"
    )
    if notes:
        comment += " " + " ".join(notes)
        if _should_speak_koikoi(game, "dealt_yaku", "aiko"):
            line = await _banter_for_koikoi(game, "dealt_yaku", "aiko")
            if line:
                comment += f" 🐱 {line}"
    if game["status"] == "playing" and game["turn"] == "aiko" and mode != "practice":
        extra = await _drain_aiko(game)
        if extra:
            comment += " · " + extra
    log.info("Koikoi match started for %s months=%s difficulty=%s", uid, months, diff)
    return _state_response(uid, ai_comment=comment)


@router.post("/move", response_model=KoiState)
async def make_move(body: MoveRequest, session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=400, detail="No active match — call POST /start first")
    game = _games[uid]
    if game.get("status") != "playing":
        raise HTTPException(status_code=400, detail=f"Match already over: {game.get('winner')}")
    game["round_result"] = None
    rng: random.Random = game["rng"]
    diff = game.get("difficulty")
    practice = game.get("mode") == "practice"

    pending = game.get("pending")

    # --- 1. Pending koi-koi decision -------------------------------------
    if pending and pending["kind"] == "decision":
        call = (body.decision or "").strip().lower()
        if call not in ("koi", "stop"):
            raise HTTPException(status_code=400, detail="Choose koi (continue) or stop (bank points)")
        side = pending["side"]
        if call == "stop":
            res = _settle_stop(game, side)
            who = "You bank" if side == "you" else "Aiko banks"
            comment = f"{who} +{res['points']} ({res['base']}×{res['multiplier']}) 🌸"
            if not practice:
                extra = await _drain_aiko(game)
                if extra:
                    comment += " · " + extra
            return _state_response(uid, ai_comment=comment)
        game["koi"] += 1
        _ack(game, side)
        game["pending"] = None
        who = "You call" if side == "you" else "Aiko calls"
        comment = f"{who} koi-koi! 🌸 stakes now ×{_mult(game)}"
        if not practice:
            if side == "you":
                game["turn"] = "aiko"
                extra = await _drain_aiko(game)
                if extra:
                    comment += " · " + extra
            else:
                game["turn"] = "you"
        return _state_response(uid, ai_comment=comment)

    # --- 2. Pending deck-flip choice --------------------------------------
    if pending and pending["kind"] == "flip":
        if not body.flip_take:
            raise HTTPException(status_code=400, detail="Pick which field cards the flip takes")
        want = sorted(body.flip_take)
        if not any(sorted(o) == want for o in pending["options"]):
            raise HTTPException(status_code=400, detail="That is not a legal take for the flip")
        _apply_flip(game, "you", pending["flip"], list(body.flip_take))
        game["pending"] = None
        new = _new_yaku(game, "you")
        if new:
            game["pending"] = {"kind": "decision", "side": "you"}
            base = C.yaku_points(_yaku(game, "you"))
            names = ", ".join(y["name"] for y in new)
            comment = f"Yaku! {names} ({base} pts) — koi-koi or stop? 🌸"
            if _should_speak_koikoi(game, "yaku_complete", "you", new):
                line = await _banter_for_koikoi(game, "yaku_complete", "you", yaku_list=new)
                if line:
                    comment = f"{comment} 🐱 {line}"
            return _state_response(uid, ai_comment=comment)
        if _hands_empty(game):
            res = _settle_exhausted(game)
            comment = f"Cards exhausted — {res['winner']} takes the month"
            if not practice:
                extra = await _drain_aiko(game)
                if extra:
                    comment += " · " + extra
            return _state_response(uid, ai_comment=comment)
        if not practice:
            game["turn"] = "aiko"
            note = await _drain_aiko(game)
            return _state_response(uid, ai_comment=note or None)
        return _state_response(uid)

    # --- 3. Fresh turn: play a hand card ----------------------------------
    if game.get("turn") != "you":
        raise HTTPException(status_code=400, detail="Not your turn")
    if body.hand is None:
        raise HTTPException(status_code=400, detail="hand card is required (or answer the pending choice)")
    if body.hand not in game["hand"]["you"]:
        raise HTTPException(status_code=400, detail="That card is not in your hand")
    options = C.match_options(body.hand, game["field"])
    if len(options) > 1 and not body.take:
        raise HTTPException(
            status_code=400,
            detail="Two matches — send take: [field id] to choose one",
        )
    take: list[int] = []
    if options:
        if len(options) == 1:
            take = list(options[0])
        else:
            want = sorted(body.take)
            match = next((o for o in options if sorted(o) == want), None)
            if match is None:
                raise HTTPException(status_code=400, detail="Illegal take for that card")
            take = list(match)
    _apply_hand_play(game, "you", body.hand, take)

    if game["stock"]:
        flip = game["stock"].pop(0)
        opts = C.match_options(flip, game["field"])
        if len(opts) > 1:
            game["pending"] = {"kind": "flip", "flip": flip, "options": [list(o) for o in opts]}
            return _state_response(
                uid, ai_comment=f"Deck flips {_card_name(flip)} — pick which cards it takes 🌸")
        chosen = list(opts[0]) if opts else []
        _apply_flip(game, "you", flip, chosen)

    new = _new_yaku(game, "you")
    if new:
        game["pending"] = {"kind": "decision", "side": "you"}
        base = C.yaku_points(_yaku(game, "you"))
        names = ", ".join(y["name"] for y in new)
        comment = f"Yaku! {names} ({base} pts) — koi-koi or stop? 🌸"
        if _should_speak_koikoi(game, "yaku_complete", "you", new):
            line = await _banter_for_koikoi(game, "yaku_complete", "you", yaku_list=new)
            if line:
                comment = f"{comment} 🐱 {line}"
        return _state_response(uid, ai_comment=comment)
    if _hands_empty(game):
        res = _settle_exhausted(game)
        comment = f"Cards exhausted — {res['winner']} takes the month"
        if not practice:
            extra = await _drain_aiko(game)
            if extra:
                comment += " · " + extra
        return _state_response(uid, ai_comment=comment)
    if not practice:
        game["turn"] = "aiko"
        note = await _drain_aiko(game)
        return _state_response(uid, ai_comment=note or None)
    return _state_response(uid)


@router.get("/state", response_model=KoiState)
async def game_state(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active match")
    return _state_response(uid)


@router.get("/legal-moves")
async def legal_moves(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active match")
    game = _games[uid]
    pending = _pending_view(game)
    plays = []
    if game.get("status") == "playing" and not pending and game.get("turn") == "you":
        for h in game["hand"]["you"]:
            opts = C.match_options(h, game["field"])
            plays.append({
                "hand": _view(h),
                "takes": [[_view(c) for c in o] for o in opts] or [[]],
            })
    return {"plays": plays, "pending": pending, "turn": game.get("turn"), "status": game.get("status")}


@router.post("/resign", response_model=KoiState)
async def resign(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active match")
    game = _games[uid]
    game["status"] = "finished"
    game["winner"] = "aiko"
    game["pending"] = None
    game["round_result"] = {
        "winner": "aiko", "points": 0, "base": 0, "multiplier": 1, "reason": "resigned",
    }
    comment = "You resign — Aiko takes the match 🌸"
    if _should_speak_koikoi(game, "resign", "aiko"):
        line = await _banter_for_koikoi(game, "resign", "aiko")
        if line:
            comment = f"{comment} 🐱 {line}"
    return _state_response(uid, ai_comment=comment)


def _self_check() -> None:  # quick stdlib sanity check, see __main__
    rng = random.Random(42)
    you, aiko, field, stock = C.deal(rng)
    assert len(you) == 8 and len(aiko) == 8 and len(field) == 8 and len(stock) == 24
    assert len(set(you + aiko + field + stock)) == 48
    y = C.detect_yaku([CRANE, CURTAIN, MOON, RAIN_MAN, PHOENIX])
    assert any(z["id"] == "goko" and z["points"] == 10 for z in y), y
    y = C.detect_yaku([CRANE, CURTAIN, MOON])
    assert any(z["id"] == "sanko" for z in y), y
    y = C.detect_yaku([MOON, SAKE_CUP])
    assert any(z["id"] == "tsukimi" for z in y), y
    y = C.detect_yaku([BOAR, BUTTERFLY, DEER])
    assert any(z["id"] == "ino-shika-cho" for z in y), y
    poetry = [c for c in range(48) if C.kind_of(c) == "tan-poetry"]
    assert len(poetry) == 3
    y = C.detect_yaku(poetry + [4])
    assert any(z["id"] == "akatan" for z in y), y
    assert C.match_options(0, [4, 8]) == []
    assert C.match_options(0, [1, 5]) == [[1]]
    assert sorted(map(sorted, C.match_options(0, [1, 2]))) == [[1], [2]]
    assert C.match_options(0, [1, 2, 3]) == [[1, 2, 3]]


if __name__ == "__main__":
    _self_check()
    print("koikoi self-check OK")
