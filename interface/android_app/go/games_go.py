"""
Go (囲碁) game API for Aiko.

Endpoints (mounted at /api/games/go):
  POST /start          — start a new game vs Aiko (or practice)
  POST /move           — GTP move (e.g. D4, pass); Aiko replies if vs_ai
  GET  /state          — board stones + status
  GET  /legal-moves    — legal GTP moves (+ pass)
  GET  /engine         — whether KataGo is available
  POST /warmup         — pre-spawn KataGo GTP
  POST /resign         — end the game

Board: pure Python (interface/android_app/go/board.py).
AI: optional KataGo GTP (KATAGO_PATH + KATAGO_MODEL); else random legal.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .board import BLACK, WHITE, GoBoard, color_name
from . import records as _records

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/games/go", tags=["games"])

# Process-local game store. Requires a **single worker process** (or sticky
# sessions). Multi-worker / restart will drop or 404 in-flight games — same
# constraint as interface/android_app/shogi. Shared store is a follow-up if needed.
_games: dict[str, dict] = {}


class StartRequest(BaseModel):
    mode: str = Field(default="vs_ai", description="vs_ai | practice")
    size: int = Field(default=9, description="9 | 13 | 19")
    difficulty: Optional[str] = Field(
        default=None, description="easy | medium | hard"
    )
    side: Optional[str] = Field(
        default=None, description="black (first) | white (second)"
    )
    use_engine: bool = Field(default=True, description="Whether to use the strong engine (KataGo)")


class MoveRequest(BaseModel):
    move: str = Field(..., description="GTP move, e.g. D4 or pass")


class GameState(BaseModel):
    size: int
    turn: str
    last_move: Optional[str] = None
    status: str  # playing | finished | resigned
    mode: str = "vs_ai"
    side: str = "black"
    stones: List[dict] = Field(default_factory=list)
    captured_black: int = 0
    captured_white: int = 0
    ai_comment: Optional[str] = None
    engine: Optional[str] = None
    moves: List[str] = Field(default_factory=list)


def _allow_owner_fallback(_request: Request) -> bool:
    """Gate the AIKO_USER_ID fallback used by the Android companion apps.

    Same-as-Shogi policy: whenever the server owner is configured, the
    companion app (which has no login flow) is treated as the owner,
    regardless of client IP or proxy headers. Keep the server on
    Tailscale-only networking — anyone able to reach these endpoints
    plays Go as the owner.
    """
    return bool((os.getenv("AIKO_USER_ID") or "").strip())


async def _require_user(request: Request) -> dict:
    """Session auth with owner fallback for the Android app (same as Shogi)."""
    from interface.webui import auth

    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except HTTPException:
        owner = (os.getenv("AIKO_USER_ID") or "").strip()
        if owner and _allow_owner_fallback(request):
            log.warning("Go session auth failed — falling back to app owner")
            return {"user_id": owner, "username": owner}
        raise
    except Exception as e:
        # Do not convert unexpected errors into an owner session.
        log.warning("Go auth failed: %s", e)
        raise HTTPException(status_code=401, detail="Authentication required") from e


_DIFFICULTY_PRESETS = {
    "easy": {"blunder": 0.40},
    "medium": {"blunder": 0.12},
    "hard": {"blunder": 0.0},
}
_DEFAULT_DIFFICULTY = "medium"


def difficulty_name(value: Optional[str] = None) -> str:
    raw = (value if value is not None else os.getenv("GO_DIFFICULTY", _DEFAULT_DIFFICULTY))
    raw = (raw or "").strip().lower()
    return raw if raw in _DIFFICULTY_PRESETS else _DEFAULT_DIFFICULTY


def _state_response(uid: str, ai_comment: Optional[str] = None) -> GameState:
    game = _games[uid]
    board: GoBoard = game["board"]
    _maybe_record_go(uid, game, board)
    return GameState(
        size=board.size,
        turn=color_name(board.turn),
        last_move=game.get("last_move"),
        status=board.status if board.status != "playing" else game.get("status", board.status),
        mode=game.get("mode", "vs_ai"),
        side=game.get("side", "black"),
        stones=board.stones_list(),
        captured_black=board.captured.get(BLACK, 0),
        captured_white=board.captured.get(WHITE, 0),
        ai_comment=ai_comment,
        engine=game.get("engine"),
        moves=list(board.history),
    )


def _blunder_rate(difficulty: Optional[str], uid: Optional[str] = None) -> float:
    """Casual-mistake rate nudged by learned bias (clamped, never raises)."""
    base = _DIFFICULTY_PRESETS[difficulty_name(difficulty)]["blunder"]
    try:
        delta = _records.biases(uid).get("blunder_delta", 0.0) if uid else 0.0
    except Exception:
        delta = 0.0
    return min(0.5, max(0.0, base + delta))


def _maybe_record_go(uid: str, game: dict, board: GoBoard) -> None:
    """Persist finished vs_ai games once (learning loop feed)."""
    try:
        if game.get("mode") != "vs_ai" or game.get("recorded"):
            return
        status = board.status if board.status != "playing" else game.get("status", board.status)
        if status in ("playing", None, ""):
            return
        game["recorded"] = True
        user_side = game.get("side", "black")
        you_black = user_side == "black"
        try:
            cap_black = board.captured.get(BLACK, 0)
            cap_white = board.captured.get(WHITE, 0)
        except Exception:
            cap_black = cap_white = 0
        _records.append_match(uid, {
            "difficulty": game.get("difficulty"),
            "size": board.size,
            # resign endpoint: only the human can resign. finished
            # (double pass) has no server scoring -> unknown winner.
            "winner": "aiko" if status == "resigned" else "",
            "moves_made": len(board.history),
            "cap_you": cap_white if you_black else cap_black,
            "cap_aiko": cap_black if you_black else cap_white,
            "engine": game.get("engine", ""),
        })
        _records.maybe_reflect(uid, _learn_think())
    except Exception:
        log.debug("go record failed", exc_info=True)


def _learn_think():
    try:
        from interface.webui import auth

        inst = auth.aiko_web_instance
        return inst._think if inst and inst._think else None
    except Exception:
        return None


def _ai_move(board: GoBoard, difficulty: Optional[str] = None, *,
             uid: Optional[str] = None, use_engine: bool = True):
    """Return (gtp_move | None, engine_name)."""
    legal = board.legal_moves_gtp()
    if not legal:
        return None, None

    if not use_engine:
        # Prefer non-pass when possible for casual play
        non_pass = [m for m in legal if m != "pass"]
        return random.choice(non_pass or legal), "random"

    if random.random() < _blunder_rate(difficulty, uid):
        # Prefer non-pass when possible for casual play
        non_pass = [m for m in legal if m != "pass"]
        return random.choice(non_pass or legal), "random"

    try:
        from . import katago

        if katago.available():
            color = "B" if board.turn == BLACK else "W"
            mv = katago.best_move_gtp(board.size, board.history, color=color)
            if mv:
                # Validate against our rules engine
                try:
                    test = board.copy()
                    test.play_gtp(mv)
                    return mv, "katago"
                except Exception:
                    log.warning("KataGo move illegal under our rules: %s", mv)
    except Exception:
        log.warning("KataGo bridge error", exc_info=True)

    non_pass = [m for m in legal if m != "pass"]
    return random.choice(non_pass or legal), "random"


def _banter_enabled() -> bool:
    """LLM move chatter on/off (GO_BANTER, default on)."""
    return (os.getenv("GO_BANTER", "1") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _banter_frequency() -> float:
    """Odds of a personality line on a quiet move (GO_BANTER_FREQUENCY)."""
    try:
        return max(0.0, min(1.0, float(os.getenv("GO_BANTER_FREQUENCY", "0.25"))))
    except (TypeError, ValueError):
        return 0.25


def _should_speak_go(game: dict, info: dict) -> tuple[bool, str]:
    """Social-intelligence gate for Go banter.

    info keys: status, user_cap, ai_cap (stones captured this round of
    moves), user_pass, move_count. Never raises; missing signals just mean
    fewer triggers — the frequency roll still applies.
    """
    status = info.get("status", "playing")
    if status in ("finished", "resigned"):
        return True, "game just ended"
    if (info.get("user_cap") or 0) >= 2:
        return True, "you just captured Aiko's stones"
    if (info.get("ai_cap") or 0) >= 2:
        return True, "Aiko just captured your stones"
    if info.get("user_pass"):
        return True, "you passed — the endgame is near"
    move_count = info.get("move_count") or 0
    if move_count and move_count % 8 == 0:
        return True, "a few quiet moves have passed"
    if random.random() < _banter_frequency():
        return True, "an ordinary moment"
    return False, ""


def _banter_for_go(
    gtp: str,
    difficulty: Optional[str],
    status: str,
    reason: Optional[str] = None,
    move_count: Optional[int] = None,
    lessons: tuple = (),
) -> Optional[str]:
    """One short Aiko line about her just-played Go move, or None.

    Never raises: any failure (no LLM instance, timeout, empty reply)
    falls back to the template comment. `lessons` are past distilled
    lessons injected by the caller (loaded per game).
    """
    try:
        from interface.webui import auth

        if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
            return None
        think = auth.aiko_web_instance._think
        ending = " and the game just ended" if status != "playing" else ""
        moment = f" Something notable just happened: {reason}." if reason else ""
        where = f" This is move {move_count}." if move_count else ""
        past = ""
        if lessons:
            past = (" Lessons Aiko distilled from your past games:\n" +
                    "\n".join(f"- {t}" for t in lessons[:5]) + "\n")
        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Aiko, OppaAI's AI companion. Your tone is quiet, "
                        "dry, and observant. You are playing go "
                        "(the board game) as White against the user. "
                        f"You just played {gtp} in a {difficulty_name(difficulty)} game{ending}."
                        f"{moment}{where}{past} "
                        "Reply with ONE short playful line (under 20 words, "
                        "English with a touch of Japanese flavor). No board "
                        "analysis, no notation lecture, no quotes."
                    ),
                },
                {"role": "user", "content": "React to your move."},
            ],
            max_tokens=60,
            timeout=30.0,
        )
        text = (response.choices[0].message.content or "").strip().splitlines()
        line = (text[0] if text else "").strip().strip("\"'")[:140]
        return line or None
    except Exception:
        log.debug("Go banter failed, using template comment", exc_info=True)
        return None


@router.get("/engine")
async def engine_status(session: dict = Depends(_require_user)):
    try:
        from . import katago

        return {
            "katago": katago.available(),
            "path": katago.engine_path(),
            "model": katago.model_path(),
            "fallback": "random",
            "difficulty": difficulty_name(),
            "difficulties": sorted(_DIFFICULTY_PRESETS),
            "sizes": [9, 13, 19],
        }
    except Exception as e:
        return {"katago": False, "error": str(e), "fallback": "random"}


@router.post("/warmup")
async def warmup_engine(session: dict = Depends(_require_user)):
    try:
        from . import katago

        if not katago.available():
            return {"warmed": False, "reason": "KataGo not configured"}
        ok = await asyncio.to_thread(katago.ensure_ready)
        return {"warmed": ok}
    except Exception as e:
        return {"warmed": False, "error": str(e)}


@router.post("/start", response_model=GameState)
async def start_game(body: StartRequest, session: dict = Depends(_require_user)):
    uid = session["user_id"]
    size = body.size if body.size in (9, 13, 19) else 9
    mode = body.mode if body.mode in ("vs_ai", "practice") else "vs_ai"
    diff = difficulty_name(body.difficulty)
    side = (body.side or "").strip().lower()
    side = side if side in ("black", "white") else "black"

    board = GoBoard(size=size)
    eng = "random"
    try:
        from . import katago

        eng = "katago" if katago.available() else "random"
    except Exception:
        eng = "random"

    _games[uid] = {
        "board": board,
        "mode": mode,
        "difficulty": diff,
        "side": side,
        "last_move": None,
        "status": "playing",
        "engine": eng,
        "uid": uid,
        "lessons": _records.lesson_texts(uid),
        "use_engine": body.use_engine,
    }

    if side == "black":
        comment = f"Let's play Go ({size}×{size})! You are Black — move first ⚫"
    else:
        comment = f"You are White on {size}×{size} — Aiko moves first ⚪"
    if eng == "katago":
        comment += f" (Aiko asks KataGo · {diff})"
    else:
        comment += " (KataGo offline — casual moves)"

    if side == "white" and mode == "vs_ai":
        ai, eng2 = await asyncio.to_thread(_ai_move, board, diff, uid=uid, use_engine=body.use_engine)
        _games[uid]["engine"] = eng2
        if ai is not None:
            board.play_gtp(ai)
            _games[uid]["last_move"] = ai
            comment = f"Aiko opens with {ai} — your move!"

    log.info(
        "Go game started for %s size=%s mode=%s engine=%s difficulty=%s side=%s",
        uid,
        size,
        mode,
        _games[uid]["engine"],
        diff,
        side,
    )
    return _state_response(uid, ai_comment=comment)


@router.post("/move", response_model=GameState)
async def make_move(body: MoveRequest, session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=400, detail="No active game — call POST /start first")

    game = _games[uid]
    board: GoBoard = game["board"]
    if board.status != "playing" or game.get("status") != "playing":
        raise HTTPException(
            status_code=400,
            detail=f"Game already over: {board.status}",
        )

    user_side = game.get("side", "black")
    user_color = BLACK if user_side == "black" else WHITE
    ai_color = WHITE if user_color == BLACK else BLACK
    move_str = (body.move or "").strip()
    if not move_str:
        raise HTTPException(status_code=400, detail="move is required (GTP, e.g. D4 or pass)")

    if game.get("mode") == "vs_ai" and board.turn != user_color:
        raise HTTPException(status_code=400, detail="Not your turn")

    user_pass = move_str.strip().upper() in ("PASS", "PA")
    try:
        cap_before = dict(board.captured)
    except Exception:
        cap_before = {}
    try:
        board.play_gtp(move_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    user_cap = 0
    try:
        user_cap = max(0, board.captured.get(ai_color, 0) - cap_before.get(ai_color, 0))
    except Exception:
        user_cap = 0

    game["last_move"] = move_str.strip().upper() if move_str.upper() not in ("PASS", "PA") else "pass"
    if board.status != "playing":
        game["status"] = board.status
        return _state_response(uid, ai_comment="Game over.")

    ai_comment = None
    if game["mode"] == "vs_ai" and board.status == "playing":
        ai, engine = await asyncio.to_thread(_ai_move, board, game.get("difficulty"), uid=uid, use_engine=game.get("use_engine", True))
        game["engine"] = engine
        if ai is not None:
            ai_cap = 0
            try:
                ai_cap_before = dict(board.captured)
            except Exception:
                ai_cap_before = {}
            try:
                board.play_gtp(ai)
                try:
                    ai_cap = max(0, board.captured.get(user_color, 0) - ai_cap_before.get(user_color, 0))
                except Exception:
                    ai_cap = 0
                game["last_move"] = ai
                if board.status != "playing":
                    game["status"] = board.status
                if engine == "katago":
                    ai_comment = f"Aiko (via KataGo) plays {ai}"
                else:
                    ai_comment = f"Aiko plays {ai}"
                speak, reason = False, ""
                if _banter_enabled():
                    speak, reason = _should_speak_go(game, {
                        "status": game.get("status", board.status),
                        "user_cap": user_cap,
                        "ai_cap": ai_cap,
                        "user_pass": user_pass,
                        "move_count": len(board.history),
                    })
                if speak:
                    # LLM chatter runs after the move is committed, in a
                    # worker; the template above survives any failure
                    # (including a quiet gate — most moves stay silent).
                    line = await asyncio.to_thread(
                        _banter_for_go, ai, game.get("difficulty"),
                        game.get("status", board.status), reason, len(board.history),
                        tuple(game.get("lessons") or ()),
                    )
                    if line:
                        ai_comment = f"{ai_comment} — {line}"
            except Exception as e:
                log.warning("AI move failed: %s", e)
                ai_comment = "Aiko hesitated… your move again?"

    return _state_response(uid, ai_comment=ai_comment)


@router.get("/state", response_model=GameState)
async def get_state(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    return _state_response(uid)


@router.get("/legal-moves")
async def legal_moves(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    board: GoBoard = _games[uid]["board"]
    return {
        "moves": board.legal_moves_gtp(),
        "turn": color_name(board.turn),
        "status": board.status,
        "size": board.size,
    }


@router.post("/resign")
async def resign(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    board: GoBoard = _games[uid]["board"]
    board.status = "resigned"
    _games[uid]["status"] = "resigned"
    return {"status": "resigned", "message": "You resigned. Aiko wins this one."}
