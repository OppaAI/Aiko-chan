"""
Shogi game API for Aiko.

Endpoints (mounted at /api/games/shogi):
  POST /start          — start a new game vs Aiko (or practice)
  POST /move           — play a USI move; Aiko replies if vs_ai
  GET  /state          — current board + status
  GET  /legal-moves    — legal USI moves for the side to move
  GET  /engine         — whether YaneuraOu is available
  POST /warmup         — pre-spawn + handshake YaneuraOu without playing a move
  POST /resign         — end the game

Board state is SFEN (Shogi FEN). Moves use USI, e.g. "7g7f", "B*5e".

AI: Aiko asks YaneuraOu (USI) for the best move when YANEURAOU_PATH
(config/android_app.yaml, or env) is set;
otherwise falls back to a random legal move.

Strength: SHOGI_DIFFICULTY (config/android_app.yaml, or per-game
`difficulty` in POST /start) — easy | medium | hard. Lower levels cap
the search depth and mix in random moves.

Clock: real byoyomi (SHOGI_MAIN_TIME_S + SHOGI_BYOYOMI_S in
config/android_app.yaml). Each side's main time ticks on their moves;
once it hits zero, every move must come within the byoyomi allowance
or that side flags (status "timeout"). 0/0 disables the clock.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from . import records as _records

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/games/shogi", tags=["games"])

# In-memory games keyed by user_id. Fine for single-user / MVP.
_games: dict[str, dict] = {}


def _clock_settings() -> tuple[Optional[float], float]:
    """(main_ms per side or None, byoyomi_ms). (None, 0) = clock off."""
    try:
        main_s = float(os.getenv("SHOGI_MAIN_TIME_S", "3600"))
    except (TypeError, ValueError):
        main_s = 3600.0
    try:
        byoyomi_s = float(os.getenv("SHOGI_BYOYOMI_S", "60"))
    except (TypeError, ValueError):
        byoyomi_s = 60.0
    main_ms = max(0.0, main_s) * 1000.0
    byoyomi_ms = max(0.0, byoyomi_s) * 1000.0
    if main_ms <= 0 and byoyomi_ms <= 0:
        return None, 0.0
    return main_ms, byoyomi_ms


def _new_clock() -> Optional[dict]:
    """Fresh clock state, or None when the clock is disabled."""
    main_ms, byoyomi_ms = _clock_settings()
    if main_ms is None:
        return None
    return {
        "remaining": {"black": main_ms, "white": main_ms},
        "byoyomi_ms": byoyomi_ms,
        "stamp": time.monotonic(),
    }


def _charge_clock(game: dict, side: str, elapsed_ms: float) -> bool:
    """Deduct elapsed_ms from side's clock. False = flagged.

    Main time first; once it is gone, the move must still fall inside
    one byoyomi period (classic single-period byoyomi).
    """
    clock = game.get("clock")
    if not clock:
        return True
    remaining = clock["remaining"][side] - elapsed_ms
    if remaining >= 0:
        clock["remaining"][side] = remaining
        return True
    if elapsed_ms <= clock["byoyomi_ms"]:
        clock["remaining"][side] = 0.0
        return True
    clock["remaining"][side] = 0.0
    return False


def _clock_view(game: dict) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """(black_ms, white_ms, byoyomi_ms) snapshot for API responses."""
    clock = game.get("clock")
    if not clock:
        return None, None, None
    return (
        max(0, int(clock["remaining"]["black"])),
        max(0, int(clock["remaining"]["white"])),
        int(clock["byoyomi_ms"]),
    )


class StartRequest(BaseModel):
    mode: str = Field(default="vs_ai", description="vs_ai | practice")
    difficulty: Optional[str] = Field(
        default=None, description="easy | medium | hard (None = server default)"
    )
    side: Optional[str] = Field(
        default=None, description="black (first, 先手) | white (second, 後手)"
    )
    use_engine: bool = Field(default=True, description="Whether to use the strong engine (YaneuraOu)")


class MoveRequest(BaseModel):
    move: str = Field(..., description="USI move, e.g. 7g7f or B*5e")


class GameState(BaseModel):
    sfen: str
    turn: str  # "black" (user / 先手) or "white" (Aiko / 後手)
    last_move: Optional[str] = None
    status: str  # playing | checkmate | stalemate | draw | resigned | timeout
    mode: str = "vs_ai"
    side: str = "black"  # user's color: "black" (first, 先手) or "white" (second, 後手)
    ai_comment: Optional[str] = None
    engine: Optional[str] = None  # "yaneuraou" | "random" | None
    clock_black_ms: Optional[int] = None  # remaining main time (None = no clock)
    clock_white_ms: Optional[int] = None
    byoyomi_ms: Optional[int] = None


async def _require_user(request: Request) -> dict:
    """Session auth with owner fallback for the Android app.

    Mirrors lingo's get_lingo_session: the Aiko-Shogi client has no
    CookieJar/login flow, so unauthenticated calls fall back to the app
    owner (AIKO_USER_ID) instead of hard-401ing every move.
    """
    from interface.webui import auth

    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except HTTPException:
        # Auth attempted but rejected (no/invalid cookie, terms unaccepted)
        # — fall back to owner before giving up.
        owner = (os.getenv("AIKO_USER_ID") or "").strip()
        if owner:
            log.warning("Shogi session auth failed — falling back to app owner")
            return {"user_id": owner, "username": owner}
        raise
    except Exception as e:
        log.warning("Shogi auth failed: %s", e)
        owner = (os.getenv("AIKO_USER_ID") or "").strip()
        if owner:
            return {"user_id": owner, "username": owner}
        raise HTTPException(status_code=401, detail="Authentication required")


def _import_shogi():
    try:
        import shogi

        return shogi
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="python-shogi is not installed. Run: pip install python-shogi",
        ) from e


def _status_for(board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    try:
        if board.is_fourfold_repetition():
            return "draw"
    except AttributeError:
        pass
    if board.is_game_over():
        return "draw"
    return "playing"


def _turn_label(board) -> str:
    shogi = _import_shogi()
    return "black" if board.turn == shogi.BLACK else "white"


# Difficulty presets: depth caps the engine search (USI `go depth`),
# movetime_ms bounds the read loop (None = YANEURAOU_MOVETIME_MS),
# blunder = probability of a uniform-random legal move instead of asking
# the engine at all (skips the search, saving Nano CPU too).
_DIFFICULTY_PRESETS: dict[str, dict] = {
    "easy": {"depth": 3, "movetime_ms": 150, "blunder": 0.35},
    "medium": {"depth": 6, "movetime_ms": 400, "blunder": 0.10},
    "hard": {"depth": None, "movetime_ms": None, "blunder": 0.0},
}

_DEFAULT_DIFFICULTY = "medium"


def difficulty_name(value: Optional[str] = None) -> str:
    """Normalize a difficulty level; unknown/blank falls back to default."""
    raw = (value if value is not None else os.getenv("SHOGI_DIFFICULTY", _DEFAULT_DIFFICULTY))
    raw = (raw or "").strip().lower()
    return raw if raw in _DIFFICULTY_PRESETS else _DEFAULT_DIFFICULTY


def _engine_bridge():
    """Resolve the YaneuraOu bridge module (separate hook so tests can patch it)."""
    try:
        from . import yaneuraou

        return yaneuraou
    except ImportError:
        return None


def _banter_enabled() -> bool:
    """LLM move chatter on/off (SHOGI_BANTER, default on — shogi is slow anyway)."""
    return (os.getenv("SHOGI_BANTER", "1") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _banter_frequency() -> float:
    """Odds of a personality line on a quiet move (SHOGI_BANTER_FREQUENCY)."""
    try:
        return max(0.0, min(1.0, float(os.getenv("SHOGI_BANTER_FREQUENCY", "0.25"))))
    except (TypeError, ValueError):
        return 0.25


# Rough piece values for "big capture" praise (SFEN letters).
_PIECE_VALUES = {"P": 1, "L": 3, "N": 4, "S": 5, "G": 6, "B": 9, "R": 10, "K": 99}


def _phase_of(move_number: int) -> str:
    if move_number < 16:
        return "opening"
    if move_number < 50:
        return "middlegame"
    return "endgame"


def _should_speak_shogi(game: dict, info: dict) -> tuple[bool, str]:
    """Social-intelligence gate: speak on notable moments, else occasionally.

    info keys: status, phase_changed, user_cap, ai_cap (piece values),
    user_check, ai_check, promoted, dropped, move_number.
    Never raises; missing signals (e.g. stub boards in tests) just mean
    fewer triggers — the frequency roll still applies.
    """
    status = info.get("status", "playing")
    if status in ("checkmate", "resigned", "timeout", "draw", "stalemate"):
        return True, "game just ended"
    if info.get("phase_changed"):
        return True, f"game entered the {info.get('phase', 'next phase')}"
    if (info.get("user_cap") or 0) >= 9:
        return True, "you captured Aiko's rook or bishop"
    if (info.get("ai_cap") or 0) >= 9:
        return True, "Aiko captured your rook or bishop"
    if info.get("user_check") or info.get("ai_check"):
        return True, "a check was just given"
    if info.get("promoted"):
        return True, "a piece just promoted"
    if info.get("dropped"):
        return True, "a piece was just dropped from hand"
    move_number = info.get("move_number") or 0
    if move_number and move_number % 6 == 0:
        return True, "a few quiet moves have passed"
    if random.random() < _banter_frequency():
        return True, "an ordinary moment"
    return False, ""


def _banter_for(
    usi: str,
    difficulty: Optional[str],
    status: str,
    reason: Optional[str] = None,
    phase: Optional[str] = None,
    lessons: tuple = (),
) -> Optional[str]:
    """One short Aiko line about her just-played move, or None.

    Never raises and never blocks the game: any failure (no LLM
    instance, timeout, empty reply) falls back to the template comment.
    `reason` (why this moment is notable), `phase`, and past `lessons`
    come from the social-intelligence gate; all optional.
    """
    try:
        from interface.webui import auth

        if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
            return None
        think = auth.aiko_web_instance._think
        ending = " and checkmated the user" if status == "checkmate" else ""
        moment = f" Something notable just happened: {reason}." if reason else ""
        where = f" The game is in the {phase}." if phase else ""
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
                        "dry, and observant. You are playing shogi "
                        "(Japanese chess) as White against the user. "
                        f"You just played {usi} in a {difficulty_name(difficulty)} game{ending}."
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
        log.debug("Shogi banter failed, using template comment", exc_info=True)
        return None


def _blunder_rate(difficulty: Optional[str], uid: Optional[str] = None) -> float:
    """Casual-mistake rate nudged by learned bias (clamped, never raises)."""
    base = _DIFFICULTY_PRESETS[difficulty_name(difficulty)]["blunder"]
    try:
        delta = _records.biases(uid).get("blunder_delta", 0.0) if uid else 0.0
    except Exception:
        delta = 0.0
    return min(0.5, max(0.0, base + delta))


def _maybe_record_shogi(uid: str, game: dict, board) -> None:
    """Persist finished vs_ai games once (learning loop feed)."""
    try:
        if game.get("mode") != "vs_ai" or game.get("recorded"):
            return
        status = game.get("status", "playing")
        if status in ("playing", None, ""):
            return
        game["recorded"] = True
        user_side = game.get("side", "black")
        if status == "checkmate":
            turn = _turn_label(board)
            winner_side = "white" if turn == "black" else "black"
            winner = "you" if winner_side == user_side else "aiko"
        elif status == "resigned":
            winner = "aiko"  # only the human can resign via the endpoint
        elif status == "timeout":
            flagged = game.get("flagged", "you")
            winner = "aiko" if flagged == "you" else "you"
        else:  # draw, stalemate
            winner = "draw"
        try:
            moves_made = int(getattr(board, "move_number", 0) or 0)
        except Exception:
            moves_made = 0
        _records.append_match(uid, {
            "difficulty": game.get("difficulty"),
            "winner": winner,
            "moves_made": moves_made,
            "end": status,
            "engine": game.get("engine", ""),
        })
        _records.maybe_reflect(uid, _learn_think())
    except Exception:
        log.debug("shogi record failed", exc_info=True)


def _learn_think():
    try:
        from interface.webui import auth

        inst = auth.aiko_web_instance
        return inst._think if inst and inst._think else None
    except Exception:
        return None


def _ai_move(
    board,
    difficulty: Optional[str] = None,
    movetime_cap_ms: Optional[float] = None,
    *,
    uid: Optional[str] = None,
    use_engine: bool = True,
):
    """
    Aiko asks YaneuraOu for the right move when available;
    otherwise picks a random legal move.
    Returns (move | None, engine_name).

    movetime_cap_ms bounds the search so the engine can never flag
    itself when the clock is on (hard mode only; depth-capped levels
    finish in milliseconds anyway).
    """
    shogi = _import_shogi()
    legal = list(board.legal_moves)
    if not legal:
        return None, None

    if not use_engine:
        return random.choice(legal), "random"

    preset = _DIFFICULTY_PRESETS[difficulty_name(difficulty)]
    movetime_ms = preset["movetime_ms"]
    if preset["depth"] is None and movetime_cap_ms is not None:
        # best_move_usi() clamps this again via normalized_movetime_ms.
        if movetime_ms is None or movetime_cap_ms < movetime_ms:
            movetime_ms = max(50, int(movetime_cap_ms))

    # 0) Human-like blunder: occasional random move, no engine search.
    if random.random() < _blunder_rate(difficulty, uid):
        return random.choice(legal), "random"

    # 1) Ask YaneuraOu (depth-capped unless hard)
    try:
        bridge = _engine_bridge()
        usi = (
            bridge.best_move_usi(
                board.sfen(),
                movetime_ms=movetime_ms,
                depth=preset["depth"],
            )
            if bridge is not None
            else None
        )
        if usi:
            try:
                mv = shogi.Move.from_usi(usi)
                if mv in board.legal_moves:
                    return mv, "yaneuraou"
                log.warning("YaneuraOu returned illegal move %s — ignoring", usi)
            except Exception:
                log.warning("Could not parse YaneuraOu move %s", usi, exc_info=True)
    except Exception:
        log.warning("YaneuraOu bridge error", exc_info=True)

    # 2) Fallback
    return random.choice(legal), "random"


def _state_response(
    uid: str,
    ai_comment: Optional[str] = None,
) -> GameState:
    game = _games[uid]
    board = game["board"]
    _maybe_record_shogi(uid, game, board)
    black_ms, white_ms, byoyomi_ms = _clock_view(game)
    return GameState(
        sfen=board.sfen(),
        turn=_turn_label(board),
        last_move=game.get("last_move"),
        status=game.get("status", _status_for(board)),
        mode=game.get("mode", "vs_ai"),
        side=game.get("side", "black"),
        ai_comment=ai_comment,
        engine=game.get("engine"),
        clock_black_ms=black_ms,
        clock_white_ms=white_ms,
        byoyomi_ms=byoyomi_ms,
    )


@router.get("/engine")
async def engine_status(session: dict = Depends(_require_user)):
    """Report whether YaneuraOu is configured and runnable."""
    try:
        from . import yaneuraou

        path = yaneuraou.engine_path()
        ok = yaneuraou.available()
        return {
            "yaneuraou": ok,
            "path": path,
            "movetime_ms": yaneuraou.normalized_movetime_ms(),
            "fallback": "random",
            "difficulty": difficulty_name(),
            "difficulties": sorted(_DIFFICULTY_PRESETS),
        }
    except Exception as e:
        return {"yaneuraou": False, "error": str(e), "fallback": "random"}


@router.post("/warmup")
async def warmup_engine(session: dict = Depends(_require_user)):
    """
    Pre-spawn YaneuraOu and complete its USI handshake now, without playing
    a move. Meant to be called when the app opens (or the lobby loads) so
    the process is already up and ready by the time the user starts a game
    — the first real move then has no extra spawn/handshake latency.

    Safe to call anytime, including with no active game and repeatedly;
    it's a fast no-op if the engine is already warm.
    """
    try:
        from . import yaneuraou

        if not yaneuraou.available():
            return {"warmed": False, "reason": "engine binary not found/configured"}
        ok = await asyncio.to_thread(yaneuraou.ensure_ready)
        return {"warmed": ok}
    except Exception as e:
        log.warning("Engine warmup failed: %s", e)
        return {"warmed": False, "error": str(e)}


@router.post("/start", response_model=GameState)
async def start_game(body: StartRequest, session: dict = Depends(_require_user)):
    """Start a new Shogi game. Default: user is 先手 (black, first move)."""
    shogi = _import_shogi()
    uid = session["user_id"]
    mode = body.mode if body.mode in ("vs_ai", "practice") else "vs_ai"
    diff = difficulty_name(body.difficulty)
    side = (body.side or "").strip().lower()
    side = side if side in ("black", "white") else "black"
    _games[uid] = {
        "board": shogi.Board(),
        "mode": mode,
        "difficulty": diff,
        "side": side,
        "clock": _new_clock(),
        "last_move": None,
        "status": "playing",
        "uid": uid,
        "lessons": _records.lesson_texts(uid),
        "use_engine": body.use_engine,
    }
    eng = None
    try:
        from . import yaneuraou

        eng = "yaneuraou" if yaneuraou.available() else "random"
    except Exception:
        eng = "random"
    _games[uid]["engine"] = eng
    if side == "black":
        comment = "Let's play Shogi! You move first ♟️"
    else:
        comment = "You are 後手 (White) — Aiko moves first ♟️"
    if eng == "yaneuraou":
        comment += f" (Aiko will ask YaneuraOu for {diff} moves)"
    else:
        comment += " (engine offline — Aiko plays casual moves)"
    if side == "white" and mode == "vs_ai":
        # Aiko (black) opens immediately so it is the user's turn.
        ai, eng2 = await asyncio.to_thread(_ai_move, _games[uid]["board"], diff, None, uid=uid, use_engine=body.use_engine)
        _games[uid]["engine"] = eng2
        if ai is not None:
            _games[uid]["board"].push(ai)
            _games[uid]["last_move"] = ai.usi()
            _games[uid]["status"] = _status_for(_games[uid]["board"])
            comment = f"Aiko opens with {ai.usi()} — your move!"
    log.info(
        "Shogi game started for %s mode=%s engine=%s difficulty=%s side=%s",
        uid,
        mode,
        _games[uid]["engine"],
        diff,
        side,
    )
    return _state_response(uid, ai_comment=comment)


@router.post("/move", response_model=GameState)
async def make_move(body: MoveRequest, session: dict = Depends(_require_user)):
    """Apply user USI move; if vs_ai and still playing, Aiko replies."""
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=400, detail="No active game — call POST /start first")

    game = _games[uid]
    board = game["board"]
    if game.get("status") != "playing":
        raise HTTPException(status_code=400, detail=f"Game already over: {game['status']}")

    user_side = game.get("side", "black")
    ai_side = "white" if user_side == "black" else "black"

    move_str = (body.move or "").strip()
    if not move_str:
        raise HTTPException(status_code=400, detail="move is required (USI, e.g. 7g7f)")

    shogi = _import_shogi()
    try:
        move = shogi.Move.from_usi(move_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid USI move: {e}") from e

    if move not in board.legal_moves:
        raise HTTPException(status_code=400, detail=f"Illegal move: {move_str}")

    if game.get("mode") == "vs_ai" and _turn_label(board) != user_side:
        raise HTTPException(status_code=400, detail="Not your turn")

    # Charge the user's clock for time since the last stamp.
    now = time.monotonic()
    if game.get("clock"):
        elapsed_ms = (now - game["clock"]["stamp"]) * 1000.0
        if not _charge_clock(game, user_side, elapsed_ms):
            game["status"] = "timeout"
            game["flagged"] = user_side
            log.info("Shogi flag: %s ran out of time", uid)
            return _state_response(uid, ai_comment="Flag! You ran out of time — Aiko wins. ⏰")
        game["clock"]["stamp"] = now

    user_cap = 0
    to_sq = getattr(move, "to_square", None)
    if to_sq is not None and hasattr(board, "piece_at"):
        try:
            target = board.piece_at(to_sq)
            if target is not None:
                user_cap = _PIECE_VALUES.get(str(target).upper(), 0)
        except Exception:
            user_cap = 0

    board.push(move)
    game["last_move"] = move_str
    status = _status_for(board)
    game["status"] = status
    try:
        user_check = bool(board.is_check())
    except Exception:
        user_check = False

    ai_comment = None
    engine = None
    if game["mode"] == "vs_ai" and status == "playing":
        cap_ms = None
        if game.get("clock"):
            cap_ms = max(50.0, game["clock"]["remaining"][ai_side] - 100.0)
        ai_start = time.monotonic()
        ai, engine = await asyncio.to_thread(
            _ai_move, board, game.get("difficulty"), cap_ms, uid=uid, use_engine=game.get("use_engine", True)
        )
        if game.get("clock"):
            ai_elapsed_ms = (time.monotonic() - ai_start) * 1000.0
            if not _charge_clock(game, ai_side, ai_elapsed_ms):
                game["status"] = "timeout"
                game["flagged"] = ai_side
                game["engine"] = engine
                return _state_response(uid, ai_comment="Flag! Aiko ran out of time — you win! ⏰")
            game["clock"]["stamp"] = time.monotonic()
        game["engine"] = engine
        if ai is not None:
            ai_cap = 0
            ai_to = getattr(ai, "to_square", None)
            if ai_to is not None and hasattr(board, "piece_at"):
                try:
                    ai_target = board.piece_at(ai_to)
                    if ai_target is not None:
                        ai_cap = _PIECE_VALUES.get(str(ai_target).upper(), 0)
                except Exception:
                    ai_cap = 0
            board.push(ai)
            usi = ai.usi()
            game["last_move"] = usi
            game["status"] = _status_for(board)
            try:
                ai_check = bool(board.is_check())
            except Exception:
                ai_check = False
            move_number = getattr(board, "move_number", 0) or 0
            phase = _phase_of(move_number) if move_number else None
            phase_changed = bool(phase) and game.get("phase") != phase
            if phase:
                game["phase"] = phase
            if engine == "yaneuraou":
                ai_comment = f"Aiko (via YaneuraOu) plays {usi}"
            else:
                ai_comment = f"Aiko plays {usi}"
            if game["status"] == "checkmate":
                ai_comment += " — checkmate!"
            speak, reason = False, ""
            if _banter_enabled():
                speak, reason = _should_speak_shogi(game, {
                    "status": game["status"],
                    "phase": phase,
                    "phase_changed": phase_changed,
                    "user_cap": user_cap,
                    "ai_cap": ai_cap,
                    "user_check": user_check,
                    "ai_check": ai_check,
                    "promoted": usi.endswith("+") or move_str.endswith("+"),
                    "dropped": "*" in usi or "*" in move_str,
                    "move_number": move_number,
                })
            if speak:
                # LLM chatter runs after the move is committed, in a worker
                # so the event loop stays free; template above survives any
                # failure (including a quiet gate — most moves stay silent).
                line = await asyncio.to_thread(
                    _banter_for, usi, game.get("difficulty"), game["status"],
                    reason, phase, tuple(game.get("lessons") or ()),
                )
                if line:
                    ai_comment = f"{ai_comment} — {line}"

    return _state_response(uid, ai_comment=ai_comment)


@router.get("/state", response_model=GameState)
async def game_state(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    return _state_response(uid)


@router.get("/legal-moves")
async def legal_moves(session: dict = Depends(_require_user)):
    """List legal USI moves for the current side to move."""
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    board = _games[uid]["board"]
    return {
        "moves": [m.usi() for m in board.legal_moves],
        "turn": _turn_label(board),
        "status": _games[uid].get("status", _status_for(board)),
    }


@router.post("/resign", response_model=GameState)
async def resign(session: dict = Depends(_require_user)):
    uid = session["user_id"]
    if uid not in _games:
        raise HTTPException(status_code=404, detail="No active game")
    _games[uid]["status"] = "resigned"
    comment = None
    if _banter_enabled():
        speak, reason = _should_speak_shogi(
            _games[uid], {"status": "resigned"})
        if speak:
            comment = await asyncio.to_thread(
                _banter_for, "", _games[uid].get("difficulty"),
                "resigned", reason or "you resigned", None,
                tuple(_games[uid].get("lessons") or ()),
            )
    return _state_response(uid, ai_comment=comment)


# ── self-play sessions (Aiko vs YaneuraOu, background thread + polling) ─────
# Games take minutes (a Jev call per Aiko move), so start() returns fast and
# the app polls state(). One active session per user; stop() halts gracefully.

_selfplay: dict[str, dict] = {}
_selfplay_lock: threading.Lock | None = None


def _sp_lock() -> threading.Lock:
    global _selfplay_lock
    if _selfplay_lock is None:
        import threading as _th
        _selfplay_lock = _th.Lock()
    return _selfplay_lock


class SelfplayStartRequest(BaseModel):
    games: int = 1


class SelfplayState(BaseModel):
    running: bool = False
    game_index: int = 0
    games_total: int = 0
    sfen: str = ""
    moves: list[str] = []
    status: str = "idle"
    last_winner: str = ""
    last_end: str = ""
    aiko_side: str = ""
    aiko_wins: int = 0
    engine_wins: int = 0
    draws: int = 0
    matches: int = 0


def _selfplay_stats(uid: str) -> dict:
    try:
        from interface.android_app import learn as _learn
        st = _learn.stats(uid, "shogi_selfplay")
        engine = sum(1 for r in _learn.load_recent(uid, "shogi_selfplay", 200)
                     if r.get("winner") == "engine")
        return {"aiko_wins": int(st.get("aiko_wins", 0)), "engine_wins": engine,
                "draws": int(st.get("draws", 0)), "matches": int(st.get("matches", 0))}
    except Exception:
        return {"aiko_wins": 0, "engine_wins": 0, "draws": 0, "matches": 0}


def _selfplay_snapshot(uid: str) -> SelfplayState:
    sess = _selfplay.get(uid) or {}
    st = _selfplay_stats(uid)
    # Current game's color follows the same alternation rule play_game uses:
    # recorded (completed) self-play games decide it, so the running game
    # and this snapshot always agree.
    completed = int(st.get("matches", 0))
    return SelfplayState(
        running=bool(sess.get("running", False)),
        game_index=int(sess.get("game_index", 0)),
        games_total=int(sess.get("games_total", 0)),
        sfen=str(sess.get("sfen", "")),
        moves=list(sess.get("moves", [])),
        status=str(sess.get("status", "idle")),
        last_winner=str(sess.get("last_winner", "")),
        last_end=str(sess.get("last_end", "")),
        aiko_side="black" if completed % 2 == 0 else "white",
        aiko_wins=st["aiko_wins"], engine_wins=st["engine_wins"],
        draws=st["draws"], matches=st["matches"],
    )


def _run_selfplay(uid: str, games: int) -> None:
    from . import selfplay as _sp
    try:
        for i in range(max(1, games)):
            with _sp_lock():
                sess = _selfplay.get(uid)
                if not sess or sess.get("stop"):
                    break
                sess["game_index"] = i + 1
                sess["status"] = "playing"

            def _on_ply(sfen: str, moves: list[str], _i=i) -> None:
                with _sp_lock():
                    sess = _selfplay.get(uid)
                    if sess is not None:
                        sess["sfen"] = sfen
                        sess["moves"] = moves

            def _is_stopped() -> bool:
                with _sp_lock():
                    sess = _selfplay.get(uid)
                    return sess is None or bool(sess.get("stop"))

            out = _sp.play_game(uid, on_ply=_on_ply, is_stopped=_is_stopped)
            with _sp_lock():
                sess = _selfplay.get(uid)
                if sess is not None:
                    sess["last_winner"] = str(out.get("winner", ""))
                    sess["last_end"] = str(out.get("end", ""))
                    sess["status"] = "stopped" if out.get("end") == "stopped" else "game-done"
                    if sess.get("stop"):
                        break
    except Exception as exc:
        log.warning("selfplay session failed for %s: %s", uid, exc)
        with _sp_lock():
            sess = _selfplay.get(uid)
            if sess is not None:
                sess["status"] = f"error: {type(exc).__name__}"
    finally:
        with _sp_lock():
            sess = _selfplay.get(uid)
            if sess is not None:
                sess["running"] = False


@router.post("/selfplay/start", response_model=SelfplayState)
async def selfplay_start(body: SelfplayStartRequest, session: dict = Depends(_require_user)):
    """Start a background Aiko-vs-engine self-play session (fast return)."""
    uid = session["user_id"]
    try:
        from . import yaneuraou
        if not yaneuraou.available():
            raise HTTPException(status_code=503, detail="YaneuraOu engine offline (set YANEURAOU_PATH)")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"engine check failed: {exc}")
    games = max(1, min(int(body.games or 1), 20))
    try:
        shogi = _import_shogi()
        sfen0 = shogi.Board().sfen()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    with _sp_lock():
        sess = _selfplay.get(uid)
        if sess is not None and sess.get("running"):
            raise HTTPException(status_code=409, detail="self-play already running")
        _selfplay[uid] = {"running": True, "stop": False, "game_index": 0,
                          "games_total": games, "sfen": sfen0, "moves": [],
                          "status": "starting", "last_winner": "", "last_end": ""}
    import threading as _th
    _th.Thread(target=_run_selfplay, args=(uid, games), daemon=True).start()
    return _selfplay_snapshot(uid)


@router.get("/selfplay/state", response_model=SelfplayState)
async def selfplay_state(session: dict = Depends(_require_user)):
    """Poll self-play progress (board SFEN, moves, last result, stats)."""
    return _selfplay_snapshot(session["user_id"])


@router.post("/selfplay/stop", response_model=SelfplayState)
async def selfplay_stop(session: dict = Depends(_require_user)):
    """Request a graceful stop (current game finishes as void)."""
    uid = session["user_id"]
    with _sp_lock():
        sess = _selfplay.get(uid)
        if sess is None or not sess.get("running"):
            raise HTTPException(status_code=404, detail="no running self-play")
        sess["stop"] = True
        sess["status"] = "stopping"
    return _selfplay_snapshot(uid)
