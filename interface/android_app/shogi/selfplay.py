"""Aiko self-play: Aiko (Jev decisions) vs YaneuraOu (USI engine).

How Aiko plays:
  1. Opening book first (learned from her own won games, epsilon-greedy).
  2. Otherwise Jev `choice` over capped candidate moves (captures/checks/
     promotions first, deterministic order), with recent loss lines in the
     rubric so she stops repeating losing openings.
  3. Random legal fallback if Jev is unreachable (game continues; logged).

How she learns (every finished game):
  - Opening book records w/d/l per (position, move).
  - Experience system records the full game (moves, result, score).
  - FlyMB DAN teaching: win +1 / loss -1 on the opening pattern.
  - Human-loop lessons (blunder_delta) are NOT touched: those tune
    human-vs-Aiko friendliness, not engine strength.

Config (env):
  YANEURAOU_PATH        USI engine binary (Jetson ARM build).
                        NOTE: start the server with cwd at the engine dir (or
                        set EvalDir) so it finds eval/nn.bin next to it.
  SELFPLAY_MOVETIME_MS  engine think time per move (default 800).
  SELFPLAY_MAX_MOVES    adjudicate draw past this ply count (default 256).
  SELFPLAY_BOOK_MIN_VISITS  book needs this many visits to play (default 3).
  SELFPLAY_JEV_CANDIDATES   max moves offered to Jev per turn (default 10).
  SELFPLAY_EXPLORATION  epsilon for book exploration (default 0.05).
  JEV_API_KEY           via .env.age (see ./util/edit_dotenv.sh).
  JEV_MODEL             override (default jev-latest).

Needs python-shogi (already in pyproject deps) for rules/legality.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

GAME_SELFPLAY = "shogi_selfplay"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _require_shogi():
    try:
        import shogi
    except ImportError as exc:
        raise RuntimeError(
            "python-shogi is required for self-play (pyproject lists "
            "python-shogi>=1.1.1; sync the Jetson env)") from exc
    return shogi


# ── opening book (per-user JSON beside the shogi DB) ─────────────────────────

def book_path(uid: str) -> Path:
    from . import records as _records
    return _records.user_shogi_db_path(uid).parent / "selfplay_book.json"


def load_book(uid: str) -> dict:
    try:
        raw = json.loads(book_path(uid).read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def save_book(uid: str, book: dict) -> None:
    try:
        p = book_path(uid)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
    except Exception as exc:
        log.debug("selfplay book save skipped: %s", exc)


def position_key(board) -> str:
    """SFEN minus the move counter (transposition-friendly)."""
    try:
        parts = board.sfen().split(" ")
        return " ".join(parts[:-1]) if len(parts) > 1 else board.sfen()
    except Exception:
        return ""


# ── candidate moves ──────────────────────────────────────────────────────────

def describe_move(board, move) -> str:
    """One-line candidate description; never raises (falls back to USI)."""
    try:
        usi = move.usi()
    except Exception:
        usi = str(move)
    try:
        promo = usi.endswith("+")
        cap = board.piece_at(move.to_square) is not None
        board.push(move)
        try:
            check = board.is_check()
        finally:
            board.pop()
        bits = [usi]
        if cap:
            bits.append("captures")
        if check:
            bits.append("gives check")
        if promo:
            bits.append("promotes")
        return " ".join(bits)
    except Exception:
        return usi


def candidate_moves(board, cap: int) -> list:
    """Deterministic shortlist: checks/captures/promotions first, USI order."""
    try:
        legal = list(board.legal_moves)
    except Exception:
        return []
    scored = []
    for m in legal:
        try:
            usi = m.usi()
        except Exception:
            continue
        desc = describe_move(board, m)
        prio = (0 if "gives check" in desc else 1,
                0 if "captures" in desc else 1,
                0 if "promotes" in desc else 1, usi)
        scored.append((prio, m, desc))
    scored.sort(key=lambda t: t[0])
    return [(m, d) for _, m, d in scored[:max(1, cap)]]


# ── Aiko's move ──────────────────────────────────────────────────────────────

def aiko_choose_move(uid: str, board, *, book: dict, rng: random.Random,
                     loss_lines: list[str]) -> tuple[str, str]:
    """Return (usi, source): book | jev | random-fallback."""
    from agentic.toolkit import jev as _jev

    cap = _env_int("SELFPLAY_JEV_CANDIDATES", 10)
    cands = candidate_moves(board, cap)
    if not cands:
        raise RuntimeError("no legal moves (position should have ended)")
    if len(cands) == 1:
        return cands[0][0].usi(), "forced"

    key = position_key(board)
    entry = book.get(key) if key else None
    if entry:
        scored = []
        for usi, rec in entry.items():
            w, d, l = (list(rec) + [0, 0, 0])[:3]
            total = w + d + l
            if total >= _env_int("SELFPLAY_BOOK_MIN_VISITS", 3):
                scored.append((((w + 0.5 * d) / total), usi))
        if scored:
            scored.sort(reverse=True)
            if rng.random() >= _env_float("SELFPLAY_EXPLORATION", 0.05):
                return scored[0][1], "book"
            return rng.choice(scored)[1], "book-explore"

    criteria = {m.usi(): desc for m, desc in cands}
    avoid = " Avoid lines resembling these recent losses: " + " | ".join(loss_lines[:3]) \
        if loss_lines else ""
    state = {
        "sfen": board.sfen(),
        "turn": "sente" if _is_sente_to_move(board) else "gote",
        "candidates": [m.usi() for m, _ in cands],
    }
    try:
        pick, _probs, _conf = _jev.choice(
            state,
            "Choose Aiko's shogi move. Prefer captures, checks, and promotions"
            " that improve her position." + avoid,
            criteria,
        )
        if pick in criteria:
            return pick, "jev"
        log.warning("selfplay: Jev picked unknown move %r", pick)
    except Exception as exc:
        log.warning("selfplay: Jev unavailable, random fallback: %s", exc)
    return rng.choice(cands)[0].usi(), "random-fallback"


def _is_sente_to_move(board) -> bool:
    try:
        return bool(board.turn)
    except Exception:
        return True


# ── engine move ──────────────────────────────────────────────────────────────

def engine_choose_move(board, *, movetime_ms: int | None = None) -> Optional[str]:
    """Ask YaneuraOu. Returns USI, 'resign', or None on engine failure."""
    from . import yaneuraou as _yu
    try:
        cmd = f"position sfen {board.sfen()}"
        return _yu.best_move_usi(cmd, movetime_ms=movetime_ms)
    except Exception as exc:
        log.warning("selfplay: engine move failed: %s", exc)
        return None


# ── game loop ────────────────────────────────────────────────────────────────

def _game_end(board) -> Optional[str]:
    """'checkmate' | 'draw' | None. Winner derived by caller from side to move."""
    try:
        if board.is_checkmate():
            return "checkmate"
    except Exception:
        pass
    for name in ("is_draw", "is_stalemate", "is_insufficient_material"):
        try:
            if getattr(board, name)():
                return "draw"
        except Exception:
            continue
    return None


def play_game(uid: str, *, aiko_sente: bool | None = None,
              movetime_ms: int | None = None,
              max_moves: int | None = None,
              rng: random.Random | None = None) -> dict[str, Any]:
    """Play one full Aiko-vs-engine game. Never raises on game logic errors."""
    shogi = _require_shogi()
    rng = rng or random.Random()
    max_plies = _env_int("SELFPLAY_MAX_MOVES", 256) if max_moves is None else max(10, int(max_moves))
    book = load_book(uid)
    loss_lines = _recent_loss_openings(uid)
    if aiko_sente is None:
        try:
            from . import records as _records
            aiko_sente = (_records.total_matches(uid) % 2 == 0)
        except Exception:
            aiko_sente = True
    aiko_color = "sente" if aiko_sente else "gote"

    board = shogi.Board()
    moves: list[str] = []
    sources: list[str] = []
    result: dict[str, Any] = {"winner": "void", "end": "aborted", "moves_made": 0,
                              "aiko_color": aiko_color, "moves": moves}
    try:
        for _ in range(max_plies):
            end = _game_end(board)
            if end == "checkmate":
                # Side to move is mated; the other side wins.
                winner_is_aiko = (_is_sente_to_move(board) != aiko_sente)
                result.update(winner="aiko" if winner_is_aiko else "engine",
                              end="checkmate")
                break
            if end == "draw":
                result.update(winner="draw", end="draw")
                break
            aiko_turn = (_is_sente_to_move(board) == aiko_sente)
            if aiko_turn:
                usi, src = aiko_choose_move(uid, board, book=book, rng=rng,
                                            loss_lines=loss_lines)
                sources.append(src)
            else:
                mv = engine_choose_move(board, movetime_ms=movetime_ms)
                if mv is None:
                    result.update(end="engine-error")
                    break
                if mv.strip().lower() == "resign":
                    result.update(winner="aiko", end="engine-resign")
                    break
                usi, src = mv.strip(), "engine"
                sources.append(src)
            try:
                board.push_usi(usi)
            except Exception as exc:
                log.warning("selfplay: illegal move %r (%s), aborting game", usi, src)
                result.update(end=f"illegal-{src}")
                break
            moves.append(usi)
        else:
            result.update(winner="draw", end="move-cap")
        result["moves_made"] = len(moves)
        result["moves"] = moves
        if result["winner"] in ("aiko", "engine", "draw"):
            _learn_from_game(uid, moves=moves, aiko_color=aiko_color,
                             result=result, book=book)
    except Exception as exc:
        log.warning("selfplay game aborted: %s", exc)
        result.update(end=f"error: {type(exc).__name__}")
    return result


def _recent_loss_openings(uid: str, n: int = 3) -> list[str]:
    """First moves of Aiko's recent engine losses (for the avoid rubric)."""
    try:
        from . import records as _records
        from interface.android_app import learn as _learn
        rows = _learn.load_recent(uid, GAME_SELFPLAY, 15)
        out = []
        for r in rows:
            if r.get("winner") != "engine":
                continue
            extra = r.get("extra") or {}
            opening = (extra.get("opening") or "") if isinstance(extra, dict) else ""
            if opening:
                out.append(opening)
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


# ── learning ─────────────────────────────────────────────────────────────────

def _learn_from_game(uid: str, *, moves: list[str], aiko_color: str,
                     result: dict, book: dict) -> None:
    """Book + experience + fly teaching. Each step guarded; never raises."""
    winner = result.get("winner", "draw")
    # 1. Opening book: credit Aiko's moves at each position she faced.
    try:
        shogi = _require_shogi()
        b = shogi.Board()
        for i, usi in enumerate(moves):
            if (i % 2 == 0) != (aiko_color == "sente"):
                b.push_usi(usi)
                continue
            key = position_key(b)
            if key:
                rec = book.setdefault(key, {}).setdefault(usi, [0, 0, 0])
                if winner == "aiko":
                    rec[0] += 1
                elif winner == "draw":
                    rec[1] += 1
                else:
                    rec[2] += 1
            b.push_usi(usi)
        save_book(uid, book)
    except Exception as exc:
        log.debug("selfplay book update skipped: %s", exc)
    opening = " ".join(moves[:8])

    # 2. Match record in the namespaced self-play store (human stats untouched).
    try:
        from interface.android_app import learn as _learn
        _learn.append_match(uid, GAME_SELFPLAY, {
            "difficulty": "yaneuraou-selfplay",
            "span": 0,
            "winner": winner if winner in ("aiko", "draw") else "engine",
            "you_pts": 0,
            "aiko_pts": 1 if winner == "aiko" else 0,
            "moves_made": len(moves),
            "extra": {"end": result.get("end", ""), "engine": "yaneuraou",
                      "opening": opening},
        })
    except Exception as exc:
        log.debug("selfplay match record skipped: %s", exc)

    # 3. Full game experience (searchable later).
    try:
        from agentic.experience.acquire import record_experience
        score = 1.0 if winner == "aiko" else (0.5 if winner == "draw" else 0.0)
        record_experience(
            "aiko-shogi-selfplay",
            f"Self-play vs YaneuraOu as {aiko_color} ({result.get('end')})",
            [{"tool": "shogi-move", "ok": True, "args": {"move": m}} for m in moves],
            f"{winner} in {len(moves)} moves",
            winner in ("aiko", "draw"), score,
        )
    except Exception as exc:
        log.debug("selfplay experience record skipped: %s", exc)

    # 4. FlyMB DAN teaching: wins approach, losses avoid.
    try:
        from cognition.fly_registry import get_flymb
        from cognition.flymemory import text_features
        mb = get_flymb(uid)
        if mb is not None and winner in ("aiko", "engine"):
            reward = 1.0 if winner == "aiko" else -1.0
            feats = text_features(f"shogi {' '.join(moves[:10])} {winner}")
            mb.reinforce(mb.encode(feats), reward)
    except Exception as exc:
        log.debug("selfplay fly teaching skipped: %s", exc)


def play_match(uid: str, games: int = 1, **kw) -> list[dict]:
    """Play N games (colors alternate). Returns per-game result dicts."""
    out = []
    for _ in range(max(1, int(games))):
        out.append(play_game(uid, **kw))
        time.sleep(0.5)
    return out


if __name__ == "__main__":  # pragma: no cover - manual runs only
    import sys
    _uid = sys.argv[1] if len(sys.argv) > 1 else "oppa"
    _n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for _r in play_match(_uid, _n):
        print(f"{_r['winner']} ({_r['end']}) in {_r['moves_made']} moves as {_r['aiko_color']}")
