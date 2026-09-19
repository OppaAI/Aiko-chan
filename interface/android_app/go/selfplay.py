"""Aiko self-play for Go: Aiko (Jev) vs KataGo-or-baseline.

Board rules come from the local GoBoard engine (no KataGo needed to play);
the opponent uses games_go._ai_move (KataGo when present, blunder/random
fallback otherwise — winnable by default at easy blunder rates).

Aiko moves by: opening book (ε-greedy on won lines) -> Jev `choice` over
heuristically capped candidates (captures first, deterministic order) ->
random legal fallback (logged, game continues).

Scoring is Tromp-Taylor area scoring implemented here (the local engine has
no scorer; human games leave winner unknown on double pass). All stones
treated alive — documented approximation, deterministic and symmetric.

Learning per finished game (game key "go_selfplay"):
  1. Opening book: position hash -> move w/d/l (9x9 openings recur).
  2. Match store (human "go" stats untouched).
  3. Experience rows: moves, result, score 1/0.5/0.
  4. FlyMB DAN teaching: win +1 / loss -1 on the opening signature.

Config (env): SELFPLAY_GO_SIZE (9), SELFPLAY_GO_KOMI (6.5),
  SELFPLAY_GO_MAX_MOVES (0 = size*size*3), SELFPLAY_BOOK_MIN_VISITS (3),
  SELFPLAY_JEV_CANDIDATES (12), SELFPLAY_EXPLORATION (0.05).
  Jev key: JEV_API_KEY via .env.age. Opponent strength: Go difficulty envs.
"""
from __future__ import annotations

import logging
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

GAME_SELFPLAY = "go_selfplay"


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


def book_path(uid: str) -> Path:
    from interface.android_app import learn as _learn
    return _learn.user_game_db_path(uid, GAME_SELFPLAY).parent / "go_book.json"


def load_book(uid: str) -> dict:
    try:
        import json
        raw = json.loads(book_path(uid).read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def save_book(uid: str, book: dict) -> None:
    try:
        import json
        p = book_path(uid)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
    except Exception as exc:
        log.debug("go selfplay book save skipped: %s", exc)


def position_key(board) -> str:
    """Compact transposition-friendly key: size + grid + turn."""
    try:
        rows = ["".join(str(c) for c in row) for row in board.grid]
        return f"{board.size}:" + "/".join(rows) + (":B" if board.turn == 1 else ":W")
    except Exception:
        return ""


def tromp_taylor(board) -> tuple[float, float]:
    """Area score (black, white) with komi. All stones treated alive.

    Documented approximation: no dead-stone removal (that needs KataGo
    adjudication). Deterministic and symmetric — fair for training.
    """
    from .board import BLACK, WHITE, EMPTY
    size = board.size
    black_stones = sum(1 for r in range(size) for c in range(size) if board.grid[r][c] == BLACK)
    white_stones = sum(1 for r in range(size) for c in range(size) if board.grid[r][c] == WHITE)
    seen = [[False] * size for _ in range(size)]
    black_terr = white_terr = 0
    for r in range(size):
        for c in range(size):
            if board.grid[r][c] != EMPTY or seen[r][c]:
                continue
            q = deque([(r, c)])
            seen[r][c] = True
            area, borders = 0, set()
            while q:
                cr, cc = q.popleft()
                area += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < size and 0 <= nc < size):
                        continue
                    v = board.grid[nr][nc]
                    if v == EMPTY and not seen[nr][nc]:
                        seen[nr][nc] = True
                        q.append((nr, nc))
                    elif v != EMPTY:
                        borders.add(v)
            if borders == {BLACK}:
                black_terr += area
            elif borders == {WHITE}:
                white_terr += area
    komi = _env_float("SELFPLAY_GO_KOMI", 6.5)
    return float(black_stones + black_terr), float(white_stones + white_terr + komi)


def _describe_candidate(board, gtp: str, captured: int) -> str:
    if gtp == "pass":
        return "pass (let opponent move)"
    s = f"play {gtp}"
    if captured > 0:
        s += f", captures {captured}"
    return s


def candidate_moves(board, cap: int) -> list[tuple[str, str]]:
    """Deterministic shortlist: captures first, pass last, GTP order."""
    try:
        legal = board.legal_moves_gtp()
    except Exception:
        return []
    if len(legal) <= 1:
        return [(m, _describe_candidate(board, m, 0)) for m in legal]
    scored = []
    for m in legal:
        if m == "pass":
            continue  # handle pass separately
        try:
            trial = board.copy()
            before = sum(trial.captured.values())
            trial.play_gtp(m)
            after = sum(trial.captured.values())
            scored.append(((0, -(after - before), m), m))
        except Exception:
            continue
    scored.sort(key=lambda t: t[0])
    out = [(m, _describe_candidate(board, m, -s[1])) for s, m in scored[:max(0, cap - 1)]]
    # Always include pass at the end
    out.append(("pass", _describe_candidate(board, "pass", 0)))
    return out[:max(1, cap)]


def aiko_choose_move(uid: str, board, *, book: dict, rng: random.Random,
                     loss_lines: list[str]) -> tuple[str, str]:
    """Return (gtp, source): book | jev | random-fallback."""
    from agentic.toolkit import jev as _jev
    cap = _env_int("SELFPLAY_JEV_CANDIDATES", 12)
    cands = candidate_moves(board, cap)
    if not cands:
        raise RuntimeError("no legal moves (position should have ended)")
    if len(cands) == 1:
        return cands[0][0], "forced"
    key = position_key(board)
    entry = book.get(key) if key else None
    if entry:
        scored = []
        for gtp, rec in entry.items():
            w, d, l = (list(rec) + [0, 0, 0])[:3]
            total = w + d + l
            if total >= _env_int("SELFPLAY_BOOK_MIN_VISITS", 3) and not (w == 0 and total >= 3):
                scored.append((((w + 0.5 * d) / total), gtp))
        if scored:
            rng.shuffle(scored)
            scored.sort(key=lambda t: t[0], reverse=True)
            if rng.random() >= _env_float("SELFPLAY_EXPLORATION", 0.05):
                return scored[0][1], "book"
            return rng.choice(scored)[1], "book-explore"
    criteria = {m: d for m, d in cands}
    avoid = " Avoid lines resembling these recent losses: " + " | ".join(loss_lines[:3]) \
        if loss_lines else ""
    state = {"size": board.size, "turn": "B" if board.turn == 1 else "W",
             "history": list(board.history[-12:]), "candidates": [m for m, _ in cands]}
    try:
        pick, _, _ = _jev.choice(
            state,
            "Choose Aiko's Go move. Prefer captures and solid shape; "
            "passing early is usually wrong." + avoid,
            criteria,
        )
        if pick in criteria:
            return pick, "jev"
        log.warning("go selfplay: Jev picked unknown move %r", pick)
    except Exception as exc:
        log.warning("go selfplay: Jev unavailable, random fallback: %s", exc)
    non_pass = [m for m, _ in cands if m != "pass"]
    return rng.choice(non_pass or [m for m, _ in cands]), "random-fallback"


def engine_choose_move(board, *, uid=None) -> tuple[Optional[str], str]:
    """Opponent via the shared AI mover (KataGo when present, else blunder/random)."""
    try:
        from . import games_go as _gg
        mv, eng = _gg._ai_move(board, difficulty="easy", uid=uid, use_engine=True)
        return mv, eng or "random"
    except Exception as exc:
        log.warning("go selfplay: opponent move failed: %s", exc)
        return None, "error"


def _recent_loss_openings(uid: str, n: int = 3) -> list[str]:
    try:
        from interface.android_app import learn as _learn
        out = []
        for r in _learn.load_recent(uid, GAME_SELFPLAY, 15):
            if r.get("winner") != "engine":
                continue
            extra = r.get("extra") or {}
            opening = extra.get("opening", "") if isinstance(extra, dict) else ""
            if opening:
                out.append(opening)
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def play_game(uid: str, *, size: int | None = None,
              rng: random.Random | None = None,
              max_moves: int | None = None,
              aiko_black: bool | None = None,
              on_move=None, is_stopped=None, on_start=None) -> dict[str, Any]:
    """Play one full game. Never raises on game logic errors."""
    from .board import GoBoard, BLACK
    rng = rng or random.Random()
    size = size or _env_int("SELFPLAY_GO_SIZE", 9)
    if size not in (9, 13, 19):
        size = 9
    max_moves = max_moves or _env_int("SELFPLAY_GO_MAX_MOVES", 0) or size * size * 3
    book = load_book(uid)
    loss_lines = _recent_loss_openings(uid)
    if aiko_black is None:
        # Random sides every game (a coin flip, not alternation).
        aiko_black = rng.random() < 0.5
    aiko_color = "B" if aiko_black else "W"
    if on_start is not None:
        try:
            on_start({"aiko_color": aiko_color})
        except Exception as exc:
            log.debug("go selfplay on_start skipped: %s", exc)

    board = GoBoard(size)
    moves: list[str] = []
    sources: list[str] = []
    result: dict[str, Any] = {"winner": "void", "end": "aborted", "moves_made": 0,
                              "aiko_color": aiko_color, "moves": moves}
    try:
        for _ in range(max(1, max_moves)):
            if is_stopped is not None:
                try:
                    if is_stopped():
                        result.update(winner="void", end="stopped")
                        break
                except Exception:
                    pass
            if board.status != "playing":
                break
            aiko_turn = (board.turn == BLACK) == aiko_black
            if aiko_turn:
                mv, src = aiko_choose_move(uid, board, book=book, rng=rng,
                                           loss_lines=loss_lines)
            else:
                mv, src = engine_choose_move(board, uid=uid)
                if mv is None:
                    result.update(end="engine-error")
                    break
            try:
                board.play_gtp(mv)
            except Exception as exc:
                log.warning("go selfplay: illegal move %r (%s), aborting", mv, src)
                result.update(end=f"illegal-{src}")
                break
            moves.append(mv)
            sources.append(src)
            if on_move is not None:
                try:
                    on_move(list(moves), board.stones_list())
                except Exception as exc:
                    log.debug("go selfplay on_move skipped: %s", exc)
            if board.passes >= 2:
                break
        else:
            pass
        if result["winner"] == "void" and result.get("end") not in ("stopped", "engine-error") \
                and not result["end"].startswith("illegal") and not result["end"].startswith("error"):
            bpts, wpts = tromp_taylor(board)
            if abs(bpts - wpts) < 1e-9:
                result.update(winner="draw", end="scored-draw")
            else:
                black_wins = bpts > wpts
                aiko_wins = black_wins == aiko_black
                result.update(winner="aiko" if aiko_wins else "engine",
                              end="scored", black_points=round(bpts, 1),
                              white_points=round(wpts, 1))
        result["moves_made"] = len(moves)
        result["moves"] = moves
        if result["winner"] in ("aiko", "engine", "draw"):
            _learn_from_game(uid, moves=moves, aiko_color=aiko_color,
                             result=result, book=book, sources=sources,
                             size=size)
    except Exception as exc:
        log.warning("go selfplay game aborted: %s", exc)
        result.update(end=f"error: {type(exc).__name__}")
    return result


def _learn_from_game(uid: str, *, moves: list[str], aiko_color: str,
                     result: dict, book: dict, sources: list[str], size: int) -> None:
    winner = result.get("winner", "draw")
    # 1. Opening book: early positions only (deep positions rarely recur).
    try:
        from .board import GoBoard
        b = GoBoard(size)
        seen = 0
        for i, mv in enumerate(moves[:12]):
            aiko_ply = ((b.turn == 1) == (aiko_color == "B"))
            key = position_key(b)
            try:
                b.play_gtp(mv)
            except Exception:
                break
            if not aiko_ply or not key:
                continue
            rec = book.setdefault(key, {}).setdefault(mv, [0, 0, 0])
            if winner == "aiko":
                rec[0] += 1
            elif winner == "draw":
                rec[1] += 1
            else:
                rec[2] += 1
            seen += 1
            if seen >= 4:
                break
        save_book(uid, book)
    except Exception as exc:
        log.debug("go book update skipped: %s", exc)
    opening = " ".join(moves[:8])
    # 2-4. Match store + experience + fly teaching (mirrors shogi/koikoi).
    try:
        from interface.android_app import learn as _learn
        _learn.append_match(uid, GAME_SELFPLAY, {
            "difficulty": f"go-{size}x{size}-selfplay",
            "span": 0,
            "winner": winner if winner in ("aiko", "draw") else "engine",
            "you_pts": 0,
            "aiko_pts": 1 if winner == "aiko" else 0,
            "moves_made": len(moves),
            "extra": {"end": result.get("end", ""), "engine": "katago-or-baseline",
                      "opening": opening, "size": size,
                      "black_points": result.get("black_points"),
                      "white_points": result.get("white_points")},
        })
    except Exception as exc:
        log.debug("go match record skipped: %s", exc)
    try:
        from agentic.experience.acquire import record_experience
        score = 1.0 if winner == "aiko" else (0.5 if winner == "draw" else 0.0)
        record_experience(
            "aiko-go-selfplay",
            f"Self-play {size}x{size} vs katago-or-baseline as {aiko_color} ({result.get('end')})",
            [{"tool": f"go-{sources[i] if i < len(sources) else 'unknown'}",
              "ok": True, "args": {"move": m}} for i, m in enumerate(moves)],
            f"{winner} in {len(moves)} moves",
            winner in ("aiko", "draw"), score,
        )
    except Exception as exc:
        log.debug("go experience record skipped: %s", exc)
    try:
        from cognition.fly_registry import get_flymb
        from cognition.flymemory import text_features
        mb = get_flymb(uid)
        if mb is not None and winner in ("aiko", "engine"):
            reward = 1.0 if winner == "aiko" else -1.0
            feats = text_features(f"go {size}x{size} {' '.join(moves[:10])} {winner}")
            mb.reinforce(mb.encode(feats), reward)
    except Exception as exc:
        log.debug("go fly teaching skipped: %s", exc)


def play_match(uid: str, games: int = 1, **kw) -> list[dict]:
    """Play N games (colors alternate). Returns per-game result dicts."""
    out = []
    for _ in range(max(1, int(games))):
        if kw.get("is_stopped") is not None:
            try:
                if kw["is_stopped"]():
                    break
            except Exception:
                pass
        out.append(play_game(uid, **kw))
        time.sleep(0.5)
    return out


if __name__ == "__main__":  # pragma: no cover - manual runs only
    import sys
    _uid = sys.argv[1] if len(sys.argv) > 1 else "oppa"
    _n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    for _r in play_match(_uid, _n):
        print(f"{_r['winner']} ({_r.get('end')}) in {_r['moves_made']} as {_r['aiko_color']}")
