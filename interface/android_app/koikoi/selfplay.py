"""Aiko self-play for Koi-Koi: Aiko (Jev) vs heuristic sparring partner.

No external engine exists for hanafuda — and none is needed. The built-in
heuristic AI (ai.choose_play/flip/decision) plays the opponent seat with a
HIGH blunder rate (beatable curriculum), while Aiko decides via Jev over
the naturally tiny option sets (a handful of plays/flips, one koi decision).

Seat mapping (documented, load-bearing): the engine occupies the "you" seat
in the shared round machinery; the learning layer translates "you" wins to
"engine" wins. Human-vs-Aiko records are namespaced separately and untouched.

Learning per finished match (game key "koikoi_selfplay"):
  1. Opening book: Aiko's first-play per round keyed by (month, hand, field).
     Sparse by nature (luck-heavy game); avoidance + experience do the lifting.
  2. Match store: winner/points/months (human "koikoi" stats untouched).
  3. Experience rows: full rounds, score 1/0.5/0.
  4. FlyMB DAN teaching: win +1 / loss -1 on the match signature.

Config (env): SELFPLAY_KOI_MONTHS (3), SELFPLAY_KOI_BLUNDER (easy),
  SELFPLAY_JEV_CANDIDATES (12). Jev key: JEV_API_KEY via .env.age.
"""
from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

GAME_SELFPLAY = "koikoi_selfplay"
ENGINE_SEAT = "you"  # shared machinery seat; means "engine" in learning layer
AIKO_SEAT = "aiko"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def book_path(uid: str) -> Path:
    from interface.android_app import learn as _learn
    return _learn.user_game_db_path(uid, GAME_SELFPLAY).parent / "koikoi_book.json"


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
        log.debug("koikoi selfplay book save skipped: %s", exc)


def _sig(hand: list[int], field: list[int], month: int) -> str:
    return f"m{month}:" + ",".join(map(str, sorted(hand))) + "|" + ",".join(map(str, sorted(field)))


def _describe_play(hand: int, take: list[int]) -> str:
    from . import cards as C
    try:
        name = C.card_name(hand) if hasattr(C, "card_name") else f"card {hand}"
    except Exception:
        name = f"card {hand}"
    if not take:
        return f"discard {name} (no take)"
    try:
        vals = [C.card_value(c) for c in [hand] + list(take)]
        return f"play {name} taking {len(take)} (value ~{sum(vals):.1f})"
    except Exception:
        return f"play {name} taking {len(take)}"


def aiko_choose_play(uid: str, hand: int, options: list[list[int]], captured: list[int],
                     loss_lines: list[str], cap: int) -> tuple[int, list[int], str]:
    """Return (hand, take, source): jev | random-fallback. Options are takes for one card."""
    from agentic.toolkit import jev as _jev
    if len(options) <= 1:
        return hand, list(options[0]) if options else [], "forced"
    # Prefer takes (captures), deterministic order, capped.
    ordered = sorted((list(o) for o in options), key=lambda o: (-len(o), o))
    capped = ordered[:max(1, cap)]
    criteria = {f"take{i}": _describe_play(hand, o) for i, o in enumerate(capped)}
    avoid = " Avoid lines resembling these recent losses: " + " | ".join(loss_lines[:3]) \
        if loss_lines else ""
    state = {"hand": hand, "captured": len(captured), "options": len(options)}
    try:
        pick, _, _ = _jev.choice(
            state,
            "Choose Aiko's hanafuda play. Prefer takes that capture high-value "
            "cards and build toward yaku." + avoid,
            criteria,
        )
        if pick in criteria:
            return hand, capped[int(pick[4:])], "jev"
        log.warning("koikoi selfplay: Jev picked unknown play %r", pick)
    except Exception as exc:
        log.warning("koikoi selfplay: Jev unavailable, random fallback: %s", exc)
    o = random.choice(capped)
    return hand, o, "random-fallback"


def aiko_choose_card(uid: str, options: list[tuple[int, list[int]]], captured: list[int],
                     loss_lines: list[str], cap: int, rng: random.Random) -> tuple[int, list[int], str]:
    """Choose which hand card to play (then take options follow that card)."""
    if len(options) <= 1:
        h, takes = options[0]
        take, src = (takes[0], "forced") if len(takes) == 1 else aiko_choose_play(
            uid, h, takes, captured, loss_lines, cap)
        return h, take, src
    ordered = sorted(options, key=lambda o: (-max((len(t) for t in o[1]), default=0), o[0]))
    capped_cards = [o[0] for o in ordered[:max(1, cap)]]
    from agentic.toolkit import jev as _jev
    criteria = {f"card{i}": f"play card {h} ({len([t for h2, t in options if h2 == h][0])} takes available)"
                for i, h in enumerate(capped_cards)}
    try:
        pick, _, _ = _jev.choice(
            {"cards": capped_cards, "captured": len(captured)},
            "Choose which hanafuda card Aiko plays. Prefer cards with rich takes." +
            (" Avoid lines resembling these recent losses: " + " | ".join(loss_lines[:3]) if loss_lines else ""),
            criteria,
        )
        if pick in criteria:
            h = capped_cards[int(pick[4:])]
            takes = next(t for hh, t in options if hh == h)
            take, src = (takes[0], "forced") if len(takes) == 1 else aiko_choose_play(
                uid, h, takes, captured, loss_lines, cap)
            return h, take, src if src != "forced" else "jev"
    except Exception as exc:
        log.warning("koikoi selfplay: Jev card pick unavailable, random fallback: %s", exc)
    h = rng.choice([o[0] for o in options])
    takes = next(t for hh, t in options if hh == h)
    return h, rng.choice(takes), "random-fallback"


def aiko_choose_decision(captured: list[int], opp_captured: list[int], koi: int,
                         cards_left: int) -> tuple[str, str]:
    """koi (continue) or stop (bank). Small option set: direct Jev choice."""
    from agentic.toolkit import jev as _jev
    from . import cards as C
    try:
        mine = C.yaku_points(C.detect_yaku(captured))
    except Exception:
        mine = 0
    try:
        pick, _, _ = _jev.choice(
            {"my_points_now": mine, "multiplier": koi + 1, "cards_left": cards_left,
             "opp_cards": len(opp_captured)},
            "Koi-koi (continue for multiplied stakes) or stop (bank points now)? "
            "Stop with solid points; continue only with a commanding lead.",
            {"koi": "continue for higher stakes", "stop": "bank the points now"},
        )
        if pick in ("koi", "stop"):
            return pick, "jev"
    except Exception as exc:
        log.warning("koikoi selfplay: Jev decision unavailable: %s", exc)
    return ("stop" if mine >= 5 else "koi"), "random-fallback"


def _recent_loss_signatures(uid: str, n: int = 3) -> list[str]:
    try:
        from interface.android_app import learn as _learn
        out = []
        for r in _learn.load_recent(uid, GAME_SELFPLAY, 15):
            if r.get("winner") != "engine":
                continue
            extra = r.get("extra") or {}
            sig = extra.get("signature", "") if isinstance(extra, dict) else ""
            if sig:
                out.append(sig)
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def play_match(uid: str, games: int = 1, *, months: int | None = None,
               rng: random.Random | None = None,
               on_round=None, is_stopped=None) -> dict[str, Any]:
    """Play a full self-play match. Never raises on game logic errors."""
    from . import cards as C
    from . import ai as _ai
    from . import games_koikoi as _g
    rng = rng or random.Random()
    months = months if months else _env_int("SELFPLAY_KOI_MONTHS", 3)
    months = max(1, min(months, 12))
    book = load_book(uid)
    loss_lines = _recent_loss_signatures(uid)
    cap = _env_int("SELFPLAY_JEV_CANDIDATES", 12)
    blunder = (os.getenv("SELFPLAY_KOI_BLUNDER") or "easy").strip() or "easy"

    game = {
        "uid": uid, "mode": "selfplay", "difficulty": blunder, "months": months,
        "month": 1, "oya": None, "turn": "aiko", "status": "playing",
        "totals": {"you": 0, "aiko": 0}, "winner": None,
        "engine": "heuristic", "round_result": None,
        "rng": rng, "koi_calls": 0, "moves_made": 0,
    }
    # Alternate dealer: even completed self-play matches -> Aiko deals.
    try:
        from interface.android_app import learn as _learn
        played = sum(1 for _ in _learn.load_recent(uid, GAME_SELFPLAY, 1000))
        game["oya"] = "aiko" if played % 2 == 0 else ENGINE_SEAT
    except Exception:
        game["oya"] = "aiko"

    result: dict[str, Any] = {"winner": "void", "end": "aborted", "months": months,
                              "aiko_pts": 0, "engine_pts": 0, "rounds": []}
    sources: list[str] = []
    try:
        _g._new_round(game)
        _g._auto_dealt(game)
        guard_rounds = 0
        while game.get("status") == "playing" and guard_rounds < months + 4:
            guard_rounds += 1
            game["round_result"] = None  # fresh round (settle fns never clear it)
            if is_stopped is not None:
                try:
                    if is_stopped():
                        result.update(winner="void", end="stopped")
                        break
                except Exception:
                    pass
            # Play one full round, both seats.
            guard_turns = 0
            while game.get("status") == "playing" and game.get("round_result") is None \
                    and guard_turns < 60:
                guard_turns += 1
                side = game["turn"]
                if side == AIKO_SEAT:
                    _aiko_turn(game, book, loss_lines, cap, rng, sources)
                else:
                    _eng_turn(game, blunder, rng, sources)
                if game.get("status") != "playing":
                    break
                if _g._hands_empty(game) and game.get("round_result") is None:
                    _g._settle_exhausted(game)
                    break
                game["turn"] = ENGINE_SEAT if side == AIKO_SEAT else AIKO_SEAT
            rr = game.get("round_result") or {}
            result["rounds"].append({"winner": rr.get("winner"), "points": rr.get("points", 0)})
            if on_round is not None:
                try:
                    on_round(dict(result))
                except Exception as exc:
                    log.debug("koikoi selfplay on_round skipped: %s", exc)
            if game.get("status") != "playing":
                break
        if result["winner"] == "void" and game.get("status") == "finished":
            w = game.get("winner") or "draw"
            result.update(winner=("engine" if w == ENGINE_SEAT else w),
                          end="finished",
                          aiko_pts=int(game["totals"]["aiko"]),
                          engine_pts=int(game["totals"]["you"]))
        if result["winner"] in ("aiko", "engine", "draw"):
            _learn_from_match(uid, game, result, book)
    except Exception as exc:
        log.warning("koikoi selfplay aborted: %s", exc)
        result.update(end=f"error: {type(exc).__name__}")
    return result


def play_matches(uid: str, games: int = 1, **kw) -> list[dict]:
    """Play N matches (dealer alternates via recorded history)."""
    out = []
    for _ in range(max(1, int(games))):
        if kw.get("is_stopped") is not None:
            try:
                if kw["is_stopped"]():
                    break
            except Exception:
                pass
        out.append(play_match(uid, **kw))
        time.sleep(0.5)
    return out


def _aiko_turn(game, book, loss_lines, cap, rng, sources):
    """One Aiko turn. Opening signature captured pre-play for book lookup."""
    from . import ai as _ai
    from . import games_koikoi as _g
    if not game["hand"]["aiko"]:
        return None
    pre_sig = _sig(game["hand"]["aiko"], game["field"], game["month"])
    # Book on the match opening signature (sparse by nature in hanafuda).
    if not game.get("opening_done"):
        entry = book.get(pre_sig)
        if entry:
            scored = [((w + 0.5 * d) / max(1, w + d + l), play)
                      for play, (w, d, l) in entry.items()
                      if (w + d + l) >= 2 and not (w == 0 and (w + d + l) >= 3)]
            if scored:
                rng.shuffle(scored)
                scored.sort(key=lambda t: t[0], reverse=True)
                best = scored[0][1]
                h, take = int(best.split(":")[0]), [int(x) for x in best.split(":")[1].split(",") if x]
                if h in game["hand"]["aiko"]:
                    take = [c for c in take if c in game["field"]]
                    _g._apply_hand_play(game, "aiko", h, take)
                    sources.append("book")
                    game["opening_done"] = True
                    game["opening_sig"], game["opening_play"] = pre_sig, best
                    _after_aiko_play(game, rng, sources)
                    return (h, take)
    options = _ai.enumerate_plays(game["hand"]["aiko"], game["field"])
    h, take, src = aiko_choose_card("x", options, game["cap"]["aiko"],
                                    loss_lines, cap, rng)
    sources.append(src)
    _g._apply_hand_play(game, "aiko", h, take)
    if not game.get("opening_done"):
        game["opening_done"] = True
        game["opening_sig"] = pre_sig
        game["opening_play"] = f"{h}:" + ",".join(map(str, take))
    _after_aiko_play(game, rng, sources)
    return (h, take)


def _after_aiko_play(game, rng, sources) -> None:
    from . import cards as C
    from . import games_koikoi as _g
    if game["stock"]:
        flip = game["stock"].pop(0)
        opts = C.match_options(flip, game["field"])
        if not opts:
            _g._apply_flip(game, "aiko", flip, [])
        else:
            _, take, src = aiko_choose_play("x", flip, opts, game["cap"]["aiko"], [], 12)
            sources.append(src)
            _g._apply_flip(game, "aiko", flip, take)
    new = _g._new_yaku(game, "aiko")
    if new and game["status"] == "playing":
        cards_left = len(game["hand"]["you"]) + len(game["hand"]["aiko"]) + len(game["stock"])
        call, src = aiko_choose_decision(game["cap"]["aiko"], game["cap"]["you"],
                                         game["koi"], cards_left)
        sources.append(src)
        if call == "koi":
            game["koi"] += 1
            game["koi_calls"] = game.get("koi_calls", 0) + 1
            _g._ack(game, "aiko")
        else:
            _g._settle_stop(game, "aiko")


def _eng_turn(game, blunder, rng, sources) -> None:
    from . import cards as C
    from . import ai as _ai
    from . import games_koikoi as _g
    if not game["hand"][ENGINE_SEAT]:
        return
    options = _ai.enumerate_plays(game["hand"][ENGINE_SEAT], game["field"])
    hand, take = _ai.choose_play(game["hand"][ENGINE_SEAT], game["field"],
                                 game["cap"][ENGINE_SEAT], blunder, rng,
                                 game.get("month"))
    sources.append("engine")
    _g._apply_hand_play(game, ENGINE_SEAT, hand, take)
    game["moves_made"] = game.get("moves_made", 0) + 1
    if game["stock"]:
        flip = game["stock"].pop(0)
        opts = C.match_options(flip, game["field"])
        chosen = _ai.choose_flip(flip, opts, game["cap"][ENGINE_SEAT], blunder,
                                 rng, game.get("month")) if opts else []
        _g._apply_flip(game, ENGINE_SEAT, flip, chosen)
    new = _g._new_yaku(game, ENGINE_SEAT)
    if new and game["status"] == "playing":
        cards_left = len(game["hand"]["you"]) + len(game["hand"]["aiko"]) + len(game["stock"])
        call = _ai.choose_decision(game["cap"][ENGINE_SEAT], game["cap"]["aiko"],
                                   game["koi"], cards_left, blunder, rng,
                                   game.get("month"))
        if call == "koi":
            game["koi"] += 1
            _g._ack(game, ENGINE_SEAT)
        else:
            _g._settle_stop(game, ENGINE_SEAT)


def _learn_from_match(uid: str, game: dict, result: dict, book: dict) -> None:
    winner = result.get("winner", "draw")
    # 1. Book first-plays (sparse; avoidance + experience do the heavy lifting).
    try:
        sig = game.get("opening_sig", "")
        play = game.get("opening_play", "")
        if sig and play:
            rec = book.setdefault(sig, {}).setdefault(play, [0, 0, 0])
            if winner == "aiko":
                rec[0] += 1
            elif winner == "draw":
                rec[1] += 1
            else:
                rec[2] += 1
            save_book(uid, book)
    except Exception as exc:
        log.debug("koikoi book update skipped: %s", exc)
    # 2. Match record (human "koikoi" stats untouched).
    try:
        from interface.android_app import learn as _learn
        _learn.append_match(uid, GAME_SELFPLAY, {
            "difficulty": "heuristic",
            "span": 0,
            "winner": winner if winner in ("aiko", "draw") else "engine",
            "you_pts": 0,
            "aiko_pts": int(result.get("aiko_pts", 0)),
            "moves_made": int(result.get("rounds", []) and len(result["rounds"]) or 0),
            "extra": {"engine": "heuristic",
                      "aiko_pts": int(result.get("aiko_pts", 0)),
                      "engine_pts": int(result.get("engine_pts", 0))},
        })
    except Exception as exc:
        log.debug("koikoi match record skipped: %s", exc)
    # 3. Experience + 4. fly teaching.
    try:
        from agentic.experience.acquire import record_experience
        score = 1.0 if winner == "aiko" else (0.5 if winner == "draw" else 0.0)
        record_experience(
            "aiko-koikoi-selfplay",
            f"Self-play vs heuristic ({result.get('aiko_pts', 0)}-{result.get('engine_pts', 0)})",
            [{"tool": "koikoi-round", "ok": True,
              "args": {"winner": r.get("winner"), "points": r.get("points", 0)}}
             for r in result.get("rounds", [])],
            f"{winner} {result.get('aiko_pts', 0)}-{result.get('engine_pts', 0)}",
            winner in ("aiko", "draw"), score,
        )
    except Exception as exc:
        log.debug("koikoi experience record skipped: %s", exc)
    try:
        from cognition.fly_registry import get_flymb
        from cognition.flymemory import text_features
        mb = get_flymb(uid)
        if mb is not None and winner in ("aiko", "engine"):
            reward = 1.0 if winner == "aiko" else -1.0
            feats = text_features(f"koikoi {result.get('aiko_pts', 0)}-{result.get('engine_pts', 0)} {winner}")
            mb.reinforce(mb.encode(feats), reward)
    except Exception as exc:
        log.debug("koikoi fly teaching skipped: %s", exc)


if __name__ == "__main__":  # pragma: no cover - manual runs only
    import sys
    _uid = sys.argv[1] if len(sys.argv) > 1 else "oppa"
    _r = play_match(_uid)
    print(f"{_r['winner']} ({_r.get('end', 'finished')}) "
          f"{_r.get('aiko_pts', 0)}-{_r.get('engine_pts', 0)}")
