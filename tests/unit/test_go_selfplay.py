"""Go self-play (Jev vs KataGo-or-baseline). Engine + Jev mocked."""
from __future__ import annotations

import random


def test_tromp_taylor_empty_board():
    from interface.android_app.go.selfplay import tromp_taylor
    from interface.android_app.go.board import GoBoard
    b, w = tromp_taylor(GoBoard(9))
    assert (b, w) == (0.0, 6.5)  # empty: white wins on komi


def test_tromp_taylor_counts_territory():
    from interface.android_app.go.selfplay import tromp_taylor
    from interface.android_app.go.board import GoBoard
    b = GoBoard(9)
    b.play_gtp("D4")
    black, white = tromp_taylor(b)
    assert black >= 1.0 > 0 and white == 6.5


def test_candidates_capped_pass_last():
    from interface.android_app.go.selfplay import candidate_moves
    from interface.android_app.go.board import GoBoard
    cands = candidate_moves(GoBoard(9), 12)
    assert len(cands) <= 12
    assert cands[-1][0] == "pass"
    assert all(isinstance(m, str) for m, _ in cands)


def _patch_learning(monkeypatch, tmp_path):
    import interface.android_app.go.selfplay as sp
    import interface.android_app.learn as learnmod
    import agentic.experience.acquire as acq
    import cognition.fly_registry as reg
    rec = {"matches": [], "exp": []}
    monkeypatch.setattr(sp, "book_path", lambda uid: tmp_path / "go_book.json")
    monkeypatch.setattr(learnmod, "append_match",
                        lambda uid, game, rec_: rec["matches"].append((uid, game, rec_)))
    monkeypatch.setattr(acq, "record_experience",
                        lambda *a, **k: rec["exp"].append((a, k)) or "exp-id")
    monkeypatch.setattr(reg, "get_flymb", lambda uid=None: None)
    return rec


def test_full_game_learns(monkeypatch, tmp_path):
    import interface.android_app.go.selfplay as sp
    import interface.android_app.go.games_go as gg
    import agentic.toolkit.jev as jevmod
    monkeypatch.setattr(jevmod, "choice",
                        lambda state, ins, crit: (sorted(crit)[0], {}, 0.9))
    monkeypatch.setattr(gg, "_ai_move",
                        lambda board, difficulty=None, **kw: ("pass", "random"))
    rec = _patch_learning(monkeypatch, tmp_path)
    out = sp.play_game("u", size=9, rng=random.Random(0), max_moves=6)
    assert out["winner"] in ("aiko", "engine", "draw")
    assert out["moves_made"] > 0
    uid, game, m = rec["matches"][0]
    assert game == "go_selfplay" and m["winner"] == out["winner"]
    assert rec["exp"]


def test_engine_failure_voids_gracefully(monkeypatch, tmp_path):
    import interface.android_app.go.selfplay as sp
    import interface.android_app.go.games_go as gg
    import agentic.toolkit.jev as jevmod
    monkeypatch.setattr(jevmod, "choice",
                        lambda state, ins, crit: (sorted(crit)[0], {}, 0.9))

    def _boom(board, difficulty=None, **kw):
        raise RuntimeError("engine down")

    monkeypatch.setattr(gg, "_ai_move", _boom)
    _patch_learning(monkeypatch, tmp_path)
    out = sp.play_game("u", size=9, rng=random.Random(0), max_moves=6)
    assert out["winner"] in ("aiko", "engine", "draw", "void")


def test_book_position_key_stable():
    from interface.android_app.go.selfplay import position_key
    from interface.android_app.go.board import GoBoard
    a, b = GoBoard(9), GoBoard(9)
    a.play_gtp("D4")
    b.play_gtp("D4")
    assert position_key(a) == position_key(b)
    assert position_key(GoBoard(9)) != position_key(a)


def test_jev_down_voids_go_game(monkeypatch, tmp_path):
    import interface.android_app.go.selfplay as sp
    import interface.android_app.learn as learnmod
    import agentic.toolkit.jev as jevmod
    monkeypatch.setattr(jevmod, "choice", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(jevmod.time, "sleep", lambda s: None)
    monkeypatch.setattr(sp, "book_path", lambda uid: tmp_path / "g.json")
    appended = []
    monkeypatch.setattr(learnmod, "append_match",
                        lambda uid, game, rec: appended.append(rec))
    out = sp.play_game("u", size=9, rng=__import__("random").Random(0), max_moves=6)
    assert out["winner"] == "void" and out["end"] == "jev-down"
    assert appended == []
