"""Koi-Koi self-play (Jev vs heuristic). Jev + learning mocked."""
from __future__ import annotations

import random


def _patch(monkeypatch, tmp_path):
    import interface.android_app.koikoi.selfplay as sp
    import interface.android_app.learn as learnmod
    import agentic.experience.acquire as acq
    import cognition.fly_registry as reg
    rec = {"matches": [], "exp": []}
    monkeypatch.setattr(sp, "book_path", lambda uid: tmp_path / "kbook.json")
    monkeypatch.setattr(learnmod, "append_match",
                        lambda uid, game, rec_: rec["matches"].append((uid, game, rec_)))
    monkeypatch.setattr(acq, "record_experience",
                        lambda *a, **k: rec["exp"].append((a, k)) or "exp-id")
    monkeypatch.setattr(reg, "get_flymb", lambda uid=None: None)
    import agentic.toolkit.jev as jevmod

    def _choice(state, ins, crit):
        return (sorted(crit)[0], {}, 0.9)

    monkeypatch.setattr(jevmod, "choice", _choice)
    return sp, rec


def test_full_match_learns(monkeypatch, tmp_path):
    sp, rec = _patch(monkeypatch, tmp_path)
    out = sp.play_match("u", months=1, rng=random.Random(0))
    assert out["winner"] in ("aiko", "engine", "draw")
    assert len(out["rounds"]) >= 1
    uid, game, m = rec["matches"][0]
    assert game == "koikoi_selfplay" and m["winner"] == out["winner"]
    assert rec["exp"]
    import json
    book = json.loads((tmp_path / "kbook.json").read_text())
    assert isinstance(book, dict)


def test_decision_and_stop(monkeypatch, tmp_path):
    sp, rec = _patch(monkeypatch, tmp_path)
    import agentic.toolkit.jev as jevmod
    monkeypatch.setattr(jevmod, "choice", lambda state, ins, crit: ("stop", {}, 0.9))
    call, src = sp.aiko_choose_decision([1, 2, 3], [4], 1, 10)
    assert call == "stop" and src == "jev"
    out = sp.play_match("u", months=1, rng=random.Random(1),
                        is_stopped=lambda: True)
    assert out["winner"] == "void" and out["end"] == "stopped"
    assert rec["matches"] == []  # stopped games teach nothing


def test_sig_stable():
    from interface.android_app.koikoi.selfplay import _sig
    assert _sig([3, 1, 2], [9, 7], 2) == _sig([1, 2, 3], [7, 9], 2)
