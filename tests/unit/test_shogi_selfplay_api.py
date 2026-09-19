"""Self-play HTTP session (start/state/stop). Engine + games mocked."""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException


class _FakeBoard:
    def sfen(self):
        return "sfen0"


class _FakeShogi:
    @staticmethod
    def Board():
        return _FakeBoard()


@pytest.fixture
def gs(monkeypatch):
    import interface.android_app.shogi.games_shogi as gs
    import interface.android_app.shogi.yaneuraou as yu
    monkeypatch.setattr(yu, "available", lambda: True)
    monkeypatch.setattr(gs, "_import_shogi", lambda: _FakeShogi())
    monkeypatch.setattr(gs, "_selfplay_stats",
                        lambda uid: {"aiko_wins": 0, "engine_wins": 0, "draws": 0, "matches": 0})
    gs._selfplay.clear()
    return gs


def test_start_state_stop(gs, monkeypatch):
    import threading
    import time
    import interface.android_app.shogi.selfplay as sp
    gate = threading.Event()

    def slow_game(uid, **kw):
        gate.wait(10)
        return {"winner": "draw", "end": "move-cap", "moves_made": 0,
                "aiko_color": "sente", "moves": []}

    monkeypatch.setattr(sp, "play_game", slow_game)
    body = gs.SelfplayStartRequest(games=2)
    st = asyncio.run(gs.selfplay_start(body, {"user_id": "u"}))
    assert st.running and st.games_total == 2 and st.sfen == "sfen0"
    with pytest.raises(HTTPException) as ei:
        asyncio.run(gs.selfplay_start(body, {"user_id": "u"}))
    assert ei.value.status_code == 409
    st = asyncio.run(gs.selfplay_state({"user_id": "u"}))
    assert st.running
    st = asyncio.run(gs.selfplay_stop({"user_id": "u"}))
    assert st.status == "stopping"
    with pytest.raises(HTTPException) as ei2:
        asyncio.run(gs.selfplay_stop({"user_id": "nobody"}))
    assert ei2.value.status_code == 404
    gate.set()
    deadline = time.monotonic() + 10
    while gs._selfplay["u"]["running"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert gs._selfplay["u"]["running"] is False


def test_engine_offline_rejected(gs, monkeypatch):
    import interface.android_app.shogi.yaneuraou as yu
    monkeypatch.setattr(yu, "available", lambda: False)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(gs.selfplay_start(gs.SelfplayStartRequest(games=1), {"user_id": "u"}))
    assert ei.value.status_code == 503


def test_run_selfplay_records_progress(gs, monkeypatch):
    import interface.android_app.shogi.selfplay as sp
    seen = {}

    def fake_game(uid, **kw):
        kw.get("on_ply")("sfen3", ["7g7f", "3c3d", "8h2b"])
        return {"winner": "aiko", "end": "checkmate", "moves_made": 3,
                "aiko_color": "sente", "moves": ["7g7f", "3c3d", "8h2b"]}

    monkeypatch.setattr(sp, "play_game", fake_game)
    gs._selfplay["u"] = {"running": True, "stop": False, "game_index": 0,
                         "games_total": 1, "sfen": "", "moves": [],
                         "status": "starting", "last_winner": "", "last_end": ""}
    gs._run_selfplay("u", 1)
    sess = gs._selfplay["u"]
    assert sess["running"] is False
    assert sess["last_winner"] == "aiko" and sess["sfen"] == "sfen3"
    assert sess["moves"] == ["7g7f", "3c3d", "8h2b"]
    st = asyncio.run(gs.selfplay_state({"user_id": "u"}))
    assert st.last_winner == "aiko" and not st.running
