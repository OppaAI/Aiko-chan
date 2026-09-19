"""Go + Koi-Koi self-play session endpoints. Games mocked."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi import HTTPException


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_go_session_lifecycle(monkeypatch):
    import interface.android_app.go.games_go as gg
    import interface.android_app.go.selfplay as sp
    gate = threading.Event()

    def slow(uid, **kw):
        gate.wait(10)
        return {"winner": "draw", "end": "scored-draw", "moves_made": 0,
                "aiko_color": "B", "moves": []}

    monkeypatch.setattr(sp, "play_game", slow)
    monkeypatch.setattr(gg, "_selfplay_stats",
                        lambda uid: {"aiko_wins": 0, "engine_wins": 0, "draws": 0, "matches": 0})
    gg._selfplay.clear()
    st = _run(gg.selfplay_start(gg.SelfplayStartRequest(games=1), {"user_id": "u"}))
    assert st.running
    with pytest.raises(HTTPException) as ei:
        _run(gg.selfplay_start(gg.SelfplayStartRequest(games=1), {"user_id": "u"}))
    assert ei.value.status_code == 409
    gate.set()
    deadline = time.monotonic() + 10
    while gg._selfplay["u"]["running"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert gg._selfplay["u"]["running"] is False


def test_go_engine_offline_rejected(monkeypatch):
    import interface.android_app.go.games_go as gg
    # No KataGo binary here and no mocks: start must still work (baseline fallback).
    gg._selfplay.clear()
    st = _run(gg.selfplay_start(gg.SelfplayStartRequest(games=1), {"user_id": "u2"}))
    assert st.running
    gg._selfplay["u2"]["stop"] = True


def test_koi_session_lifecycle(monkeypatch):
    import interface.android_app.koikoi.games_koikoi as kk
    import interface.android_app.koikoi.selfplay as sp
    gate = threading.Event()

    def slow(uid, **kw):
        gate.wait(10)
        return {"winner": "draw", "end": "finished", "aiko_pts": 0,
                "engine_pts": 0, "rounds": []}

    monkeypatch.setattr(sp, "play_match", slow)
    monkeypatch.setattr(kk, "_selfplay_stats",
                        lambda uid: {"aiko_wins": 0, "engine_wins": 0, "draws": 0, "matches": 0})
    kk._selfplay.clear()
    st = _run(kk.selfplay_start(kk.SelfplayStartRequest(games=1), {"user_id": "u"}))
    assert st.running
    st = _run(kk.selfplay_stop({"user_id": "u"}))
    assert st.status == "stopping"
    gate.set()
    deadline = time.monotonic() + 10
    while kk._selfplay["u"]["running"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert kk._selfplay["u"]["running"] is False
    with pytest.raises(HTTPException) as ei:
        _run(kk.selfplay_stop({"user_id": "nobody"}))
    assert ei.value.status_code == 404
