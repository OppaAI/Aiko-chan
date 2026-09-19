"""Self-play (Aiko/Jev vs YaneuraOu) + Jev client. Engine + network mocked."""
from __future__ import annotations

import io
import json
import random
import urllib.error

import pytest


# ── fakes ────────────────────────────────────────────────────────────────────

class _Move:
    def __init__(self, usi, to_sq=50):
        self._usi = usi
        self.to_square = to_sq

    def usi(self):
        return self._usi


class _Board:
    """Minimal python-shogi surface. Mate when pushed >= script_end plies."""

    def __init__(self, script_end=None):
        self.pushed = []
        self.turn = True
        self._script_end = script_end

    def sfen(self):
        return "sfen%d" % len(self.pushed)

    @property
    def legal_moves(self):
        return [_Move("7g7f"), _Move("2g2f")]

    def piece_at(self, sq):
        return None

    def is_check(self):
        return False

    def push(self, m):
        self.pushed.append(m.usi())
        self.turn = not self.turn

    def pop(self):
        self.pushed.pop()
        self.turn = not self.turn

    def push_usi(self, usi):
        self.pushed.append(usi)
        self.turn = not self.turn

    def is_checkmate(self):
        return self._script_end is not None and len(self.pushed) >= self._script_end

    def is_draw(self):
        return False


class _ShogiMod:
    Board = _Board


@pytest.fixture
def sp(monkeypatch):
    import interface.android_app.shogi.selfplay as sp
    monkeypatch.setattr(sp, "_require_shogi", lambda: _ShogiMod())
    monkeypatch.setattr(sp, "book_path", lambda uid: __import__("pathlib").Path(str(_tmp[0])) / "b.json")
    return sp


_tmp = []


@pytest.fixture(autouse=True)
def _tmpdir(tmp_path, monkeypatch):
    _tmp.clear()
    _tmp.append(tmp_path)
    yield


# ── Jev client ────────────────────────────────────────────────────────────────

class _HTTPResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, body=b"nope"):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body))


def test_jev_choice_parses(monkeypatch):
    from agentic.toolkit import jev
    monkeypatch.setenv("JEV_API_KEY", "k")
    payload = {"model": "jev-latest",
               "answers": {"q": {"type": "choice", "choice": "7g7f",
                                 "probabilities": {"7g7f": 0.8, "2g2f": 0.2},
                                 "confidence": 0.75}},
               "usage": {}}
    monkeypatch.setattr(jev.urllib.request, "urlopen", lambda *a, **k: _HTTPResp(payload))
    opt, probs, conf = jev.choice("sfen", "pick", {"7g7f": "push", "2g2f": "push2"})
    assert opt == "7g7f" and probs["7g7f"] == 0.8 and conf == 0.75


def test_jev_auth_and_retry(monkeypatch):
    from agentic.toolkit import jev
    monkeypatch.setenv("JEV_API_KEY", "k")
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429)
        return _HTTPResp({"answers": {"q": {"type": "noul", "noul": 0.9}}, "usage": {}})

    monkeypatch.setattr(jev.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(jev.time, "sleep", lambda s: None)
    assert jev.noul("s", "urgent?") == 0.9
    assert len(calls) == 3


def test_jev_401_and_missing_key(monkeypatch):
    from agentic.toolkit import jev
    monkeypatch.setenv("JEV_API_KEY", "k")
    monkeypatch.setattr(jev.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(_http_error(401)))
    with pytest.raises(jev.JevAuthError):
        jev.noul("s", "q?")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(jev.JevAuthError):
        jev.noul("s", "q?")


def test_jev_score_validates_criteria(monkeypatch):
    from agentic.toolkit import jev
    monkeypatch.setenv("JEV_API_KEY", "k")
    with pytest.raises(ValueError):
        jev.score("s", "rate?", ["only-one"])


# ── candidates / book ──────────────────────────────────────────────────────────

def test_candidates_capped_and_deterministic(sp):
    b = _Board()
    cands = sp.candidate_moves(b, 1)
    assert len(cands) == 1
    assert [u for _, u in [(m, m.usi()) for m, _ in sp.candidate_moves(b, 10)]] == ["2g2f", "7g7f"]


def test_book_hit_skips_jev(sp, monkeypatch):
    import interface.android_app.shogi.selfplay as real
    real.save_book("u", {"sfen0": {"7g7f": [5, 0, 0], "2g2f": [0, 0, 5]}})
    called = []
    import agentic.toolkit.jev as jevmod
    monkeypatch.setattr(jevmod, "choice", lambda *a, **k: (called.append(1), ("x", {}, 0))[1])
    b = _Board()
    usi, src = sp.aiko_choose_move("u", b, book=real.load_book("u"),
                                   rng=random.Random(0), loss_lines=[])
    assert usi == "7g7f" and src == "book" and not called


def test_jev_fallback_is_random(sp, monkeypatch):
    import agentic.toolkit.jev as jevmod
    monkeypatch.setattr(jevmod, "choice", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    b = _Board()
    usi, src = sp.aiko_choose_move("u", b, book={}, rng=random.Random(0), loss_lines=[])
    assert src == "random-fallback" and usi in ("7g7f", "2g2f")


# ── full games ─────────────────────────────────────────────────────────────────

def _patch_learning(sp, monkeypatch, mb=None):
    rec = {"matches": [], "exp": []}
    monkeypatch.setattr("interface.android_app.learn.append_match",
                        lambda uid, game, rec_: rec["matches"].append((uid, game, rec_)))
    import agentic.experience.acquire as acq
    monkeypatch.setattr(acq, "record_experience",
                        lambda *a, **k: rec["exp"].append((a, k)) or "exp-id")
    if mb is not None:
        import cognition.fly_registry as reg
        monkeypatch.setattr(reg, "get_flymb", lambda uid=None: mb)
    return rec


class _FakeMB:
    def __init__(self):
        self.taught = []

    def encode(self, feats):
        return list(feats)

    def reinforce(self, pattern, reward):
        self.taught.append(reward)
        return 1.0


def test_aiko_wins_and_learns(sp, monkeypatch):
    import agentic.toolkit.jev as jevmod
    monkeypatch.setattr(jevmod, "choice", lambda state, ins, crit: (sorted(crit)[0], {}, 0.9))
    monkeypatch.setattr(sp, "engine_choose_move", lambda board, movetime_ms=None: "2g2f")
    mb = _FakeMB()
    rec = _patch_learning(sp, monkeypatch, mb)
    import interface.android_app.shogi.selfplay as real
    real._Board = _Board  # noqa - keep linters calm about the fake
    # script mate on ply 3 (after Aiko's 2nd move, engine to move mated)
    orig_board = _ShogiMod.__dict__["Board"]
    _ShogiMod.Board = staticmethod(lambda: _Board(script_end=3))
    try:
        out = sp.play_game("u", aiko_sente=True)
    finally:
        _ShogiMod.Board = orig_board
    assert out["winner"] == "aiko" and out["end"] == "checkmate"
    assert out["moves_made"] == 3
    uid, game, m = rec["matches"][0]
    assert game == "shogi_selfplay" and m["winner"] == "aiko"
    assert rec["exp"] and mb.taught == [1.0]


def test_engine_win_and_loss_lines(sp, monkeypatch):
    import agentic.toolkit.jev as jevmod
    monkeypatch.setattr(jevmod, "choice", lambda state, ins, crit: (sorted(crit)[0], {}, 0.9))
    monkeypatch.setattr(sp, "engine_choose_move", lambda board, movetime_ms=None: "2g2f")
    rec = _patch_learning(sp, monkeypatch)
    orig_board = _ShogiMod.__dict__["Board"]
    _ShogiMod.Board = staticmethod(lambda: _Board(script_end=2))
    try:
        out = sp.play_game("u", aiko_sente=True)
    finally:
        _ShogiMod.Board = orig_board
    assert out["winner"] == "engine"
    _uid, _game, _m = rec["matches"][0]
    assert _game == "shogi_selfplay" and _m["winner"] == "engine"
