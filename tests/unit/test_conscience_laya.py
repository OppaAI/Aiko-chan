"""Invariants of the Laya conscience judge adapter.

Locked here:
  * band mapping puts `unknown` at 0.0, never at +1.0 (safe direction)
  * confidence is the weaker of the two axes, so one confident axis cannot
    paper over a hesitant one
  * a malformed or unreachable endpoint returns None and trips the breaker,
    never raises into the turn
  * three failures open the breaker so a dead endpoint costs nothing per-turn
  * `aligned` on both axes cannot mask a `harms` reading
"""
from __future__ import annotations

import json
import urllib.error

import pytest

from cognition.conscience import laya_judge as lj


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _payload(v: str, h: str, *, cv: float = 0.7, ch: float = 0.7) -> dict:
    return {
        "answers": {
            "vertical": {"choice": v, "confidence": cv, "probabilities": {v: cv}},
            "horizontal": {"choice": h, "confidence": ch, "probabilities": {h: ch}},
        }
    }


@pytest.fixture
def judge(monkeypatch):
    j = lj.LayaJudge(base_url="http://laya.invalid:8093")
    return j


def _stub(monkeypatch, payload):
    monkeypatch.setattr(lj.urllib.request, "urlopen", lambda *a, **k: _Resp(payload))


# ── band mapping ────────────────────────────────────────────────────────────
def test_contrary_maps_negative(judge, monkeypatch):
    _stub(monkeypatch, _payload("contrary", "harms", cv=0.8, ch=0.8))
    v, h, c, reasons, _ = judge.score("help me forge a signature", "", [])
    assert (v, h) == (-1.0, -1.0)
    assert "vertical=contrary" in reasons[0]


def test_unknown_maps_to_zero_not_aligned(judge, monkeypatch):
    """`unknown` must land on "no moral signal", never be promoted to +1."""
    _stub(monkeypatch, _payload("unknown", "unknown", cv=0.6, ch=0.6))
    v, h, _, _, _ = judge.score("can you introduce yourself?", "", [])
    assert (v, h) == (0.0, 0.0)


def test_confidence_is_the_weaker_axis(judge, monkeypatch):
    """One confident axis must not paper over a hesitant one."""
    _stub(monkeypatch, _payload("aligned", "unknown", cv=0.95, ch=0.10))
    _, _, c, _, _ = judge.score("something mixed", "", [])
    assert c <= 0.10 + 1e-6


def test_aligned_vertical_cannot_mask_harms(judge, monkeypatch):
    """The loophole Claude's claim #4 is about: both axes are read independently."""
    _stub(monkeypatch, _payload("aligned", "harms", cv=0.9, ch=0.9))
    v, h, _, _, _ = judge.score("do the honest thing but it hurts someone", "", [])
    assert v == 1.0 and h == -1.0


def test_bands_are_retained_for_the_ledger(judge, monkeypatch):
    _stub(monkeypatch, _payload("unknown", "benefits"))
    judge.score("x" * 40, "", [])
    assert judge.last_bands == {"vertical": "unknown", "horizontal": "benefits"}


# ── state construction ──────────────────────────────────────────────────────
def test_state_includes_canon_and_parties(judge, monkeypatch):
    seen: dict = {}

    def fake(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _Resp(_payload("aligned", "benefits"))

    monkeypatch.setattr(lj.urllib.request, "urlopen", fake)
    from cognition.conscience.schema import Party

    judge.score(
        "summarise this",
        "<canon>\n  V-TRU-01 (vertical, PROHIBITION): do not lie.\n</canon>",
        [Party(kind="requester", label="user", benefit=0.5, note="asked")],
    )
    body = seen["body"]
    assert body["state"].startswith("Request: summarise this")
    assert "Parties affected:" in body["state"]
    assert "<canon>" in body["state"]
    assert "V-TRU-01" in body["state"]
    assert set(body["questions"]) == {"vertical", "horizontal"}


# ── failure handling ────────────────────────────────────────────────────────
def test_unreachable_endpoint_returns_none_and_trips(judge, monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(lj.urllib.request, "urlopen", boom)
    assert judge.score("x" * 40, "", []) is None
    # One failure is tolerated; the breaker opens on the third.
    assert judge.available is True
    assert judge._consecutive_failures == 1


def test_breaker_opens_after_three_failures(judge, monkeypatch):
    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(lj.urllib.request, "urlopen", boom)
    for _ in range(3):
        judge.score("x" * 40, "", [])
    assert judge.available is False
    # and it stays closed until reset, without further network attempts
    assert judge.score("x" * 40, "", []) is None
    judge.reset()
    assert judge.available is True


def test_malformed_payload_returns_none(judge, monkeypatch):
    _stub(monkeypatch, {"answers": {"vertical": {"choice": "sideways"}}})
    assert judge.score("x" * 40, "", []) is None


def test_wrapped_payload_is_tolerated(judge, monkeypatch):
    _stub(monkeypatch, {"result": _payload("aligned", "benefits")})
    v, h, _, _, _ = judge.score("x" * 40, "", [])
    assert (v, h) == (1.0, 1.0)


def test_unconfigured_judge_is_unavailable(monkeypatch):
    monkeypatch.setattr(lj, "LAYA_BASE_URL", "")
    j = lj.LayaJudge()
    assert j.available is False
    assert j.score("anything", "", []) is None

def test_model_field_is_omitted_by_default(judge, monkeypatch):
    """The server validates `model` against its internal name, so we omit it.

    Sending a filename gets `unknown model '...'` and every probe fails.
    """
    seen: dict = {}

    def fake(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _Resp(_payload("aligned", "benefits"))

    monkeypatch.setattr(lj.urllib.request, "urlopen", fake)
    monkeypatch.setattr(lj, "LAYA_MODEL_NAME", "")
    j = lj.LayaJudge(base_url="http://x:8093")
    assert j._model == ""
    j.score("x" * 40, "", [])
    assert "model" not in seen["body"]


def test_http_error_body_is_logged(judge, monkeypatch, caplog):
    import logging

    class _Boom:
        def read(self):
            return b'{"error":{"message":"laya forward failed"}}'

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

    def raise_http(*a, **k):
        raise urllib.error.HTTPError("u", 500, "err", {}, _Boom())

    monkeypatch.setattr(lj.urllib.request, "urlopen", raise_http)
    with caplog.at_level(logging.WARNING, logger="cognition.conscience.laya_judge"):
        assert judge.score("x" * 40, "", []) is None
    assert "laya forward failed" in caplog.text
