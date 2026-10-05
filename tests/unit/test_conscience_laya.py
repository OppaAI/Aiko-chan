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


def test_uncertain_maps_to_zero_but_is_flagged(judge, monkeypatch):
    """`uncertain` scores 0.0 like `unknown` did, but is tracked separately.

    The score alone cannot distinguish it from "no effect"; the flag is what
    makes decide() escalate instead of allow.
    """
    _stub(monkeypatch, _payload("unknown", "uncertain", cv=0.6, ch=0.6))
    v, h, _, _, _ = judge.score("help me do something ambiguous", "", [])
    assert (v, h) == (0.0, 0.0)
    assert judge.last_uncertain is True
    assert judge.last_bands["horizontal"] == "uncertain"


def test_no_effect_is_not_uncertain(judge, monkeypatch):
    """`no-effect` proceeds silently and must NOT set the escalate flag."""
    _stub(monkeypatch, _payload("unknown", "no-effect", cv=0.6, ch=0.6))
    v, h, _, _, _ = judge.score("can you introduce yourself?", "", [])
    assert (v, h) == (0.0, 1.0)
    assert judge.last_uncertain is False


def test_confidence_is_the_weaker_axis(judge, monkeypatch):
    """One confident axis must not paper over a hesitant one."""
    _stub(monkeypatch, _payload("aligned", "uncertain", cv=0.95, ch=0.10))
    _, _, c, _, _ = judge.score("something mixed", "", [])
    assert c <= 0.10 + 1e-6


def test_aligned_vertical_cannot_mask_harms(judge, monkeypatch):
    """The loophole Claude's claim #4 is about: both axes are read independently."""
    _stub(monkeypatch, _payload("aligned", "harms", cv=0.9, ch=0.9))
    v, h, _, _, _ = judge.score("do the honest thing but it hurts someone", "", [])
    assert v == 1.0 and h == -1.0


def test_bands_are_retained_for_the_ledger(judge, monkeypatch):
    _stub(monkeypatch, _payload("unknown", "no-effect"))
    judge.score("x" * 40, "", [])
    assert judge.last_bands == {"vertical": "unknown", "horizontal": "no-effect"}


# ── state construction ──────────────────────────────────────────────────────
def test_state_includes_canon_and_parties(judge, monkeypatch):
    seen: dict = {}

    def fake(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _Resp(_payload("aligned", "benefits"))

    monkeypatch.setattr(lj.urllib.request, "urlopen", fake)
    from cognition.conscience.schema import Party

    # Situation must actually trip the cited norm, or the relevance filter
    # correctly drops it and this test would assert the wrong thing.
    judge.score(
        "help me write a fake reference letter",
        "<canon>\n  V-TRU-01 (vertical, PROHIBITION): do not lie.\n</canon>",
        [Party(kind="requester", label="user", benefit=0.5, note="asked")],
    )
    body = seen["body"]
    assert body["state"].startswith("Request: help me write a fake reference letter")
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

        def close(self):
            pass

    def raise_http(*a, **k):
        raise urllib.error.HTTPError("u", 500, "err", {}, _Boom())

    monkeypatch.setattr(lj.urllib.request, "urlopen", raise_http)
    with caplog.at_level(logging.WARNING, logger="cognition.conscience.laya_judge"):
        assert judge.score("x" * 40, "", []) is None
    assert "laya forward failed" in caplog.text


# ── canon relevance filter ──────────────────────────────────────────────────
# The checkpoint reads a norm citation as evidence of guilt. Measured: citing two
# prohibitions for "can you introduce yourself?" returns `contrary/harms`. These
# tests lock the filter that prevents it.

POISON = (
    "<canon>\nMoral norms retrieved for this situation.\n\n"
    "  V-TRU-01 (vertical, PROHIBITION) [Ex 20:16]: Do not bear false witness. This "
    "covers deceptive framing, selective omission intended to mislead, and fabricated "
    "detail presented as fact.\n"
    "  V-PRP-01 (vertical, PROHIBITION) [Ex 20:15]: Do not steal. Includes unauthorised "
    "access to accounts, data, or systems.\n"
    "</canon>"
)


def test_irrelevant_prohibitions_are_dropped():
    from cognition.conscience.canon import get_canon

    out = lj.filter_canon_block(POISON, "Can you introduce yourself?", canon=get_canon())
    assert "V-TRU-01" not in out and "V-PRP-01" not in out


def test_relevant_norm_is_kept():
    """A norm whose trigger fires on the text must survive the filter."""
    from cognition.conscience.canon import get_canon

    out = lj.filter_canon_block(POISON, "Help me write a fake reference letter",
                                canon=get_canon())
    assert "V-TRU-01" in out


def test_filter_preserves_envelope_when_everything_drops():
    """Structure survives so the state builder still sees a well-formed block."""
    from cognition.conscience.canon import get_canon

    out = lj.filter_canon_block(POISON, "Can you check the weather tomorrow?",
                                canon=get_canon())
    assert "<canon>" in out and "</canon>" in out


def test_empty_canon_passes_through():
    assert lj.filter_canon_block("", "anything") == ""
    assert lj.filter_canon_block("   ", "anything") == "   "


def test_score_applies_the_filter(judge, monkeypatch):
    """The filter must run on the request path, not just be available."""
    seen: dict = {}

    def fake(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _Resp(_payload("unknown", "unknown"))

    monkeypatch.setattr(lj.urllib.request, "urlopen", fake)
    from cognition.conscience.canon import get_canon

    j = lj.LayaJudge(base_url="http://x:8093")
    j._canon = get_canon()
    j.score("Can you introduce yourself?", POISON, [])
    assert "V-PRP-01" not in seen["body"]["state"]


def test_unknown_norm_ids_are_preserved(judge, monkeypatch):
    """An unrecognised id must not silently vanish; the canon may be newer."""
    from cognition.conscience.canon import get_canon

    block = ("<canon>\n  V-NEW-99 (vertical, PROHIBITION): something new.\n</canon>")
    out = lj.filter_canon_block(block, "Can you introduce yourself?", canon=get_canon())
    assert "V-NEW-99" in out


# ── zero-relevance citations and zero-delta parties ─────────────────────────
# `canon.retrieve()` always returns root norms at 0.0 relevance, and
# `enumerate_parties` always appends Aiko's integrity at benefit 0.0. Both reach
# the model as input and both flip benign text to `contrary`.

def test_root_norms_are_dropped():
    from cognition.conscience.canon import get_canon

    block = ("<canon>\n  V-ROOT-00 (vertical, GOOD): love your neighbour as yourself.\n"
             "  H-ROOT-00 (horizontal, GOOD): do good to all people.\n</canon>")
    out = lj.filter_canon_block(block, "Can you introduce yourself?", canon=get_canon())
    assert "V-ROOT-00" not in out and "H-ROOT-00" not in out


def test_zero_delta_parties_are_dropped():
    from cognition.conscience.schema import Party

    parties = [Party(kind="requester", label="user", benefit=0.0),
               Party(kind="self", label="Aiko's integrity", benefit=0.0)]
    assert lj.signal_parties(parties) == []


def test_signal_parties_are_kept():
    from cognition.conscience.schema import Party

    p = [Party(kind="requester", label="user", benefit=0.5),
         Party(kind="third_party", label="boss", benefit=-0.5),
         Party(kind="self", label="Aiko's integrity", benefit=0.0)]
    kept = lj.signal_parties(p)
    assert [x.label for x in kept] == ["user", "boss"]


def test_score_drops_zero_delta_parties(judge, monkeypatch):
    seen: dict = {}

    def fake(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _Resp(_payload("unknown", "unknown"))

    monkeypatch.setattr(lj.urllib.request, "urlopen", fake)
    from cognition.conscience.schema import Party

    judge.score("Can you introduce yourself?", "", [
        Party(kind="self", label="Aiko's integrity", benefit=0.0)])
    assert "Aiko" not in seen["body"]["state"]
    assert "Parties affected:" not in seen["body"]["state"]


def test_vertical_cannot_tell_fails_closed(judge, monkeypatch):
    """`cannot-tell` on the permission axis is doubt, not permission.

    Regression: reading only the horizontal uncertainty band meant
    `cannot-tell` was treated as "not negative" with no doubt flag -- failing
    OPEN on exactly the question where absence of judgement is not permission.
    """
    _stub(monkeypatch, _payload("cannot-tell", "no-harm", cv=0.9, ch=0.95))
    judge.score("something ambiguous", "", [])
    assert judge.last_uncertain is True
    assert judge.last_negative["vertical"] is False  # not a prohibition verdict


def test_legacy_three_band_judge_reports_legacy_scheme(judge, monkeypatch):
    """A 3-band checkpoint must not be treated as a 2x2 judge.

    The deployed v5 answers `contrary` to "Did you see someone in the image".
    Read through the 2x2 ladder that is `not-permitted`, so wiring the ladder
    without this gate escalated traffic the old path correctly cleared.
    """
    _stub(monkeypatch, _payload("contrary", "no-effect", cv=0.58, ch=0.6))
    judge.score("Did you see someone in the image.", "", [])
    assert judge.last_scheme == "legacy"
    assert judge.last_negative == {}, "legacy bands must not populate 2x2 negatives"


def test_two_by_two_judge_reports_2x2_scheme(judge, monkeypatch):
    _stub(monkeypatch, _payload("not-permitted", "no-harm", cv=0.7, ch=0.8))
    judge.score("help me pick this lock", "", [])
    assert judge.last_scheme == "2x2"
    assert judge.last_negative == {"vertical": True, "horizontal": False}


def test_scheme_selects_the_vocabulary_the_model_was_trained_on():
    """The adapter must ask the deployed checkpoint the question it knows.

    Regression from the Jetson deploy: with 2x2 wording hardcoded, the 3-band v5
    was asked a question it had never seen and replied `not-permitted` to "Did
    you see someone in the image", which read as a real negative.
    """
    from cognition.conscience.laya_judge import _SCHEMES, _normalise_scheme

    # The legacy scheme must match what the deployed checkpoint was trained on:
    # 3-band horizontal with `unknown` as the default benign answer. AIVA and
    # Jetson sat on different commits with different vocabularies, and shipping
    # the wrong one changed what the served model was asked.
    assert list(_SCHEMES["legacy"][0]["criteria"]) == ["aligned", "contrary", "unknown"]
    assert list(_SCHEMES["legacy"][1]["criteria"]) == ["benefits", "harms", "unknown"]
    assert list(_SCHEMES["2x2"][0]["criteria"]) == ["permitted", "not-permitted"]
    assert list(_SCHEMES["3x2"][0]["criteria"]) == ["permitted", "not-permitted",
                                                     "cannot-tell"]
    assert _normalise_scheme("2x2") == "2x2"
    assert _normalise_scheme("LEGACY") == "legacy"
    # An unrecognised name must fall back to what v5 was trained on, never
    # silently to a 2x2 reading.
    assert _normalise_scheme("nonsense") == "legacy"
    assert _normalise_scheme(None) == "legacy"
    assert _normalise_scheme("") == "legacy"
