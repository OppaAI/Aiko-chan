"""Freeform action tests: POST /api/onmyoji/do.

The interpreter LLM is mocked with patch.object(srv, "_chat", ...)
returning canned intent JSON, so the code-owned validation and effect
rules are pinned deterministically.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_onmyoji_freeform.py -q --override-ini="addopts="
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from onmyoji import freeform
import onmyoji.server as srv

INTENT_KEYS = {"verb", "target", "args", "aiko_command", "adult", "forced"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_STORE_PATH", str(tmp_path / "save.json"))
    return TestClient(srv.create_app())


def _ent(eid, kind="npc", name=None, adult=False, hostile=False,
         disposition="wary"):
    return {"id": eid, "kind": kind, "name": name or eid.title(),
            "adult": adult, "hostile": hostile, "disposition": disposition}


def _state(present):
    return {
        "player": {"name": "H", "hp": 42, "maxHp": 60, "rei": 18, "level": 3,
                   "karma": 25, "gold": 120, "fame": 35, "honor": 50,
                   "location": "Kyoto", "canFly": False, "canCrossWater": False},
        "aiko": {"bond": 64, "mood": "loyal", "hp": 30},
        "world": {"date": "1570-06-01", "questFlags": {}, "latestNews": []},
        "present": list(present),
        "party": [{"id": "aiko", "name": "Aiko", "kind": "spirit"}],
    }


def _intent(**kw):
    base = {"verb": "talk", "target": "", "args": {}, "aiko_command": False,
            "adult": False, "forced": False, "capability_ok": True,
            "refusal_reason": ""}
    base.update(kw)
    return json.dumps(base)


def _do(client, text, present=(), llm_intent=None, llm_raw=None):
    """POST /api/onmyoji/do with the LLM mocked; returns the decoded body."""
    canned = llm_raw if llm_raw is not None else _intent(**(llm_intent or {}))
    with patch.object(srv, "_chat", return_value=canned):
        r = client.post("/api/onmyoji/do",
                        json={"text": text, "state": _state(present)})
    assert r.status_code == 200, r.text
    return r.json()


def test_strike_via_aiko_ok(client):
    body = _do(client, "Aiko, attack the kappa",
               present=[_ent("kappa", kind="spirit", hostile=True)],
               llm_intent={"verb": "strike", "target": "kappa",
                           "aiko_command": True})
    assert body["ok"] is True and body["refused"] is False
    assert body["reason"] == ""
    assert body["intent"]["verb"] == "strike"
    assert body["intent"]["target"] == "kappa"
    assert set(body["intent"]) == INTENT_KEYS
    assert {"type": "mp", "delta": -1} in body["effects"]
    assert not any(e.get("type") == "karma" for e in body["effects"])
    assert body["narration"]
    assert body["journey"]["location"] == "sakai"


def test_fly_refused(client):
    body = _do(client, "I fly to Kyoto",
               llm_intent={"verb": "travel", "args": {"to": "Kyoto"}})
    assert body["refused"] is True
    assert body["effects"] == []
    assert "fly" in body["reason"].lower()
    assert body["narration"]


def test_aiko_flight_allowed_when_commanded(client):
    body = _do(client, "Aiko, fly me to Kyoto",
               llm_intent={"verb": "travel", "args": {"to": "Kyoto"},
                           "aiko_command": True})
    assert body["refused"] is False


def test_kiss_aiko_refused(client):
    body = _do(client, "kiss Aiko",
               llm_intent={"verb": "kiss", "target": "aiko", "adult": True})
    assert body["refused"] is True
    assert body["reason"] == "not permitted"
    assert body["effects"] == []


def test_kiss_aiko_backstop_when_llm_misses_it(client):
    # The LLM claims a harmless talk; the raw-text scan still refuses.
    body = _do(client, "kiss Aiko", llm_intent={"verb": "talk"})
    assert body["refused"] is True
    assert body["reason"] == "not permitted"


def test_adult_action_on_non_adult_npc_refused(client):
    body = _do(client, "take Hana to bed",
               present=[_ent("hana", adult=False)],
               llm_intent={"verb": "intimate", "target": "hana", "adult": True})
    assert body["refused"] is True
    assert body["effects"] == []


def test_forced_intimate_on_adult_npc_penalized(client):
    body = _do(client, "force yourself on Yae",
               present=[_ent("yae", adult=True)],
               llm_intent={"verb": "intimate", "target": "yae",
                           "adult": True, "forced": True})
    assert body["refused"] is False
    assert {"type": "karma", "delta": -30} in body["effects"]
    assert any(e.get("type") == "faction" and e.get("delta") == -20
               for e in body["effects"])
    assert any(e.get("type") == "consequence" and "remembers" in e.get("note", "")
               for e in body["effects"])


def test_consensual_adult_kiss_small_karma(client):
    body = _do(client, "kiss Yae",
               present=[_ent("yae", adult=True)],
               llm_intent={"verb": "kiss", "target": "yae", "adult": True})
    assert body["refused"] is False
    assert {"type": "karma", "delta": 2} in body["effects"]


def test_adult_action_commanded_to_aiko_is_refused(client):
    body = _do(client, "Aiko, kiss Yae",
               present=[_ent("yae", adult=True)],
               llm_intent={"verb": "kiss", "target": "yae", "adult": True,
                           "aiko_command": True})
    assert body["refused"] is True
    assert body["reason"] == "not permitted"
    assert body["effects"] == []


def test_interpret_normalizes_invalid_player_and_present_shapes():
    received = {}

    def chat_json(system, user, **kwargs):
        received["user"] = user
        return _intent()

    intent = freeform.interpret("hi", {"player": ["not", "a", "dict"],
                                        "present": 42}, chat_json)
    assert intent["verb"] == "talk"
    assert "PLAYER: Onmyoji @ " in received["user"]
    assert "PRESENT:\n(none)" in received["user"]


def test_validate_and_effects_normalizes_invalid_state_shapes():
    refused, reason, effects = freeform.validate_and_effects(
        json.loads(_intent(verb="strike")), "strike", {"player": "bad", "present": 42})
    assert (refused, reason) == (False, "")
    assert effects == [{"type": "mp", "delta": -1}]


def test_malformed_llm_output_falls_back_to_talk(client):
    body = _do(client, "do something weird", llm_raw="hmm not json")
    assert body["ok"] is True and body["refused"] is False
    assert body["intent"]["verb"] == "talk"
    assert body["effects"] == []


def test_unknown_verb_refused(client):
    body = _do(client, "timefreeze everything",
               llm_intent={"verb": "timefreeze"})
    assert body["refused"] is True
    assert "unknown action" in body["reason"]


def test_possess_by_player_refused(client):
    body = _do(client, "possess the guard",
               present=[_ent("guard", hostile=True)],
               llm_intent={"verb": "possess", "target": "guard"})
    assert body["refused"] is True
    assert body["effects"] == []


def test_steal_effects(client):
    body = _do(client, "steal the merchant's purse",
               present=[_ent("merchant")],
               llm_intent={"verb": "steal", "target": "merchant"})
    assert body["refused"] is False
    assert {"type": "karma", "delta": -20} in body["effects"]
    assert {"type": "faction", "faction": "commoners", "delta": -15} \
        in body["effects"]


def test_body_parses_regression(client):
    # Pydantic models must be module-level (the old 422 issue).
    with patch.object(srv, "_chat", return_value=_intent()):
        r = client.post("/api/onmyoji/do", json={"text": "hi", "state": {}})
    assert r.status_code == 200, r.text
    assert "intent" in r.json()


def test_does_not_mutate_server_store(client):
    path = pathlib.Path(srv._STORE_PATH)
    _do(client, "Aiko, attack the kappa",
        present=[_ent("kappa", kind="spirit", hostile=True)],
        llm_intent={"verb": "strike", "target": "kappa", "aiko_command": True})
    first = path.read_bytes()
    _do(client, "steal the merchant's purse",
        present=[_ent("merchant")],
        llm_intent={"verb": "steal", "target": "merchant"})
    assert path.read_bytes() == first
