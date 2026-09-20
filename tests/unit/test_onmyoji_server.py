"""Server contract tests: request bodies must parse (regression).

FastAPI cannot resolve Pydantic models defined inside create_app(), which
once made every POST return 422 "Field required". These tests pin the body
contract with a mocked LLM backend.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_onmyoji_server.py -q --override-ini="addopts="
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import onmyoji.server as srv


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_STORE_PATH", str(tmp_path / "save.json"))
    return TestClient(srv.create_app())


def test_talk_body_parses(client):
    with patch.object(srv, "_chat", return_value="Mocked."):
        r = client.post("/talk", json={"to": "aiko", "text": "hi"})
    assert r.status_code == 200, r.text
    assert r.json()["speaker"] == "aiko"


def test_act_body_parses_and_rejects_unknown_verb(client):
    with patch.object(srv, "_chat", return_value="Mocked."):
        r = client.post("/act", json={"verb": "bogus", "args": {}})
    assert r.status_code == 200
    assert "unknown verb" in r.json()["error"]


def test_act_search_mutates_inventory(client):
    with patch.object(srv, "_chat", return_value="Mocked."):
        r = client.post("/act", json={"verb": "search", "args": {"find": "salt pouch"}})
    assert r.status_code == 200
    assert r.json()["effects"] == ["found: salt pouch"]
    assert "salt pouch" in client.get("/state").json()["player"]["inventory"]


def test_android_contract_health_state_options(client):
    assert client.get("/api/onmyoji/health").json() == {"ok": True, "game": "onmyoji", "phase": 1}
    state = client.get("/api/onmyoji/state").json()
    assert state["location"] == "sakai" and state["hp"] == 10 and state["mp"] == 10
    options = client.get("/api/onmyoji/options").json()
    assert "travel" in options["actions"] and "kyoto" in options["destinations"]


def test_android_start_act_talk_shapes(client):
    journey = client.post("/api/onmyoji/start", json={"location": "kyoto", "date": "1582-01-01"}).json()
    assert journey["location"] == "kyoto" and journey["date"] == "1582-01-01"
    with patch.object(srv, "_chat", return_value="Mocked."):
        acted = client.post("/api/onmyoji/act", json={"action": "attack", "target": "gaki"}).json()
    assert acted["journey"]["mp"] == 9  # strike costs 1 mp
    assert any("gaki" in e for e in acted["events"])
    with patch.object(srv, "_chat", return_value="Mocked."):
        talked = client.post("/api/onmyoji/talk", json={"target": "aiko", "message": "hi"}).json()
    assert talked["ok"] is True and talked["journey"]["location"] == "kyoto"
