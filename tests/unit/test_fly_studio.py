import json
import logging

import pytest
from fastapi.testclient import TestClient

from cognition.neural_state import clear_neural_state, get_neural_state, peek_neural_state
from interface.webui import auth
from interface.webui.studio.fly.backend import api, body_routes


@pytest.mark.parametrize("handler", [api.fly_state, api.fly_circuit, api.fly_trace])
def test_fly_api_reads_existing_state_without_creating_and_disables_caching(monkeypatch, handler):
    user_id = "studio-reader"
    clear_neural_state(user_id)
    monkeypatch.setattr(api, "_uid", lambda _request: user_id)

    response = handler(object())

    assert response.headers["cache-control"] == "no-store"
    assert json.loads(response.body)["neural_state"] == {}
    assert peek_neural_state(user_id) is None


@pytest.mark.parametrize("handler", [api.fly_state, api.fly_circuit, api.fly_trace])
def test_fly_api_returns_existing_state(monkeypatch, handler):
    user_id = "studio-existing"
    clear_neural_state(user_id)
    get_neural_state(user_id).publish_mb(0.75)
    monkeypatch.setattr(api, "_uid", lambda _request: user_id)

    response = handler(object())

    assert response.headers["cache-control"] == "no-store"
    assert json.loads(response.body)["neural_state"]["valence"] == 0.75


def test_fly_trace_hides_internal_errors_and_logs_details(monkeypatch, caplog):
    def fail_trace(_user_id):
        raise RuntimeError("sensitive detail")

    monkeypatch.setattr(api, "_uid", lambda _request: "studio-error")
    monkeypatch.setattr("cognition.fly_runtime.peek_fly_runtime", fail_trace)

    with caplog.at_level(logging.ERROR, logger=api.logger.name):
        response = api.fly_trace(object())

    body = json.loads(response.body)
    assert body["trace"] == {"error": "trace unavailable"}
    assert body["neural_state"] == {}
    assert "sensitive detail" in caplog.text


def test_fly_causal_hides_internal_errors_and_returns_500(monkeypatch, caplog):
    def fail_causal(_user_id):
        raise RuntimeError("sensitive detail")

    monkeypatch.setattr(api, "_uid", lambda _request: "studio-error")
    monkeypatch.setattr("cognition.neural_state.peek_neural_state", fail_causal)

    with caplog.at_level(logging.ERROR, logger=api.logger.name):
        response = api.fly_causal(object())

    body = json.loads(response.body)
    assert response.status_code == 500
    assert response.headers["cache-control"] == "no-store"
    assert body == {
        "user_id": "studio-error",
        "error": "causal trail unavailable",
        "events": [],
    }
    assert "sensitive detail" not in body["error"]
    assert "sensitive detail" in caplog.text


def test_body_payload_hides_internal_errors_and_logs_details(monkeypatch, caplog):
    def fail_body_drive(*, user_id):
        raise RuntimeError(f"sensitive detail for {user_id}")

    monkeypatch.setattr("cognition.fly_behavior.dn_body.body_drive", fail_body_drive)

    with caplog.at_level(logging.ERROR, logger=body_routes.logger.name):
        payload = body_routes.build_body_payload("studio-error")

    assert payload["body"] == {"mode": "off"}
    assert "sensitive detail" not in json.dumps(payload)
    assert "sensitive detail for studio-error" in caplog.text


def test_fly_body_fallback_hides_internal_errors_and_logs_details(monkeypatch, caplog):
    def fail_payload(_user_id):
        raise RuntimeError("sensitive detail")

    monkeypatch.setattr(api, "_uid", lambda _request: "studio-error")
    monkeypatch.setattr(body_routes, "build_body_payload", fail_payload)

    with caplog.at_level(logging.DEBUG, logger=api.logger.name):
        response = api.fly_body(object())

    body = json.loads(response.body)
    assert body["body"] == {"mode": "off"}
    assert body["avatar_intents"] == []
    assert "sensitive detail" not in json.dumps(body)
    assert "sensitive detail" in caplog.text


def test_fly_activity_preserves_zero_arousal_and_circadian(monkeypatch):
    user_id = "studio-zero"
    state = get_neural_state(user_id)
    state.action_drive = 0.0
    state.circadian_phase = 0.0
    monkeypatch.setattr(api, "_uid", lambda _request: user_id)

    activity = json.loads(api.fly_activity(object()).body)["activity"]

    assert activity["arousal"] == 0.0
    assert activity["circadian"] == 0.0
    clear_neural_state(user_id)


def test_mounted_fly_api_requires_session():
    client = TestClient(auth.app)

    response = client.get("/studio/fly/api/state")

    assert response.status_code == 401
