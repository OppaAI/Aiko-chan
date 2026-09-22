import json
import logging

import pytest
from fastapi.testclient import TestClient

from cognition.neural_state import clear_neural_state, get_neural_state, peek_neural_state
from interface.webui import auth
from interface.webui.studio.fly.backend import api


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


def test_mounted_fly_api_requires_session():
    client = TestClient(auth.app)

    response = client.get("/studio/fly/api/state")

    assert response.status_code == 401
