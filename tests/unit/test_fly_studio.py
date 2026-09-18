import json

import pytest

from cognition.neural_state import clear_neural_state, get_neural_state, peek_neural_state
from interface.webui.studio.fly.backend import api


@pytest.mark.parametrize("handler", [api.fly_state, api.fly_circuit])
def test_fly_api_reads_existing_state_without_creating_and_disables_caching(monkeypatch, handler):
    user_id = "studio-reader"
    clear_neural_state(user_id)
    monkeypatch.setattr(api, "_uid", lambda _request: user_id)

    response = handler(object())

    assert response.headers["cache-control"] == "no-store"
    assert json.loads(response.body)["neural_state"] == {}
    assert peek_neural_state(user_id) is None


@pytest.mark.parametrize("handler", [api.fly_state, api.fly_circuit])
def test_fly_api_returns_existing_state(monkeypatch, handler):
    user_id = "studio-existing"
    clear_neural_state(user_id)
    get_neural_state(user_id).publish_mb(0.75)
    monkeypatch.setattr(api, "_uid", lambda _request: user_id)

    response = handler(object())

    assert response.headers["cache-control"] == "no-store"
    assert json.loads(response.body)["neural_state"]["valence"] == 0.75
