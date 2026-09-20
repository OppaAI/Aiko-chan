"""Gating tests: the narrator must know the scene and the past — never the future.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_onmyoji_gatekeeper.py -q --override-ini="addopts="
"""
from __future__ import annotations

import pytest

from onmyoji.gatekeeper import (
    build_aiko_context,
    build_narrator_context,
    eligible_anchors,
    eligible_figures,
    present_entities,
)
from onmyoji.state import Entity, GameState


@pytest.fixture()
def state_1576_sakai():
    state = GameState()
    state.scene.location_id = "sakai"
    state.scene.date = "1576-06-01"
    state.scene.visible_detail = "Rain on the merchant row."
    state.player.inventory = ["ofuda papers", "salt pouch"]
    state.player.completed_quests = ["warded_warehouse"]
    return state


def test_future_anchors_excluded(state_1576_sakai):
    titles = [a["title"] for a in eligible_anchors(state_1576_sakai)]
    assert not any("Honno-ji" in t or "Sekigahara" in t or "Osaka unrest" in t for t in titles)
    assert any("merchant fire" in t for t in titles)  # local, window open


def test_realm_scope_rumors_included(state_1576_sakai):
    titles = [a["title"] for a in eligible_anchors(state_1576_sakai)]
    assert any("Okehazama" in t or "Nagashino" in t for t in titles)  # past realm events = rumors


def test_figure_place_and_date_gating(state_1576_sakai):
    names = [f["name"] for f in eligible_figures(state_1576_sakai)]
    assert "Sen no Rikyu" in names  # Sakai, active
    assert "Oda Nobunaga" not in names  # wrong city
    state_1576_sakai.scene.location_id = "azuchi"
    assert "Oda Nobunaga" in [f["name"] for f in eligible_figures(state_1576_sakai)]
    state_1576_sakai.scene.date = "1600-01-01"  # dead men tell no tales
    state_1576_sakai.scene.location_id = "kiyosu"
    assert "Oda Nobunaga" not in [f["name"] for f in eligible_figures(state_1576_sakai)]


def test_unmet_entities_invisible(state_1576_sakai):
    stranger = Entity(entity_id="mystery_man", kind="human", name="???", role="???")
    # not met, not present -> absent even though object exists
    assert present_entities(state_1576_sakai) == []
    state_1576_sakai.entities["mystery_man"] = stranger
    state_1576_sakai.scene.present_npc_ids = ["mystery_man"]
    assert [e.entity_id for e in present_entities(state_1576_sakai)] == ["mystery_man"]


def test_context_contains_state_not_future(state_1576_sakai):
    ctx = build_narrator_context(state_1576_sakai, task="test")
    assert "ofuda papers" in ctx and "warded_warehouse" in ctx and "Rain on the merchant row" in ctx
    assert "1582" not in ctx and "Honno-ji" not in ctx and "Sekigahara" not in ctx


def test_aiko_context_has_pact_voice(state_1576_sakai):
    ctx = build_aiko_context(state_1576_sakai)
    assert "bound shikigami" in ctx and "Bond=" in ctx
