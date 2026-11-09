"""Fail-closed mode-gate tests for the opt-in FlyWorld training CLI."""
from __future__ import annotations

import os

import pytest

from scripts.train_flyworld import _configure_modes


def test_default_training_forces_shadow(monkeypatch):
    monkeypatch.setenv("AIKO_FLYWORLD_MODE", "live")
    monkeypatch.setenv("AIKO_FLY_DOPAMINE_MODE", "live")
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")

    _configure_modes(apply=False, user_id=None)

    assert os.environ["AIKO_FLYWORLD_MODE"] == "shadow"
    assert os.environ["AIKO_FLY_DOPAMINE_MODE"] == "shadow"
    assert os.environ["MEMORY_FLYMB_MODE"] == "shadow"


def test_apply_requires_explicit_identity(monkeypatch):
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("AIKO_FLY_DOPAMINE_MODE", "live")
    with pytest.raises(ValueError, match="--user-id"):
        _configure_modes(apply=True, user_id=None)


@pytest.mark.parametrize("missing", ["MEMORY_FLYMB_MODE", "AIKO_FLY_DOPAMINE_MODE"])
def test_apply_requires_both_live_gates(monkeypatch, missing):
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("AIKO_FLY_DOPAMINE_MODE", "live")
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match=missing):
        _configure_modes(apply=True, user_id="owner")


def test_apply_enables_only_flyworld_after_gates(monkeypatch):
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("AIKO_FLY_DOPAMINE_MODE", "live")
    _configure_modes(apply=True, user_id="owner")
    assert os.environ["AIKO_FLYWORLD_MODE"] == "live"
