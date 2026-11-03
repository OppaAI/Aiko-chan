"""Tests for Stage 1 SOUL teaching configuration."""

from cognition.flymemory import soul_teach


def test_mode_defaults_off_and_preserves_explicit_value(monkeypatch):
    monkeypatch.delenv("FLY_SOUL_TEACH_ON_BOOT", raising=False)
    assert soul_teach._mode() == "off"

    monkeypatch.setenv("FLY_SOUL_TEACH_ON_BOOT", "ShAdOw")
    assert soul_teach._mode() == "shadow"


def test_mode_fallback_defaults_off_and_preserves_explicit_value(monkeypatch):
    def unavailable_env_str(*_args):
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("system.config.env_str", unavailable_env_str)
    monkeypatch.delenv("FLY_SOUL_TEACH_ON_BOOT", raising=False)
    assert soul_teach._mode() == "off"

    monkeypatch.setenv("FLY_SOUL_TEACH_ON_BOOT", "LIVE")
    assert soul_teach._mode() == "live"
