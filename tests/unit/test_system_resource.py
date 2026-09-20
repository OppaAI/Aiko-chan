"""Unit tests for system/resource.py cooperative RAM hygiene."""
from __future__ import annotations

import os

import system.resource as ram


def test_release_ram_reports_and_throttles(monkeypatch):
    monkeypatch.setenv("RAM_RELEASE_MIN_INTERVAL_S", "300")
    ram._last_release_at = 0.0
    first = ram.release_ram("test")
    assert first["ok"] is True
    assert "before_mb" in first and "after_mb" in first
    second = ram.release_ram("test")
    assert second == {"ok": False, "skipped": "throttled"}


def test_release_ram_zero_interval_always_runs(monkeypatch):
    monkeypatch.setenv("RAM_RELEASE_MIN_INTERVAL_S", "0")
    ram._last_release_at = 0.0
    assert ram.release_ram("test")["ok"] is True
    assert ram.release_ram("test")["ok"] is True


def test_rss_mb_positive_or_none():
    value = ram.rss_mb()
    assert value is None or value > 0


def test_release_ram_survives_broken_gc(monkeypatch):
    monkeypatch.setenv("RAM_RELEASE_MIN_INTERVAL_S", "0")
    ram._last_release_at = 0.0
    monkeypatch.setattr(ram.gc, "collect", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = ram.release_ram("test")
    assert out["ok"] is True
    assert out["gc_collected"] == -1


def test_malloc_trim_returns_bool():
    assert isinstance(ram._malloc_trim(), bool)
