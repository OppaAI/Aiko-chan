"""Unit tests for system/resource.py cooperative RAM hygiene."""
from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

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


def test_companion_rss_matches_case_insensitive_override(monkeypatch):
    psutil = ModuleType("psutil")
    psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    psutil.ZombieProcess = type("ZombieProcess", (Exception,), {})
    psutil.process_iter = lambda _attrs: [SimpleNamespace(
        pid=123, info={"pid": 123, "cmdline": ["/opt/miotts/server"],
                       "memory_info": SimpleNamespace(rss=1024 * 1024)},
    )]
    monkeypatch.setitem(sys.modules, "psutil", psutil)

    assert ram.companion_rss_mb(patterns=("MioTTS",)) == {"miotts:123": 1.0}


def test_companion_rss_logs_missing_memory_but_ignores_exit_race(monkeypatch, caplog):
    psutil = ModuleType("psutil")
    psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    psutil.ZombieProcess = type("ZombieProcess", (Exception,), {})

    class Exited:
        pid = 123

        @property
        def info(self):
            raise psutil.NoSuchProcess()

    exited = Exited()
    missing = SimpleNamespace(
        pid=124, info={"pid": 124, "cmdline": ["miotts"], "memory_info": None},
    )
    psutil.process_iter = lambda _attrs: [exited, missing]
    monkeypatch.setitem(sys.modules, "psutil", psutil)

    assert ram.companion_rss_mb(patterns=("miotts",)) == {}
    assert "companion process 124" in caplog.text
    assert "memory_info is unavailable" in caplog.text
    assert "companion process 123" not in caplog.text
