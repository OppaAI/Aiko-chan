"""
tests/unit/test_fetch_heal.py

Unit tests for agentic/toolkit/fetch_heal.py — the per-domain strategy
ledger behind web_fetch's self-healing order. No network; the ledger
path is redirected to tmp_path via FETCH_HEAL_FILE monkeypatching of
user_state_dir.
"""
from __future__ import annotations

import json
import time

import pytest

from agentic.toolkit import fetch_heal


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("system.userspace.user_state_dir", lambda *a, **k: tmp_path)
    return tmp_path


def _entry(domain, strat, fails, age=0):
    now = time.time()
    return {domain: {"strategies": {strat: {
        "consecutive_fails": fails, "last_fail": now - age}},
        "updated": now}}


class TestPreferredOrder:
    def test_unknown_domain_returns_available_defaults(self, ledger_dir, monkeypatch):
        monkeypatch.setattr(fetch_heal, "strategy_available", lambda s: True)
        order = fetch_heal.preferred_order("https://example.com/x")
        assert order[0] == "trafilatura"
        assert set(order) <= set(fetch_heal.STRATEGIES)

    def test_chronic_failure_deprioritized(self, ledger_dir, monkeypatch):
        (ledger_dir / "fetch_health.json").write_text(
            json.dumps(_entry("example.com", "trafilatura", 5)))
        monkeypatch.setattr(fetch_heal, "strategy_available", lambda s: True)
        order = fetch_heal.preferred_order("https://example.com/x")
        assert order[-1] == "trafilatura"
        assert order[0] != "trafilatura"

    def test_stale_failures_forgiven(self, ledger_dir, monkeypatch):
        (ledger_dir / "fetch_health.json").write_text(
            json.dumps(_entry("example.com", "trafilatura", 9,
                              age=fetch_heal.FETCH_HEAL_FAIL_TTL_SECONDS + 10)))
        monkeypatch.setattr(fetch_heal, "strategy_available", lambda s: True)
        order = fetch_heal.preferred_order("https://example.com/x")
        assert order[0] == "trafilatura"

    def test_unavailable_strategies_dropped(self, ledger_dir, monkeypatch):
        monkeypatch.setattr(fetch_heal, "strategy_available",
                            lambda s: s == "trafilatura")
        assert fetch_heal.preferred_order("https://example.com/x") == ["trafilatura"]

    def test_corrupt_ledger_degrades_to_defaults(self, ledger_dir, monkeypatch):
        monkeypatch.setattr(fetch_heal, "strategy_available", lambda s: True)
        (ledger_dir / "fetch_health.json").write_text("not json{{{")
        order = fetch_heal.preferred_order("https://example.com/x")
        assert order[0] == "trafilatura"


class TestRecord:
    def test_success_resets_failures(self, ledger_dir):
        url = "https://example.com/x"
        for _ in range(4):
            fetch_heal.record(url, "trafilatura", False)
        fetch_heal.record(url, "trafilatura", True)
        data = json.loads((ledger_dir / "fetch_health.json").read_text())
        hist = data["example.com"]["strategies"]["trafilatura"]
        assert hist["consecutive_fails"] == 0
        assert hist["last_ok"] > 0

    def test_unknown_strategy_ignored(self, ledger_dir):
        fetch_heal.record("https://example.com/x", "telepathy", False)
        assert not (ledger_dir / "fetch_health.json").exists()

    def test_ledger_pruned_at_cap(self, ledger_dir, monkeypatch):
        monkeypatch.setattr(fetch_heal, "FETCH_HEAL_MAX_DOMAINS", 3)
        for i in range(6):
            fetch_heal.record(f"https://d{i}.example/x", "trafilatura", False)
        data = json.loads((ledger_dir / "fetch_health.json").read_text())
        assert len(data) <= 3


class TestWebFetchHealing:
    def test_healed_order_skips_dead_default(self, ledger_dir, monkeypatch):
        import agentic.toolkit.websearch as ws
        (ledger_dir / "fetch_health.json").write_text(
            json.dumps(_entry("example.com", "trafilatura", 5)))
        monkeypatch.setattr(fetch_heal, "strategy_available", lambda s: True)
        monkeypatch.setattr(ws, "_tavily_extract",
                            lambda url, max_chars: "tavily rescued text " * 30)
        monkeypatch.setattr(ws, "ingest_from_url",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("trafilatura path must not run")))
        out = ws.web_fetch("https://example.com/x", use_cache=False)
        assert out.startswith("tavily rescued text")

    def test_good_default_stops_before_vendor_call(self, ledger_dir, monkeypatch):
        import sys
        import importlib.util as _ilu
        from types import ModuleType
        import agentic.toolkit.websearch as ws
        fake = ModuleType("trafilatura")
        fake.extract = lambda downloaded, **k: "word " * 500
        monkeypatch.setitem(sys.modules, "trafilatura", fake)
        monkeypatch.setattr(_ilu, "find_spec", lambda name, *a, **k: object())
        monkeypatch.setattr(fetch_heal, "strategy_available", lambda s: True)
        calls = []
        monkeypatch.setattr(ws, "ingest_from_url",
                            lambda *a, **k: (b"<html><body><p>" + b"word " * 500 + b"</p></body></html>", None))
        monkeypatch.setattr(ws, "_tavily_extract",
                            lambda url, max_chars: calls.append(url) or "x" * 500)
        out = ws.web_fetch("https://example.com/good", use_cache=False)
        assert not out.startswith("[fetch failed")
        assert calls == [], "vendor fallback must not fire on good extraction"
