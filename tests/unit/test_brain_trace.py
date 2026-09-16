"""Tests for the brain-trace sink contract: steps -> log only (one record per
step), turn banners -> log + UI sink, nothing sprays the chat per line.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_brain_trace.py -q --override-ini="addopts="
"""
from __future__ import annotations

import logging

import pytest

from system import brain_trace as bt


class FakeSink:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def add_message(self, sender: str, text: str) -> None:
        self.messages.append((sender, text))


@pytest.fixture()
def enabled(monkeypatch):
    monkeypatch.setattr(bt, "TRACE_ENABLED", True)
    bt.reset()
    yield
    bt.reset()


@pytest.fixture()
def test_logger(monkeypatch, caplog):
    """Route the module logger at a capture-friendly stdlib logger (no real files)."""
    logger = logging.getLogger("test.brain_trace")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = True
    monkeypatch.setattr(bt, "_log", logger)
    with caplog.at_level(logging.INFO, logger="test.brain_trace"):
        yield caplog


class TestStepSink:
    def test_step_is_one_log_record_no_ui(self, enabled, test_logger):
        sink = FakeSink()
        bt.set_ui_sink(sink)
        try:
            bt.record_step("think.route", layer="route", inputs={"q": "hi"},
                           outputs={"dest": "chat"}, factors=["why"])
        finally:
            bt.set_ui_sink(None)
        assert len(test_logger.records) == 1
        assert "think.route" in test_logger.records[0].message
        assert sink.messages == []

    def test_step_context_manager_start_end(self, enabled, test_logger):
        sink = FakeSink()
        bt.set_ui_sink(sink)
        try:
            with bt.step("memorize.search", layer="recall") as ctx:
                ctx.set(outputs={"hits": 3})
        finally:
            bt.set_ui_sink(None)
        assert len(test_logger.records) == 2  # start frame + end frame
        assert sink.messages == []

    def test_banners_go_to_ui_and_log(self, enabled, test_logger):
        sink = FakeSink()
        bt.set_ui_sink(sink)
        try:
            bt.begin_turn("t")
            bt.end_turn()
        finally:
            bt.set_ui_sink(None)
        assert [s for s, _ in sink.messages] == ["sys", "sys"]
        assert len(test_logger.records) == 2

    def test_disabled_emits_nothing(self, monkeypatch, test_logger):
        monkeypatch.setattr(bt, "TRACE_ENABLED", False)
        sink = FakeSink()
        bt.set_ui_sink(sink)
        try:
            bt.begin_turn()
            bt.record_step("think.route")
            bt.end_turn()
        finally:
            bt.set_ui_sink(None)
        assert test_logger.records == []
        assert sink.messages == []

    def test_no_ansi_in_log_records(self):
        assert bt._strip_ansi("\033[38;5;141mhello\033[0m") == "hello"
        assert bt._strip_ansi("plain") == "plain"
