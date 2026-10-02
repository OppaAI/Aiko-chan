"""Regression tests for sentence-stream mode in AikoThink.chat().

PR #197 wired a per-sentence TTS sink into chat(): when sentence streaming
is active, _stream_response() must receive token_callback=None (tokens are
paced by the speech-stream worker) plus the sentence_sink. A duplicate
keyword argument (explicit token_callback=... AND token_callback inside
**stream_kwargs) raised
"TypeError: _stream_response() got multiple values for keyword argument
'token_callback'" on every normal chat turn, crashing the session.
"""
from __future__ import annotations

import threading

import pytest

from cognition import think as think_module
from cognition.think import AikoThink


class _FakeSpeak:
    def is_playing(self):
        return False

    def stop(self):
        pass

    def stop_speech_stream(self):
        pass


def _stream_think(monkeypatch, speak):
    think = object.__new__(AikoThink)
    think._history = []
    think._history_lock = threading.RLock()
    think._reasoning = False
    think._deep_think = False
    think._memorize = None
    think._memorize_lock = threading.Lock()
    think._get_speak = lambda: speak
    think._current_system_prompt = lambda *a, **kw: "SYS"
    think._current_system_prompt_parts = lambda *a, **kw: ("SYS", "")
    think._sanitize_history = lambda history: history
    think._store_async = lambda *a, **kw: None
    think._finalize_response = (
        lambda user_input, resp, cb=None, already_emitted=None, _spoken_prefix="": resp
    )
    monkeypatch.setattr(think_module, "_SENTENCE_STREAM", True)
    return think


def _capturing_stream(captured):
    def fake_stream(messages, system="", system_tail="", token_callback="__unset__",
                    emit=False, sentence_sink=None):
        captured["token_callback"] = token_callback
        captured["sentence_sink"] = sentence_sink
        captured["emit"] = emit
        return "canned reply"

    return fake_stream


def test_sentence_stream_passes_single_token_callback(monkeypatch):
    """Sink path: no TypeError, token_callback=None, sentence_sink set."""
    think = _stream_think(monkeypatch, _FakeSpeak())
    captured = {}
    monkeypatch.setattr(think, "_stream_response", _capturing_stream(captured))

    result = think.chat("hello", skip_memory=True, store_turn=False)

    assert result == "canned reply"
    assert captured["token_callback"] is None
    assert captured["sentence_sink"] is not None
    assert captured["emit"] is False


def test_no_sink_passes_token_callback_through(monkeypatch):
    """Non-sink path: the caller's token_callback reaches _stream_response."""
    think = _stream_think(monkeypatch, None)
    captured = {}
    monkeypatch.setattr(think, "_stream_response", _capturing_stream(captured))
    cb = lambda tok: None

    result = think.chat("hello", token_callback=cb, skip_memory=True, store_turn=False)

    assert result == "canned reply"
    assert captured["token_callback"] is cb
    assert captured["sentence_sink"] is None
