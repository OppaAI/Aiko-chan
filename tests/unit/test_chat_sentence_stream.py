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
        lambda user_input, resp, cb=None, already_emitted=None, _spoken_prefix="", **kw: resp
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


def test_no_history_still_sends_current_request(monkeypatch):
    think = _stream_think(monkeypatch, None)
    think._history = [{'role': 'user', 'content': 'Old question'}, {'role': 'assistant', 'content': 'Old answer'}]
    captured = []
    def stream(messages, **kwargs):
        captured.extend(messages)
        return 'answer'
    monkeypatch.setattr(think, '_stream_response', stream)
    think.chat('Current task', skip_memory=True, store_turn=False, include_history=False)
    assert captured == [{'role': 'user', 'content': 'Current task'}]


def test_request_style_reaches_model_without_polluting_history(monkeypatch):
    from cognition.attention import soft_user_prompt
    think = _stream_think(monkeypatch, None)
    captured = []
    def stream(messages, **kwargs):
        captured.extend(messages)
        return 'answer'
    monkeypatch.setattr(think, '_stream_response', stream)
    prompt = soft_user_prompt('Original request', 'clarify', 'test reason')
    think.chat(prompt, skip_memory=True, store_turn=False)
    assert captured[-1]['content'] == prompt
    assert think._history[0]['content'] == 'Original request'


def test_llm_failure_is_not_stored_as_conversation_or_memory(monkeypatch):
    think = _stream_think(monkeypatch, None)
    stores = []
    monkeypatch.setattr(think, '_stream_response', lambda *a, **kw: '[LLM error] unavailable')
    monkeypatch.setattr(think, '_store_async', lambda *a: stores.append(a))
    displayed = []
    assert think.chat('Question', token_callback=displayed.append, skip_memory=True) == '[LLM error] unavailable'
    assert think._history == []
    assert stores == []
    assert displayed == ['[LLM error] unavailable']


def test_chat_forwards_related_sentence_questions_into_hold(monkeypatch):
    from cognition.conscience import hooks
    think = _stream_think(monkeypatch, None)
    think._stream_response = lambda *a, **kw: 'answer'
    captured = []
    def finalize(user_input, response, cb=None, **kw):
        kw['_ask_out'].update({
            'escalation_id': 'abc123', 'related_escalation_ids': ['def456', 'abc789'],
            'held_text': response, 'sentence_indices': [0, 1], 'categories': [],
            'user_id': 'u1', 'notice_summary': 'turn flagged',
        })
        return response
    think._finalize_response = finalize
    monkeypatch.setattr(hooks, 'hold_output_for_approval', lambda eid, **kw: captured.append((eid, kw)))
    monkeypatch.setattr(hooks, 'push_turn_ask_notice', lambda *a: None)
    assert think.chat('hello', skip_memory=True, store_turn=False) == 'answer'
    assert captured[0][0] == 'abc123'
    assert captured[0][1]['related_escalation_ids'] == ['def456', 'abc789']


def test_failed_stream_discards_pending_sentence_questions(monkeypatch):
    from cognition.conscience import hooks
    think = _stream_think(monkeypatch, _FakeSpeak())
    def make_sink(user_input, speak, cb, state):
        state['ask_flags'] = [{'escalation_id': 'def456'}]
        return lambda sentence: True
    think._make_sentence_sink = make_sink
    think._stream_response = lambda *a, **kw: '[LLM error] failed after a sentence'
    discarded = []
    monkeypatch.setattr(hooks, 'discard_output_questions', lambda ids, **kw: discarded.extend(ids))
    assert think.chat('hello', skip_memory=True, store_turn=False).startswith('[LLM error]')
    assert discarded == ['def456']
    assert think._history == []
