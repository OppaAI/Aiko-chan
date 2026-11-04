"""Tests for the no-system-role latch (ministral-type llama-server templates).

Some servers reject the `system` role with 500. The first rejection must
latch process-wide so later turns send user-merged messages upfront instead
of paying a doomed first attempt every turn.
"""
from __future__ import annotations

from types import SimpleNamespace

import cognition.think as think_mod
from cognition.think import AikoThink


SYSTEM_ROLE_500 = (
    "Error code: 500 - {'error': {'code': 500, 'message': "
    "\"Only user, assistant and tool roles ar... raise_exception('got system')\"}}"
)


def _chunk(text):
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))],
    )


def _think_with_client(create):
    think = object.__new__(AikoThink)
    think._reasoning = False
    think._deep_think = False
    think._llm_model = "test"
    think._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    think.last_usage = {}
    return think


def test_stream_converts_upfront_when_latched(monkeypatch):
    seen = {}

    def create(**kwargs):
        seen["messages"] = kwargs["messages"]
        return iter([_chunk("hello")])

    monkeypatch.setattr(think_mod, "_SYSTEM_ROLE_REJECTED", True)
    think = _think_with_client(create)
    text = think._stream_response(
        [{"role": "user", "content": "hi"}],
        system="SYS PROMPT",
        system_tail="TAIL",
    )
    assert text == "hello"
    roles = [m.get("role") for m in seen["messages"]]
    assert "system" not in roles
    blob = "\n".join(m.get("content", "") for m in seen["messages"])
    assert "SYS PROMPT" in blob and "TAIL" in blob and "hi" in blob


def test_first_rejection_latches_and_retries(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs["messages"])
        if any(m.get("role") == "system" for m in kwargs["messages"]):
            raise Exception(SYSTEM_ROLE_500)
        return iter([_chunk("recovered")])

    monkeypatch.setattr(think_mod, "_SYSTEM_ROLE_REJECTED", False)
    # Hermetic: the repo config sets LLM_NO_SYSTEM=1, which would merge
    # upfront and bypass the latch path this test exercises.
    monkeypatch.setattr(think_mod, "_LLM_NO_SYSTEM_ENV", False)
    think = _think_with_client(create)
    text = think._stream_response(
        [{"role": "user", "content": "hi"}],
        system="SYS PROMPT",
    )
    assert text == "recovered"
    assert len(calls) == 2
    assert think_mod._SYSTEM_ROLE_REJECTED is True
    assert "system" not in [m.get("role") for m in calls[1]]

    # Second turn goes out merged on the first attempt — no more 500s.
    calls.clear()
    think._stream_response([{"role": "user", "content": "again"}], system="SYS")
    assert len(calls) == 1
    assert "system" not in [m.get("role") for m in calls[0]]


def test_agentic_message_uses_latch_without_think_import(monkeypatch):
    from agentic.agentic import _stream_agent_message, _agent_messages_sendable

    # Duck-typed owner without the helpers: passthrough, errors propagate.
    bare = SimpleNamespace(_llm_model="m", _client=None)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert _agent_messages_sendable(bare, messages) is messages

    # Owner with the real helpers + a rejecting server: one retry, then merged.
    calls = []

    def create(**kwargs):
        calls.append(kwargs["messages"])
        if any(m.get("role") == "system" for m in kwargs["messages"]):
            raise Exception(SYSTEM_ROLE_500)
        return iter([_chunk("agent ok")])

    owner = SimpleNamespace(
        _llm_model="m",
        _client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        _should_avoid_system_role=AikoThink._should_avoid_system_role,
        _note_system_role_rejected=AikoThink._note_system_role_rejected,
        _is_system_role_error=AikoThink._is_system_role_error,
        _messages_without_system=AikoThink._messages_without_system,
    )
    monkeypatch.setattr(think_mod, "_SYSTEM_ROLE_REJECTED", False)
    # Hermetic: the repo config sets LLM_NO_SYSTEM=1, which would merge
    # upfront and bypass the latch path this test exercises.
    monkeypatch.setattr(think_mod, "_LLM_NO_SYSTEM_ENV", False)
    msg, _usage = _stream_agent_message(owner, messages, tools=[], token_callback=None)
    assert msg.content == "agent ok"
    assert len(calls) == 2
    assert think_mod._SYSTEM_ROLE_REJECTED is True
    assert "system" not in [m.get("role") for m in calls[1]]


def test_retry_filters_private_thinking_like_primary_stream(monkeypatch):
    calls = []
    def create(**kwargs):
        calls.append(kwargs['messages'])
        if len(calls) == 1:
            raise Exception(SYSTEM_ROLE_500)
        return iter([_chunk('<thinking>private thought</thinking>Public answer')])
    monkeypatch.setattr(think_mod, '_SYSTEM_ROLE_REJECTED', False)
    monkeypatch.setattr(think_mod, '_LLM_NO_SYSTEM_ENV', False)
    think = _think_with_client(create)
    assert think._stream_response([{'role': 'user', 'content': 'hi'}], system='SYS') == 'Public answer'
    assert think_mod._thinking_text_ctx.get() == 'private thought'


def test_failed_stream_cancels_early_audio_and_closes_connection(monkeypatch):
    events = []
    class BrokenStream:
        def __iter__(self):
            yield _chunk('One complete sentence. ')
            raise RuntimeError('connection lost')
        def close(self):
            events.append('close')
    think = _think_with_client(lambda **kwargs: BrokenStream())
    monkeypatch.setattr(think, '_fallback_completion', lambda *args: 'Recovered answer')
    monkeypatch.setattr(think_mod, '_SENTENCE_STREAM', True)
    sink = think_mod._SentenceSink(lambda sentence: True, on_abort=lambda: events.append('abort'))
    result = think._stream_response([{'role': 'user', 'content': 'hi'}], sentence_sink=sink)
    assert result == 'Recovered answer'
    assert events == ['abort', 'close']
    assert sink.aborted and sink.consumed_chars == 0


def test_reasoning_only_flag_is_request_local():
    from contextvars import copy_context
    think_mod._note_stream_reasoned(False)
    first, second = copy_context(), copy_context()
    first.run(think_mod._note_stream_reasoned, True)
    assert second.run(think_mod._take_stream_reasoned) is False
    assert first.run(think_mod._take_stream_reasoned) is True


def test_oversized_current_request_is_not_sent_to_model(monkeypatch):
    calls = []
    think = _think_with_client(lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv('LLM_CTX_SIZE', '1024')
    reply = think._stream_response([{'role': 'user', 'content': 'request ' * 5000}], system='Instructions')
    assert reply.startswith('[LLM error]')
    assert calls == []


def test_assembled_chat_budget_keeps_current_request_and_reserves_output(monkeypatch):
    from cognition.chat_budget import estimate_messages
    captured = []
    def create(**kwargs):
        captured.append(kwargs)
        return iter([_chunk('answer')])
    think = _think_with_client(create)
    monkeypatch.setenv('LLM_CTX_SIZE', '1024')
    monkeypatch.setattr(think_mod, '_LLM_NO_SYSTEM_ENV', False)
    monkeypatch.setattr(think_mod, '_SYSTEM_ROLE_REJECTED', False)
    messages = [{'role': 'user', 'content': 'old history ' * 1000},
                {'role': 'assistant', 'content': 'old answer'},
                {'role': 'user', 'content': 'Current question'}]
    assert think._stream_response(messages, system='Required instructions') == 'answer'
    sent = captured[0]
    assert 'Current question' in sent['messages'][-1]['content']
    assert all('old history' not in m['content'] for m in sent['messages'])
    assert estimate_messages(sent['messages']) + sent['max_tokens'] <= 1024 * 0.9 + 8
    assert len(messages) == 3
