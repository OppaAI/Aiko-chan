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
    msg, _usage = _stream_agent_message(owner, messages, tools=[], token_callback=None)
    assert msg.content == "agent ok"
    assert len(calls) == 2
    assert think_mod._SYSTEM_ROLE_REJECTED is True
    assert "system" not in [m.get("role") for m in calls[1]]
