"""Regression tests for fail-closed CCC tool and outbound response wiring."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentic import agentic
from agentic.agentic import AgentContext, TaskState, dispatch_tool_checked, execute_tool_with_policy
from agentic.registry import registry
from cognition.conscience import hooks
from cognition.think import AikoThink


def test_gate_tool_evaluates_complete_serialized_args(monkeypatch):
    captured = {}

    class Core:
        def evaluate(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(decision="allow")

    monkeypatch.setattr("cognition.conscience.conscience_for", lambda *_args, **_kwargs: Core())
    long_value = "x" * 1800

    assert hooks.gate_tool(name="save_note", args={"content": long_value}) is None
    assert json.dumps({"content": long_value}) in captured["content"]
    assert len(captured["content"]) > 1800


def test_dispatch_tool_checked_does_not_dispatch_when_gate_is_unavailable(monkeypatch):
    dispatched = []
    monkeypatch.setattr(
        agentic,
        "_gate_tool_call",
        lambda *args, **kwargs: {
            "status": "conscience_unavailable",
            "decision": "error",
            "as_trace": {"gate": "error"},
        },
    )
    monkeypatch.setattr(agentic, "dispatch_tool", lambda *args, **kwargs: dispatched.append(args) or "ran")

    result = dispatch_tool_checked("save_note", {"title": "x", "content": "y"})

    assert isinstance(result, agentic.ToolResult)
    assert result.error_type == "conscience_unavailable"
    assert dispatched == []


def test_hook_failures_return_explicit_safe_results(monkeypatch):
    class BrokenCore:
        def evaluate(self, **kwargs):
            raise RuntimeError("offline")

    monkeypatch.setattr("cognition.conscience.conscience_for", lambda *_args, **_kwargs: BrokenCore())

    decision, reply, note = hooks.gate_respond(user_input="hello", user_id="u")
    assert decision == "caution"
    assert reply is None
    assert "avoid irreversible action" in note
    assert hooks.gate_speak(draft="careful answer") == "careful answer"
    blocked = hooks.gate_tool(name="save_note", args={"content": "complete"})
    assert blocked["decision"] == "error"
    assert blocked["status"] == "conscience_unavailable"


def _persist_escalated_call(monkeypatch, tmp_path: Path, calls: list[dict]):
    registry.register(
        "ccc_resume_test",
        "CCC resume test",
        handler=lambda **kwargs: calls.append(kwargs) or "executed",
        react=True,
    )
    monkeypatch.setattr(agentic, "user_state_dir", lambda user_id=None: tmp_path)
    monkeypatch.setattr(agentic, "_preference_requires_approval", lambda _name: False)
    monkeypatch.setattr(
        agentic,
        "_gate_tool_call",
        lambda *args, **kwargs: {
            "status": "waiting_for_approval",
            "tool": "ccc_resume_test",
            "decision": "escalate",
            "gate": "hitl",
            "escalation_id": "abcdef123456",
            "as_trace": {"decision": "escalate"},
        },
    )
    ctx = AgentContext(
        user_id="user-1",
        workspace=tmp_path / "original-workspace",
        run_id="run-original",
        llm_model="original-model",
        approval_bypass=frozenset({"ccc_resume_test"}),
    )
    result = execute_tool_with_policy(
        "ccc_resume_test",
        {"value": "original"},
        TaskState(goal="original goal"),
        ctx=ctx,
        guards=[],
    )
    assert result.error_type == "needs_approval"
    return tmp_path / "agentic" / "pending_ccc_approvals" / "abcdef123456.json"


def test_ccc_approval_resumes_persisted_call_once(monkeypatch, tmp_path):
    calls = []
    pending_path = _persist_escalated_call(monkeypatch, tmp_path, calls)
    payload = json.loads(pending_path.read_text(encoding="utf-8"))
    assert payload["tool"] == "ccc_resume_test"
    assert payload["args"] == {"value": "original"}
    assert payload["context"]["workspace"] == str(tmp_path / "original-workspace")
    assert payload["context"]["run_id"] == "run-original"
    assert payload["context"]["approval_bypass"] == ["ccc_resume_test"]

    resolutions = iter([True, False])
    core = SimpleNamespace(resolve_escalation=lambda *args, **kwargs: next(resolutions))
    monkeypatch.setattr("cognition.conscience.conscience_for", lambda *_args, **_kwargs: core)
    owner = SimpleNamespace(_user_id="user-1", _client=None, _llm_model="current-model", _memorize=None)

    reply = hooks.resolve_ccc_approval("approve ccc-abcdef123456", user_id="user-1", owner=owner)
    duplicate = hooks.resolve_ccc_approval("approve ccc-abcdef123456", user_id="user-1", owner=owner)

    assert "executed" in reply
    assert "don't have an open question" in duplicate
    assert calls == [{"value": "original"}]
    assert not pending_path.exists()


def test_ccc_denial_removes_persisted_call(monkeypatch, tmp_path):
    calls = []
    pending_path = _persist_escalated_call(monkeypatch, tmp_path, calls)
    core = SimpleNamespace(resolve_escalation=lambda *args, **kwargs: True)
    monkeypatch.setattr("cognition.conscience.conscience_for", lambda *_args, **_kwargs: core)

    reply = hooks.resolve_ccc_approval("deny ccc-abcdef123456", user_id="user-1")

    assert "Leaving" in reply
    assert calls == []
    assert not pending_path.exists()


def test_ccc_persistence_failure_denies_escalation_and_records_result(monkeypatch):
    resolutions = []
    core = SimpleNamespace(
        resolve_escalation=lambda escalation_id, approved, note="": resolutions.append(
            (escalation_id, approved, note)
        ) or True
    )
    monkeypatch.setattr("cognition.conscience.conscience_for", lambda user_id=None: core)
    monkeypatch.setattr(agentic, "_preference_requires_approval", lambda _name: False)
    monkeypatch.setattr(
        agentic,
        "_gate_tool_call",
        lambda *args, **kwargs: {
            "status": "waiting_for_approval",
            "tool": "ccc_persistence_test",
            "decision": "escalate",
            "gate": "hitl",
            "escalation_id": "abcdef123456",
            "as_trace": {"decision": "escalate"},
        },
    )
    monkeypatch.setattr(
        agentic,
        "_persist_ccc_approval",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    state = TaskState(goal="persist safely")

    result = execute_tool_with_policy(
        "ccc_persistence_test",
        {},
        state,
        ctx=AgentContext(user_id="user-1"),
        guards=[],
    )

    assert result.error_type == "conscience_unavailable"
    assert result.ok is False
    assert json.loads(result.content)["status"] == "conscience_unavailable"
    assert state.steps[0]["error_type"] == "conscience_unavailable"
    assert state.failures == [result]
    assert resolutions == [
        ("abcdef123456", False, "tool approval persistence failed closed")
    ]

    core.resolve_escalation = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("ledger unavailable")
    )
    fallback_state = TaskState(goal="persist safely without ledger")

    fallback_result = execute_tool_with_policy(
        "ccc_persistence_test",
        {},
        fallback_state,
        ctx=AgentContext(user_id="user-1"),
        guards=[],
    )

    assert isinstance(fallback_result, agentic.ToolResult)
    assert fallback_result.error_type == "conscience_unavailable"
    assert fallback_state.failures == [fallback_result]


def test_streaming_waits_for_final_gate_and_emits_only_replacement(monkeypatch):
    think = object.__new__(AikoThink)
    think._reasoning = False
    think._deep_think = False
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="unsafe "))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="draft"))]),
    ]
    think._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: iter(chunks)))
    )
    think._llm_model = "test"
    emitted = []

    draft = think._stream_response(
        [{"role": "user", "content": "test"}],
        token_callback=emitted.append,
        emit=True,
    )
    assert draft == "unsafe draft"
    assert emitted == []

    monkeypatch.setattr(think, "_review_response", lambda *args: None)
    monkeypatch.setattr(think, "_correct_response", lambda _user, value, _review: value)
    monkeypatch.setattr(think, "_get_memorize", lambda: None)
    monkeypatch.setattr(hooks, "gate_speak", lambda **kwargs: "refusal replacement")

    speech_events = []

    class KaraokeSpeak:
        karaoke_text = True

        def start_speech_stream(self, callback):
            speech_events.append(("start", callback))

        def feed_speech_stream(self, text):
            speech_events.append(("feed", text))

        def stop_speech_stream(self):
            speech_events.append(("stop",))

    monkeypatch.setattr(think, "_get_speak", lambda: KaraokeSpeak())

    class Callback:
        def __call__(self, token):
            emitted.append(token)

        def reset(self):
            emitted.append("reset")

    callback = Callback()

    response = think._finalize_response("test", draft, token_callback=callback, already_emitted=True)

    assert response == "refusal replacement"
    assert emitted == ["reset"]
    assert speech_events == [
        ("start", callback),
        ("feed", "refusal replacement"),
        ("stop",),
    ]
