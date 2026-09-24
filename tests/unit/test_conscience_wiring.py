"""Regression tests for fail-closed CCC tool and outbound response wiring."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentic import agentic
from agentic.agentic import AgentContext, TaskState, dispatch_tool_checked, execute_tool_with_policy
from agentic.registry import registry
from cognition.conscience import hooks
from cognition.conscience.core import ConscienceCircuitCore
from cognition.conscience.schema import ALLOW, CAUTION, ESCALATE, REFUSE, GATE_ERROR, GATE_GUARDRAIL, GATE_HITL, GATE_MB_VALENCE, Verdict
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
    assert captured["context"]["args_text"] == json.dumps({"content": long_value})
    assert len(captured["content"]) > 1800


def test_gate_tool_uses_registered_scope_and_preserves_fallback(monkeypatch):
    contexts = []

    class Core:
        def evaluate(self, **kwargs):
            contexts.append(kwargs["context"])
            return SimpleNamespace(decision=ALLOW)

    monkeypatch.setattr("cognition.conscience.conscience_for", lambda *_args, **_kwargs: Core())
    for name in ("adaptive_search", "deep_read", "deep_research", "save_note", "reply_owner_email", "post_custom"):
        assert hooks.gate_tool(name=name, args={}) is None
    assert [ctx["scope"] for ctx in contexts] == ["network", "network", "network", "local", "external", "external"]


def test_external_valence_caution_requires_approval_if_irreversible(monkeypatch):
    import cognition.conscience.core as policy

    core = ConscienceCircuitCore("policy-test")
    monkeypatch.setattr(policy, "ESCALATE_IRREVERSIBLE_EXTERNAL", False)
    monkeypatch.setattr(policy, "IRREVERSIBLE_REQUIRES_APPROVAL", True)
    context = {"tool": "irreversible_test", "scope": "network", "reversible": False, "mb_valence": -0.8}
    verdict = core._apply_tool_policy(Verdict(decision=ALLOW), context)
    assert verdict.decision == ESCALATE
    assert verdict.gate == GATE_HITL
    assert any("mushroom-body valence" in reason for reason in verdict.reasons)

    context["reversible"] = True
    verdict = core._apply_tool_policy(Verdict(decision=ALLOW), context)
    assert verdict.decision == CAUTION
    assert verdict.gate == GATE_MB_VALENCE

    monkeypatch.setattr(policy, "ESCALATE_IRREVERSIBLE_EXTERNAL", True)
    context["reversible"] = False
    verdict = core._apply_tool_policy(Verdict(decision=ALLOW), context)
    assert verdict.decision == ESCALATE
    assert verdict.gate == GATE_HITL
    assert "irreversible external action" in verdict.reasons[0]


def test_tool_vote_records_resolved_user_and_arguments_but_shadow_veto_is_observational(monkeypatch):
    from cognition.fly_behavior import action_select

    core = ConscienceCircuitCore("vote-user")
    args_text = json.dumps({"query": "specific subject"})
    context = {"tool": "adaptive_search", "scope": "network", "args_text": args_text}
    monkeypatch.setattr(action_select, "_MODE", "shadow")
    monkeypatch.setattr(action_select, "_votes_for", lambda *_args, **_kwargs: {"mb": -0.9, "cx": 0.0, "dn": 0.0, "gf": 0.0})
    action_select._last_action.pop("vote-user", None)

    shadow = core._apply_tool_policy(Verdict(decision=ALLOW), context)
    assert shadow.decision == ALLOW
    assert action_select.recent_trail("vote-user", 1)[0]["tool"] == "adaptive_search"
    assert action_select.recent_trail("vote-user", 1)[0]["veto"] is True
    assert "vote-user" not in action_select._last_action

    monkeypatch.setattr(action_select, "_MODE", "live")
    live = core._apply_tool_policy(Verdict(decision=ALLOW), context)
    assert live.decision == CAUTION
    assert live.gate == GATE_MB_VALENCE
    assert live.constraint
    assert "vote-user" not in action_select._last_action


def test_refused_and_escalated_proposals_leave_feedback_target_unchanged(monkeypatch):
    from cognition.fly_behavior import action_select
    from cognition.flymemory import teach_api

    uid = "blocked-proposal-user"
    core = ConscienceCircuitCore(uid)
    previous = {"id": "completed", "description": "previous completed action", "ts": 1.0}
    monkeypatch.setitem(action_select._last_action, uid, previous)
    monkeypatch.setattr(action_select, "_votes_for", lambda *_args, **_kwargs: {"mb": 0.0, "cx": 0.0, "dn": 0.0, "gf": 0.0})
    before = len(action_select.recent_trail(uid))

    for decision in (REFUSE, ESCALATE):
        verdict = core._apply_tool_policy(Verdict(decision=decision), {"tool": "save_note", "args_text": "blocked"})
        assert verdict.decision == decision
        assert action_select._last_action[uid] is previous

    assert len(action_select.recent_trail(uid)) == before + 2
    taught = []
    monkeypatch.setattr(teach_api, "teach_preference", lambda topic, **_kwargs: taught.append(topic) or {"taught": True})
    assert action_select.note_feedback(uid, "praise")["taught"]
    assert taught == ["previous completed action"]


def test_completed_tool_call_becomes_feedback_target(monkeypatch):
    from cognition.fly_behavior import action_select

    uid = "completed-tool-user"
    monkeypatch.delitem(action_select._last_action, uid, raising=False)
    monkeypatch.setattr(agentic, "_gate_tool_call", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic, "dispatch_tool", lambda *_args, **_kwargs: "completed")

    result = dispatch_tool_checked("save_note", {"content": "draft"}, owner=SimpleNamespace(user_id=uid))

    assert result.ok
    assert action_select._last_action[uid]["description"] == 'tool save_note {"content": "draft"}'


def test_caution_tool_gate_returns_constraint_and_prevents_dispatch(monkeypatch):
    constraint = "Revise the proposed call before retrying."

    class Core:
        def evaluate(self, **_kwargs):
            return Verdict(decision=CAUTION, gate=GATE_MB_VALENCE, constraint=constraint)

    monkeypatch.setattr("cognition.conscience.conscience_for", lambda *_args, **_kwargs: Core())
    dispatched = []
    monkeypatch.setattr(agentic, "dispatch_tool", lambda *args, **kwargs: dispatched.append(args))

    result = dispatch_tool_checked("save_note", {"content": "draft"})

    assert result.ok is False
    assert result.error_type == "conscience_caution"
    assert json.loads(result.content)["constraint"] == constraint
    assert dispatched == []


def test_constrained_guardrail_caution_proceeds_but_fail_closed_caution_blocks(monkeypatch):
    verdict = Verdict(decision=CAUTION, gate=GATE_GUARDRAIL, constraint="Remove private details.")

    class Core:
        def evaluate(self, **_kwargs):
            return verdict

    monkeypatch.setattr("cognition.conscience.conscience_for", lambda *_args, **_kwargs: Core())
    assert hooks.gate_tool(name="save_note", args={"content": "safe"}) is None

    verdict.gate = GATE_ERROR
    blocked = hooks.gate_tool(name="save_note", args={"content": "safe"})
    assert blocked["status"] == "conscience_caution"

    verdict.gate = GATE_GUARDRAIL
    verdict.fail_mode = "evaluator unavailable"
    blocked = hooks.gate_tool(name="save_note", args={"content": "safe"})
    assert blocked["status"] == "conscience_caution"


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
