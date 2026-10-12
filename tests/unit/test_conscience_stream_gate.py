"""Two-tier streamed conscience gate: BLOCK purges, ASK holds for accept/decline.

Locked here:
  * gate_speak(return_verdict=True) exposes ESCALATE (else it collapses to "keep draft")
  * gate_speak default behavior is unchanged (backwards compatible)
  * held outputs are one-shot, content-hash bound, TTL-bounded, in-memory only
  * approve commits only when hash + user + position all check out; else purge
  * deny/expired/mismatch always purges (fail closed); approval never overrides
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from cognition.conscience import ESCALATE, REFUSE, ALLOW
from cognition.conscience.schema import Verdict
from cognition.conscience import hooks


def _verdict(decision, **kw):
    v = Verdict(decision=decision, gate="judge", act="speak",
                reasons=[kw.pop("reason", "because")], **kw)
    v.escalation_id = kw.get("escalation_id", "abc123")
    return v


def _mock_conscience(decision, **kw):
    v = _verdict(decision, **kw)
    return patch("cognition.conscience.conscience_for",
                 return_value=type("C", (), {"evaluate": lambda self, **k: v})())


# ── gate_speak verdict visibility ────────────────────────────────────────────

def test_gate_speak_default_unchanged_on_escalate():
    with _mock_conscience(ESCALATE):
        out = hooks.gate_speak(draft="hello")
    assert out is None  # keep draft, as before


def test_gate_speak_return_verdict_exposes_escalate():
    with _mock_conscience(ESCALATE, suggested_category="self-harm", p_violation=0.62):
        replaced, verdict = hooks.gate_speak(draft="hello", return_verdict=True)
    assert replaced is None
    assert verdict.decision == ESCALATE
    assert verdict.suggested_category == "self-harm"


def test_gate_speak_return_verdict_on_refuse():
    with _mock_conscience(REFUSE, pastoral_note="nope"):
        replaced, verdict = hooks.gate_speak(draft="hello", return_verdict=True)
    assert replaced == "nope"
    assert verdict.decision == REFUSE


def test_gate_speak_failure_tuple_when_verdict_requested():
    with patch("cognition.conscience.conscience_for", side_effect=RuntimeError("down")):
        replaced, verdict = hooks.gate_speak(draft="hello", return_verdict=True)
    assert replaced == "hello"  # fail-open default preserved
    assert verdict is None


def test_gate_speak_quiet_escalation_reaches_context():
    seen = {}

    class C:
        def evaluate(self, **kw):
            seen.update(kw.get("context") or {})
            return _verdict(ESCALATE)

    with patch("cognition.conscience.conscience_for", return_value=C()):
        hooks.gate_speak(draft="hello", quiet_escalation=True)
    assert seen.get("quiet_escalation") is True


# ── held-output registry ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_registry():
    hooks._HELD_OUTPUTS.clear()
    yield
    hooks._HELD_OUTPUTS.clear()


def _hold(eid="e1", text="held text here", **kw):
    d = dict(held_text=text, sentence_indices=[0], categories=["x"],
             user_id="u1", history_len=3, hold_token="tok123",
             user_input="hi", store_turn=True)
    d.update(kw)
    hooks.hold_output_for_approval(eid, **d)
    return hooks.peek_held_output(eid)


def test_hold_and_pop_one_shot():
    _hold()
    h1 = hooks.pop_held_output("e1")
    assert h1["held_text"] == "held text here"
    assert hooks.pop_held_output("e1") is None  # consumed


def test_hold_expired_flagged_not_dropped_silently():
    _hold()
    with patch.object(hooks, "HELD_OUTPUT_TTL_S", 0.0):
        h = hooks.pop_held_output("e1")
    assert h["expired"] is True


def test_sweep_expired():
    _hold("old")
    _hold("fresh")
    with patch.object(hooks, "HELD_OUTPUT_TTL_S", 0.0):
        expired = hooks.sweep_held_outputs()
    assert {h["held_text"] for h in expired} >= {"held text here"}
    assert hooks.peek_held_output("fresh") is None  # swept too under 0 TTL


def test_push_turn_ask_notice_no_crash():
    # Notice bus may be absent in unit context; must never raise.
    hooks.push_turn_ask_notice("u1", "abc123", "turn flagged")


# ── completion: commit vs purge ──────────────────────────────────────────────

class _Owner:
    import threading as _threading

    def __init__(self):
        self._history = [{"role": "user", "content": "hi", "_hold_token": "tok123"}]
        self._history_lock = self._threading.Lock()
        self.stored = []

    def _commit_held_output(self, held):
        token = held.get("hold_token")
        with self._history_lock:
            idx = None
            for i, m in enumerate(self._history):
                if m.get("_hold_token") == token:
                    idx = i + 1
                    break
            if idx is None:
                raise ValueError("anchor missing")
            self._history.insert(idx, {"role": "assistant", "content": held["held_text"]})

    def _purge_held_output(self, held, marker):
        token = held.get("hold_token")
        with self._history_lock:
            idx = None
            for i, m in enumerate(self._history):
                if m.get("_hold_token") == token:
                    idx = i + 1
                    break
            if idx is None:
                idx = len(self._history)
            self._history.insert(idx, {"role": "assistant", "content": marker})

    def _store_async(self, user_input, response_text):
        self.stored.append((user_input, response_text))


def _complete(owner, approved, held):
    return hooks._complete_held_output(owner, "u1", "e1", approved, held)


def test_approve_commits_when_all_checks_pass():
    owner = _Owner()
    held = _hold(text="the answer")
    msg = _complete(owner, True, held)
    assert owner._history[1]["content"] == "the answer"
    assert "record" in msg


def test_deny_purges_with_marker():
    owner = _Owner()
    held = _hold(text="the answer")
    msg = _complete(owner, False, held)
    assert owner._history[1]["content"].startswith("[withheld")
    assert owner.stored == []


def test_expired_hold_cannot_be_approved():
    owner = _Owner()
    held = _hold(text="the answer")
    held["expired"] = True
    _complete(owner, True, held)
    assert owner._history[1]["content"].startswith("[withheld")


def test_tampered_text_purges_even_on_approve():
    owner = _Owner()
    held = _hold(text="the answer")
    held["held_text"] = "different text"  # hash no longer matches
    _complete(owner, True, held)
    assert owner._history[1]["content"].startswith("[withheld")


def test_admin_may_approve_another_users_hold():
    # Oversight model: an admin approving another user's held turn commits
    # it (the can_approve gate already blocks non-admin approvers).
    owner = _Owner()
    held = _hold(text="the answer", user_id="someone-else")
    msg = hooks._complete_held_output(owner, "admin-u1", "e1", True, held)
    assert owner._history[1]["content"] == "the answer"
    assert "record" in msg


def test_completion_never_raises_without_owner():
    held = _hold(text="the answer")
    msg = hooks._complete_held_output(None, "u1", "e1", True, held)
    assert isinstance(msg, str)


def test_expired_hold_tombstoned_for_late_approval():
    _hold("gone")
    with patch.object(hooks, "HELD_OUTPUT_TTL_S", 0.0):
        hooks.sweep_held_outputs()
    assert hooks.is_expired_hold("gone") is True
    assert hooks.pop_held_output("gone") is None  # already swept


def test_pop_expired_tombstones_too():
    _hold("late")
    with patch.object(hooks, "HELD_OUTPUT_TTL_S", 0.0):
        h = hooks.pop_held_output("late")
    assert h["expired"] is True
    assert hooks.is_expired_hold("late") is True


def test_commit_lands_after_token_anchor_despite_growth():
    owner = _Owner()
    # Simulate a newer turn arriving while the hold was pending.
    owner._history.append({"role": "user", "content": "next question"})
    owner._history.append({"role": "assistant", "content": "next answer"})
    held = _hold(text="the answer")
    hooks._complete_held_output(owner, "u1", "e1", True, held)
    # Inserted right after the ORIGINAL user message (index 1), not at end.
    assert owner._history[1] == {"role": "assistant", "content": "the answer"}
    assert owner._history[0]["content"] == "hi"


def test_resolve_uses_held_users_ledger(monkeypatch):
    # An admin approving another user's held turn resolves the escalation
    # against the HELD USER's ledger (where it was opened), not the
    # approver's.
    import cognition.conscience.hooks as hooks_mod
    _hold("abc123", text="hello", user_id="user-a")
    resolved_for = {}

    class FakeCore:
        def __init__(self, uid):
            self.uid = uid
        def resolve_escalation(self, eid, approved, note=""):
            resolved_for["uid"] = self.uid
            resolved_for["eid"] = eid
            return True

    monkeypatch.setattr("cognition.conscience.conscience_for", lambda uid=None: FakeCore(uid))
    monkeypatch.setattr("system.userspace.can_approve", lambda uid: True)

    owner = _Owner()
    msg = hooks_mod.resolve_ccc_approval("approve ccc-abc123", user_id="admin-b", owner=owner)
    assert resolved_for["uid"] == "user-a"
    assert resolved_for["eid"] == "abc123"
    assert "record" in msg


def test_commit_fails_closed_when_anchor_trimmed():
    owner = _Owner()
    owner._history = [{"role": "user", "content": "someone else"}]  # anchor gone
    held = _hold(text="the answer")
    msg = hooks._complete_held_output(owner, "u1", "e1", True, held)
    assert "moved on" in msg
    assert all(m["content"] != "the answer" for m in owner._history)
