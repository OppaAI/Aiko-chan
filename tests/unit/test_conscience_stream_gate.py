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
    hooks._EXPIRED_HOLD_IDS.clear()
    yield
    hooks._HELD_OUTPUTS.clear()
    hooks._EXPIRED_HOLD_IDS.clear()


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


@pytest.fixture
def output_ledger(monkeypatch):
    import sqlite3
    import cognition.conscience.core as core_mod
    import cognition.conscience.ledger as ledger_mod
    from cognition.conscience.schema import LEDGER_DDL
    ledger = ledger_mod.ConscienceLedger('u1')
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(LEDGER_DDL)
    ledger._conn = conn
    monkeypatch.setattr(ledger_mod, 'LEDGER_ENABLED', True)
    monkeypatch.setattr(core_mod, 'ledger_for', lambda uid: ledger)
    core = core_mod.ConscienceCircuitCore('u1')
    monkeypatch.setattr(core, '_note_self_decision', lambda *a: None)
    monkeypatch.setattr('cognition.conscience.conscience_for', lambda uid=None: core)
    monkeypatch.setattr('system.userspace.can_approve', lambda uid: True)
    for eid in ['abc123', 'def456', 'abc789']:
        ledger.record(_verdict(ESCALATE, escalation_id=eid), content='draft', surface='chat')
    yield ledger
    ledger.close()


def _output_states(ledger):
    ledger.flush()
    return dict(ledger._db().execute('SELECT id, hitl_state FROM conscience_ledger').fetchall())


def _group_hold():
    return _hold('abc123', text='the answer', related_escalation_ids=['def456', 'abc789', 'def456', 'abc123'])


@pytest.mark.parametrize('verb', ['approve', 'deny'])
def test_turn_decision_closes_all_questions_with_one_human_label(output_ledger, verb):
    _group_hold()
    owner = _Owner()
    reply = hooks.resolve_ccc_approval(f'{verb} ccc-abc123', user_id='admin', owner=owner)
    assert _output_states(output_ledger) == {
        'abc123': 'approved' if verb == 'approve' else 'denied',
        'def456': 'covered', 'abc789': 'covered',
    }
    assert output_ledger.pending() == []
    assert [row['id'] for row in output_ledger.harvest(only_gold=True)] == ['abc123']
    if verb == 'approve':
        assert owner._history[1]['content'] == 'the answer'
    else:
        assert owner._history[1]['content'].startswith('[withheld')
    assert 'record' in reply


def test_late_approval_times_out_all_rows_without_a_human_label(output_ledger):
    held = _group_hold()
    held['created_at'] -= hooks.HELD_OUTPUT_TTL_S + 1
    owner = _Owner()
    reply = hooks.resolve_ccc_approval('approve ccc-abc123', user_id='admin', owner=owner)
    assert 'expired' in reply
    assert set(_output_states(output_ledger).values()) == {'timeout'}
    assert output_ledger.harvest(only_gold=True) == []
    assert owner._history[1]['content'].startswith('[withheld')
    assert hooks.is_expired_hold('abc123')


def test_tampered_output_is_discarded_without_a_human_label(output_ledger):
    held = _group_hold()
    held['held_text'] = 'changed since evaluation'
    owner = _Owner()
    reply = hooks.resolve_ccc_approval('approve ccc-abc123', user_id='admin', owner=owner)
    assert 'changed' in reply
    assert set(_output_states(output_ledger).values()) == {'discarded'}
    assert output_ledger.harvest(only_gold=True) == []
    assert owner._history[1]['content'].startswith('[withheld')


def test_expiry_sweep_closes_the_group(output_ledger):
    held = _group_hold()
    held['created_at'] -= hooks.HELD_OUTPUT_TTL_S + 1
    expired = hooks.sweep_held_outputs(user_id='u1')
    assert len(expired) == 1
    assert expired[0]['expired'] is True
    assert set(_output_states(output_ledger).values()) == {'timeout'}


def test_user_scoped_sweep_leaves_other_users_holds_alone(monkeypatch):
    from types import SimpleNamespace
    expired_for = []
    monkeypatch.setattr('cognition.conscience.conscience_for', lambda uid=None: SimpleNamespace(
        expire_escalation=lambda eid, **kw: expired_for.append((uid, eid))))
    for eid, uid in [('aaa111', 'u1'), ('bbb222', 'u2')]:
        held = _hold(eid, user_id=uid)
        held['created_at'] -= hooks.HELD_OUTPUT_TTL_S + 1
    expired = hooks.sweep_held_outputs(user_id='u1')
    assert [h['user_id'] for h in expired] == ['u1']
    assert expired_for == [('u1', 'aaa111')]
    assert hooks.peek_held_output('bbb222') is not None


def test_capacity_eviction_closes_questions_and_blocks_late_approval(output_ledger, monkeypatch):
    monkeypatch.setattr(hooks, '_APPROVED_MAX', 1)
    _group_hold()
    _hold('fff999', text='new answer')
    assert set(_output_states(output_ledger).values()) == {'discarded'}
    reply = hooks.resolve_ccc_approval('approve ccc-abc123', user_id='admin', owner=_Owner())
    assert 'expired' in reply
    assert output_ledger.harvest(only_gold=True) == []


def test_competing_approvals_commit_once(output_ledger):
    from concurrent.futures import ThreadPoolExecutor
    _group_hold()
    owner = _Owner()
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda _: hooks.resolve_ccc_approval(
            'approve ccc-abc123', user_id='admin', owner=owner), range(2)))
    assert sum('in the record now' in reply for reply in replies) == 1
    assert [m['content'] for m in owner._history].count('the answer') == 1
    assert output_ledger.pending() == []


def test_cross_user_commit_uses_held_identity_and_restores_admin_context():
    from system.userspace import (
        set_current_user_id, reset_current_user_id, current_user_id,
        set_current_display_name, reset_current_display_name, current_display_name,
    )
    uid_token = set_current_user_id('u1')
    name_token = set_current_display_name('Alice')
    try:
        held = _hold()
    finally:
        reset_current_display_name(name_token)
        reset_current_user_id(uid_token)
    owner = _Owner()
    observed = []
    commit = owner._commit_held_output
    def record_identity(record):
        observed.append((current_user_id(), current_display_name()))
        commit(record)
    owner._commit_held_output = record_identity
    uid_token = set_current_user_id('admin')
    name_token = set_current_display_name('Admin')
    try:
        _complete(owner, True, held)
        assert observed == [('u1', 'Alice')]
        assert (current_user_id(), current_display_name()) == ('admin', 'Admin')
    finally:
        reset_current_display_name(name_token)
        reset_current_user_id(uid_token)


def test_real_history_purge_never_falls_back_after_anchor_is_trimmed():
    from cognition.think import AikoThink
    owner = _Owner()
    held = _hold()
    owner._history = [{'role': 'user', 'content': 'a newer or different user turn'}]
    before = list(owner._history)
    AikoThink._purge_held_output(owner, held, '[withheld]')
    assert owner._history == before


def _finalizing_owner(monkeypatch):
    from cognition.think import AikoThink
    owner = object.__new__(AikoThink)
    owner._review_response = lambda *a: {}
    owner._get_memorize = lambda: None
    owner._get_speak = lambda: None
    owner._client = None
    owner._emit_finalized_response = lambda *a, **kw: None
    return owner


def test_finalization_collects_all_sentence_question_ids(monkeypatch):
    owner = _finalizing_owner(monkeypatch)
    monkeypatch.setattr(hooks, 'gate_speak', lambda **kw: (
        None, _verdict(ESCALATE, escalation_id='abc123')))
    out = {}
    owner._finalize_response('hi', 'answer', _ask_out=out, ask_flags=[
        {'escalation_id': 'def456', 'sentence_index': 0},
        {'escalation_id': 'abc789', 'sentence_index': 1},
        {'escalation_id': 'def456', 'sentence_index': 2},
    ])
    assert out['escalation_id'] == 'abc123'
    assert out['related_escalation_ids'] == ['def456', 'abc789']


@pytest.mark.parametrize('outage', [False, True])
def test_final_refusal_or_outage_discards_sentence_questions(output_ledger, monkeypatch, outage):
    owner = _finalizing_owner(monkeypatch)
    def gate(**kw):
        if outage:
            raise RuntimeError('judge unavailable')
        return 'refusal', _verdict(REFUSE)
    monkeypatch.setattr(hooks, 'gate_speak', gate)
    out = {}
    response = owner._finalize_response('hi', 'answer', sentences_gated=True, _ask_out=out, ask_flags=[
        {'escalation_id': eid, 'sentence_index': i}
        for i, eid in enumerate(['abc123', 'def456', 'abc789'])
    ])
    assert not out
    assert response == ('[withheld: final conscience check unavailable]' if outage else 'refusal')
    assert set(_output_states(output_ledger).values()) == {'discarded'}
    assert output_ledger.harvest(only_gold=True) == []


def test_memory_failure_does_not_claim_approved_history_was_withheld():
    from cognition.think import AikoThink
    owner = _Owner()
    owner._commit_held_output = lambda held: AikoThink._commit_held_output(owner, held)
    def fail_store(*args):
        raise ValueError('memory unavailable')
    owner._store_async = fail_store
    held = _hold(text='the answer')
    reply = _complete(owner, True, held)
    assert owner._history[1]['content'] == 'the answer'
    assert 'conversation record' in reply
    assert "couldn't confirm the memory" in reply
    assert 'moved on' not in reply
