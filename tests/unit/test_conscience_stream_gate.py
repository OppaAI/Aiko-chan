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
             user_id="u1", history_len=3, user_input="hi", store_turn=True)
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
    def __init__(self):
        self._history = [{"role": "user", "content": "hi"}]
        self.stored = []

    def _commit_held_output(self, held):
        self._history.insert(held["history_len"],
                             {"role": "assistant", "content": held["held_text"]})

    def _purge_held_output(self, held, marker):
        self._history.insert(held["history_len"],
                             {"role": "assistant", "content": marker})

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


def test_wrong_user_purges_even_on_approve():
    owner = _Owner()
    held = _hold(text="the answer", user_id="someone-else")
    msg = hooks._complete_held_output(owner, "u1", "e1", True, held)
    assert owner._history[1]["content"].startswith("[withheld")
    assert "different user" in msg


def test_completion_never_raises_without_owner():
    held = _hold(text="the answer")
    msg = hooks._complete_held_output(None, "u1", "e1", True, held)
    assert isinstance(msg, str)
