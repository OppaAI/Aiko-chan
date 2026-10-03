"""Caller-level test: AikoThink._current_system_prompt_parts() carries exactly
one inner-voice block per turn (CodeRabbit nitpick on PR #222).

Unit tests on EdgeCognitiveState cover the block builders, but nothing
exercised the think.py insertion point — removing it would have left those
tests green. These pin the integration: one block on a casual turn, and still
exactly one when a deliberation turn appends metacognitive_context().
"""
from __future__ import annotations

import time

import pytest

import cognition.think as think_mod
from cognition.attention import EdgeCognitiveState
from cognition.think import AikoThink


@pytest.fixture()
def think_and_state(monkeypatch):
    think = object.__new__(AikoThink)
    think._persona = "PERSONA"
    think._last_chat_time = time.time()
    think._proactive_resting = False

    monkeypatch.setattr(think_mod, "_load_user_context", lambda: ("TestUser", ""))
    monkeypatch.setattr(think_mod, "current_user_id", lambda: "test-user")
    monkeypatch.setattr("system.schedule.list_schedule_records", lambda user_id=None: [])

    state = EdgeCognitiveState()
    monkeypatch.setattr(state, "persist", lambda: None)
    import cognition.attention as attention_mod
    monkeypatch.setattr(attention_mod, "for_identity", lambda identity: state)
    return think, state


def test_casual_turn_has_exactly_one_inner_voice_block(think_and_state):
    think, _ = think_and_state
    _, volatile = think._current_system_prompt_parts("hi")
    assert volatile.count("<inner_voice>") == 1
    assert volatile.count("</inner_voice>") == 1


def test_deliberation_turn_still_has_exactly_one_inner_voice_block(think_and_state):
    think, state = think_and_state
    _, volatile = think._current_system_prompt_parts("hi")
    # Simulate think.py's deliberation path appending metacognitive_context().
    metacognitive_block = state.metacognitive_context(
        "please explain in detail how the scheduler decides when to fire jobs", [])
    combined = volatile + "\n\n" + metacognitive_block
    assert combined.count("<inner_voice>") == 1
    assert combined.count("</inner_voice>") == 1
    # The confidence checkpoint itself still rides along.
    assert "<metacognitive_checkpoint>" in combined
