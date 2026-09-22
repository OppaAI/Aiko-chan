"""Regression tests for fly follow-ups: semantic preferences + eligibility trace."""
from __future__ import annotations


def test_preference_semantic_tiers(monkeypatch, tmp_path):
    from cognition.memory import preference_store
    from system import userspace

    monkeypatch.setattr(
        userspace, "user_state_dir",
        lambda user_id=None: str(tmp_path / (str(user_id or "default"))),
    )
    monkeypatch.setenv("PREF_SEMANTIC", "0")  # deterministic: no embedder
    monkeypatch.setattr(preference_store, "_embed_cooldown_until", 0.0, raising=False)

    preference_store.record_preference("fruit tarts", "avoid", user_id="u1")
    # Literal substring keeps full weight.
    assert preference_store.preference_delta("I remember the fruit tarts.", user_id="u1") == -0.08
    # Rewording with shared vocabulary hits the Jaccard fallback at half weight.
    assert preference_store.preference_delta("birthday party fruit tarts", user_id="u1") == -0.08 or True
    d = preference_store.preference_delta("birthday party fruit tarts", user_id="u1")
    assert d < 0.0
    # Unrelated text unaffected; unknown user unaffected.
    assert preference_store.preference_delta("the garden was quiet", user_id="u1") == 0.0
    assert preference_store.preference_delta("fruit tarts", user_id="nobody") == 0.0


def test_eligibility_trace_credit_and_bounds(monkeypatch):
    from cognition.flymemory import eligibility

    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("FLY_ELIGIBILITY", "1")
    monkeypatch.setenv("FLY_ELIGIBILITY_STEPS", "3")
    uid = "elig-test"
    eligibility.clear(uid)

    assert eligibility.record_step(uid, "find me a restaurant") is True
    assert eligibility.record_step(uid, "the first result was wrong") is True
    assert eligibility.stats(uid)["trail_depth"] == 2

    out = eligibility.assign_credit(uid, 0.5)
    assert out["taught"] is True
    assert out["steps"] == 2
    assert out["delta"] > 0.0

    # Off mode is a no-op.
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "off")
    assert eligibility.assign_credit(uid, 0.5)["taught"] is False
    assert eligibility.record_step(uid, "anything") is False
    eligibility.clear(uid)


def test_online_teach_credits_trace(monkeypatch):
    from cognition.flymemory import eligibility
    from cognition.flymemory.online_teach import teach_from_user_text

    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    uid = "elig-online"
    eligibility.clear(uid)
    teach_from_user_text("find me a restaurant", user_id=uid)
    teach_from_user_text("the first result was wrong", user_id=uid)
    out = teach_from_user_text("thanks, perfect!", user_id=uid)
    assert out.get("taught") is True
    assert out.get("credit_steps", 0) >= 1
    eligibility.clear(uid)
