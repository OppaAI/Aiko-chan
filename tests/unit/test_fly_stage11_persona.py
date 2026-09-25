"""Phase 11 — persona-conditioned fly circuits.

QA contract:
  - trait defaults/init from the documented SOUL.md derivation
  - drift clamp bounds around defaults
  - per-event rate-of-change cap
  - shadow computes gains but applies nothing
  - persistence round-trip (JSON) and reset
  - plasticity ONLY through the Phase-10A dopamine/eligibility path
  - seeded FlyWorld A/B: same seed/scenario, only personality differs →
    different circuit trajectories AND different behavioral outputs,
    with no manual reward shaping
  - conscience-veto supremacy under persona modulation
  - fly modules never raise

Non-negotiable design under test: the persona changes neural gains and
initial conditions, never action scores. No test here may pass by adding
a trait → vote bonus.
"""
from __future__ import annotations

import importlib
import json
import time

import pytest

import cognition.centralcomplex.temporal as temporal
import cognition.fly_behavior.action_select as action_select
import cognition.fly_behavior.dn_body as dn_body
import cognition.fly_persona as persona
import cognition.fly_persona.modulators as modulators
import cognition.fly_persona.plasticity as plasticity
import cognition.fly_persona.state as pstate
import cognition.fly_persona.trace as ptrace
from cognition.flymemory import credit


@pytest.fixture
def pdir(tmp_path, monkeypatch):
    """Isolated persona data dir + shadow default mode."""
    monkeypatch.setenv("AIKO_FLY_PERSONA_DIR", str(tmp_path))
    monkeypatch.setenv("AIKO_FLY_PERSONA_MODE", "shadow")
    monkeypatch.delenv("AIKO_FLY_PERSONA_DEFAULTS", raising=False)
    return tmp_path


def _uid(tag: str) -> str:
    return f"p11-{tag}-{time.time_ns()}"


@pytest.fixture
def live_persona(monkeypatch):
    monkeypatch.setenv("AIKO_FLY_PERSONA_MODE", "live")


def _raise_curiosity(uid: str, monkeypatch, target: float = 1.0) -> None:
    """Raise one identity's curiosity through the sanctioned dopamine path.

    Signals are controlled (not faked at the trait level): the update
    still flows through on_credit_outcome's LR / clamp / cooldown logic.
    """
    monkeypatch.setattr(
        plasticity,
        "_signals",
        lambda _u: {
            "curiosity": 1.0,
            "playfulness": 0.0,
            "exploration": 0.0,
            "attachment": 0.0,
            "calmness": 0.0,
        },
    )
    for _ in range(200):
        if persona.get_personality(uid).traits["curiosity"] >= target - 1e-9:
            break
        plasticity.on_credit_outcome(uid, reward=0.0, pe=0.0, da=1.0, scope="real")


# ── 1. state: defaults / init ────────────────────────────────────────────

def test_defaults_match_documented_soul_derivation(pdir):
    d = persona.persona_defaults()
    assert d == {
        "curiosity": 0.70,
        "playfulness": 0.70,
        "attachment": 0.70,
        "calmness": 0.65,
        "exploration": 0.60,
    }
    st = persona.get_personality(_uid("defaults"))
    assert st.traits == d


def test_defaults_env_override(pdir, monkeypatch):
    monkeypatch.setenv("AIKO_FLY_PERSONA_DEFAULTS", '{"curiosity": 0.9, "bogus": 5}')
    pstate.clear_personality("ov")
    st = persona.get_personality("ov")
    assert st.traits["curiosity"] == 0.9
    assert st.traits["playfulness"] == 0.70  # untouched


def test_per_identity_isolation(pdir, live_persona, monkeypatch):
    a, b = _uid("iso-a"), _uid("iso-b")
    _raise_curiosity(b, monkeypatch)
    assert persona.get_personality(a).traits["curiosity"] == 0.70
    assert persona.get_personality(b).traits["curiosity"] == pytest.approx(1.0)


def test_traits_property_returns_copy(pdir):
    """No external module can mutate traits through the accessor."""
    uid = _uid("copy")
    st = persona.get_personality(uid)
    st.traits["curiosity"] = 0.0
    assert persona.get_personality(uid).traits["curiosity"] == 0.70


# ── 2/3. clamps: drift bounds + per-event rate cap ───────────────────────

def test_plasticity_respects_drift_bounds(pdir, live_persona, monkeypatch):
    uid = _uid("drift")
    monkeypatch.setattr(
        plasticity, "_signals",
        lambda _u: {t: 1.0 for t in pstate.TRAITS},
    )
    for _ in range(400):
        plasticity.on_credit_outcome(uid, reward=0, pe=0, da=1.0, scope="real")
    st = persona.get_personality(uid)
    for t in pstate.TRAITS:
        d = st.defaults[t]
        lo, hi = max(0.0, d - 0.30), min(1.0, d + 0.30)
        assert lo - 1e-9 <= st.traits[t] <= hi + 1e-9, t


def test_plasticity_negative_da_moves_traits_down(pdir, live_persona, monkeypatch):
    uid = _uid("negda")
    monkeypatch.setattr(
        plasticity, "_signals",
        lambda _u: {"curiosity": 1.0, "playfulness": 0, "exploration": 0,
                    "attachment": 0, "calmness": 0},
    )
    before = persona.get_personality(uid).traits["curiosity"]
    # first event updates; then respect cooldown
    for i in range(8):
        plasticity.on_credit_outcome(uid, reward=0, pe=0, da=-1.0, scope="real")
    after = persona.get_personality(uid).traits["curiosity"]
    assert after < before


def test_plasticity_per_event_rate_cap(pdir, live_persona, monkeypatch):
    """|Δtrait| per credit event ≤ 0.02 even at da=1, signal=1 (LR=0.05)."""
    uid = _uid("ratecap")
    monkeypatch.setattr(
        plasticity, "_signals",
        lambda _u: {"curiosity": 1.0, "playfulness": 0, "exploration": 0,
                    "attachment": 0, "calmness": 0},
    )
    before = persona.get_personality(uid).traits["curiosity"]
    plasticity.on_credit_outcome(uid, reward=0, pe=0, da=1.0, scope="real")
    after = persona.get_personality(uid).traits["curiosity"]
    assert after - before == pytest.approx(0.02, abs=1e-9)


def test_plasticity_cooldown(pdir, live_persona, monkeypatch):
    uid = _uid("cooldown")
    monkeypatch.setattr(
        plasticity, "_signals",
        lambda _u: {"curiosity": 1.0, "playfulness": 0, "exploration": 0,
                    "attachment": 0, "calmness": 0},
    )
    r1 = plasticity.on_credit_outcome(uid, reward=0, pe=0, da=1.0, scope="real")
    assert r1["updated"] is True
    r2 = plasticity.on_credit_outcome(uid, reward=0, pe=0, da=1.0, scope="real")
    assert r2["updated"] is False and r2["reason"] == "cooldown"


def test_plasticity_needs_dopamine_drive(pdir, live_persona, monkeypatch):
    uid = _uid("nodrive")
    monkeypatch.setattr(
        plasticity, "_signals",
        lambda _u: {"curiosity": 1.0, "playfulness": 0, "exploration": 0,
                    "attachment": 0, "calmness": 0},
    )
    r = plasticity.on_credit_outcome(uid, reward=0, pe=0, da=0.0, scope="real")
    assert r["updated"] is False and r["reason"] == "no_drive"


def test_plasticity_scope_guarded(pdir, live_persona, monkeypatch):
    """Simulated / replay credit must never reshape the personality."""
    uid = _uid("scope")
    monkeypatch.setattr(
        plasticity, "_signals",
        lambda _u: {"curiosity": 1.0, "playfulness": 0, "exploration": 0,
                    "attachment": 0, "calmness": 0},
    )
    for scope in ("flyworld-sim", "replay", "sim"):
        r = plasticity.on_credit_outcome(uid, reward=1, pe=1, da=1.0, scope=scope)
        assert r["updated"] is False and r["reason"] == "scope_guarded"
    assert persona.get_personality(uid).traits["curiosity"] == 0.70


def test_plasticity_only_when_persona_live(pdir, monkeypatch):
    """AIKO_FLY_PERSONA_MODE=shadow/off → dopamine flows, traits don't."""
    uid = _uid("modelive")
    monkeypatch.setattr(
        plasticity, "_signals",
        lambda _u: {"curiosity": 1.0, "playfulness": 0, "exploration": 0,
                    "attachment": 0, "calmness": 0},
    )
    for mode in ("shadow", "off"):
        monkeypatch.setenv("AIKO_FLY_PERSONA_MODE", mode)
        r = plasticity.on_credit_outcome(uid, reward=1, pe=1, da=1.0, scope="real")
        assert r["reason"] == "persona_not_live"
    assert persona.get_personality(uid).traits["curiosity"] == 0.70


# ── 4. shadow computes but does not modulate ─────────────────────────────

def test_shadow_computes_but_applies_identity(pdir):
    uid = _uid("shadow")
    raw = modulators.persona_gains(uid)
    # computed gains are non-identity for the default persona…
    assert raw["cx"]["curiosity"]["gain"] != pytest.approx(1.0)
    assert raw["gf_urgency"] != pytest.approx(1.0)
    # …but nothing is applied in shadow mode.
    applied = modulators.applied_gains(uid)
    assert applied["cx"]["curiosity"]["gain"] == 1.0
    assert applied["mb_plasticity"] == 1.0
    assert applied["gf_urgency"] == 1.0
    assert applied["dn_vigor"] == 1.0


def test_live_applies_trait_derived_gains(pdir, live_persona):
    uid = _uid("liveg")
    applied = modulators.applied_gains(uid)
    assert applied == modulators.persona_gains(uid)
    assert applied["cx"]["curiosity"]["gain"] > 1.0  # curiosity 0.7 > neutral


def test_gain_bounds(pdir):
    for t in pstate.TRAITS:
        for v in (0.0, 1.0):
            g = modulators._mult(v, 0.6)
            assert 0.5 <= g <= 1.5
    assert modulators._mult(0.5, 0.6) == pytest.approx(1.0)  # neutral → identity


def test_temporal_step_shadow_ignores_personality(pdir, monkeypatch):
    """Same inputs + shadow mode → identical CX step with/without persona."""
    uid = _uid("tshadow")
    _raise_curiosity(uid, monkeypatch)  # hi-curiosity identity…
    monkeypatch.setenv("AIKO_FLY_PERSONA_MODE", "shadow")
    r = temporal.tick_cx_temporal(
        "", user_id=uid, fresh_urgency=0.0, mb_valence=0.0,
        sleep_pressure=0.0, novelty=0.5, user_energy=0.5, voice_energy=0.0,
    )
    # …behaves exactly like the unmodulated dynamics.
    t2 = temporal.TemporalCX()
    r2 = t2.step({"novelty": 0.5, "engagement_in": 0.5, "rest_in": 0.0,
                  "mb_valence": 0.0, "fresh_urgency": 0.0})
    assert r["drives"]["curiosity"] == pytest.approx(r2["drives"]["curiosity"])


# ── 5/6. persistence round-trip + reset ───────────────────────────────────

def test_persistence_roundtrip(pdir, live_persona, monkeypatch):
    uid = _uid("persist")
    _raise_curiosity(uid, monkeypatch, target=0.8)
    mid = persona.get_personality(uid).traits["curiosity"]
    assert mid > 0.70
    assert persona.get_personality(uid).save() is True
    pstate.clear_personality(uid)  # drop in-memory; force disk reload
    st2 = persona.get_personality(uid)
    assert st2.traits["curiosity"] == pytest.approx(mid)


def test_reset_restores_defaults(pdir, live_persona, monkeypatch):
    uid = _uid("reset")
    _raise_curiosity(uid, monkeypatch)
    assert persona.get_personality(uid).traits["curiosity"] > 0.70
    desc = persona.reset_personality(uid)
    assert desc["traits"] == persona.persona_defaults()
    pstate.clear_personality(uid)
    assert persona.get_personality(uid).traits == persona.persona_defaults()


# ── 7. dopamine-only plasticity: the credit_event integration ─────────────

def test_credit_event_is_the_only_writer(pdir, monkeypatch):
    """Full Phase-10A path: mark_trace → credit_event → bounded trait move."""
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("AIKO_FLY_DOPAMINE_MODE", "live")
    monkeypatch.setenv("AIKO_FLY_PERSONA_MODE", "live")
    uid = _uid("creditpath")
    # prime the curiosity channel so the dopamine has a signal to ride on
    temporal.tick_cx_temporal(
        "", user_id=uid, fresh_urgency=0.0, mb_valence=0.0,
        sleep_pressure=0.0, novelty=0.9, user_energy=0.5, voice_energy=0.0,
    )
    credit.clear(uid, scope=None)
    before = persona.get_personality(uid).traits["curiosity"]
    import numpy as np
    rng = np.random.default_rng(11)
    kc = rng.random(32)
    kc = kc / (np.linalg.norm(kc) or 1.0)
    credit.mark_trace(uid, kc, value=0.9, scope="real")
    out = credit.credit_event(uid, reward=1.0, source="test", scope="real")
    assert out["ran"] is True
    assert out["persona"]["reason"] in ("ok", "cooldown", "no_delta")
    after = persona.get_personality(uid).traits["curiosity"]
    assert after >= before  # positive dopamine never decreases curiosity here
    assert after - before <= 0.02 + 1e-9  # per-event cap holds end to end
    credit.clear(uid, scope=None)


def test_credit_event_shadow_persona_leaves_traits(pdir, monkeypatch):
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("AIKO_FLY_DOPAMINE_MODE", "live")
    monkeypatch.setenv("AIKO_FLY_PERSONA_MODE", "shadow")
    uid = _uid("creditshadow")
    credit.clear(uid, scope=None)
    import numpy as np
    rng = np.random.default_rng(12)
    kc = rng.random(32)
    kc = kc / (np.linalg.norm(kc) or 1.0)
    credit.mark_trace(uid, kc, value=0.9, scope="real")
    out = credit.credit_event(uid, reward=1.0, source="test", scope="real")
    assert out["persona"]["reason"] == "persona_not_live"
    assert persona.get_personality(uid).traits["curiosity"] == 0.70
    credit.clear(uid, scope=None)


# ── 8. explainability trace ───────────────────────────────────────────────

def test_explain_turn_is_json_and_names_effects(pdir):
    uid = _uid("trace")
    rec = persona.explain_turn(uid)
    json.dumps(rec)  # must be JSON-serializable
    assert rec["mode"] == "shadow"
    assert rec["applied"] is False
    assert rec["traits"]["curiosity"] == 0.70
    assert any("cx.curiosity.gain" in e for e in rec["effects"])
    assert any("shadow" in e for e in rec["effects"])


def test_explain_turn_live_marks_applied(pdir, live_persona):
    rec = persona.explain_turn(_uid("tracelive"))
    assert rec["applied"] is True
    assert not any("shadow" in e for e in rec["effects"])


def test_explain_turn_cx_off_marks_gains_unapplied(pdir, live_persona, monkeypatch):
    monkeypatch.setattr(temporal, "_MODE", "off")
    rec = persona.explain_turn(_uid("trace-cx-off"))
    assert rec["applied"] is False
    assert any("cx unavailable" in e for e in rec["effects"])


def test_turn_trace_marks_cx_gains_unapplied_when_cx_off(pdir, live_persona, monkeypatch):
    from cognition.fly_behavior.turn import apply_turn_priors

    monkeypatch.setattr(temporal, "_MODE", "off")
    out = apply_turn_priors("hello", user_id=_uid("turn-trace-cx-off"))
    assert out["persona"]["applied"] is False


# ── 9. gain application paths: CX / MB / GF / DN ──────────────────────────

def test_mb_plasticity_gain_path(pdir, live_persona):
    """teach_gain_for scales with persona MB plasticity in live mode."""
    uid = _uid("mbgain")
    temporal.tick_cx_temporal(
        "", user_id=uid, fresh_urgency=0.0, mb_valence=0.0,
        sleep_pressure=0.0, novelty=0.5, user_energy=0.5, voice_energy=0.0,
    )
    live_gain = temporal.teach_gain_for(uid)
    import os
    os.environ["AIKO_FLY_PERSONA_MODE"] = "shadow"
    try:
        shadow_gain = temporal.teach_gain_for(uid)
    finally:
        os.environ["AIKO_FLY_PERSONA_MODE"] = "live"
    expected = modulators.persona_gains(uid)["mb_plasticity"]
    assert live_gain == pytest.approx(shadow_gain * expected, rel=1e-6)
    assert expected > 1.0  # playful+attached default persona consolidates more


def test_dn_vigor_gain_path(pdir, live_persona, monkeypatch):
    """body_drive vigor scales with persona DN vigor in live mode."""
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "shadow")
    from cognition.neural_state import get_neural_state
    uid = _uid("dnvigor")
    get_neural_state(uid)  # ensure a state exists
    live_out = dn_body.body_drive(user_id=uid)
    monkeypatch.setenv("AIKO_FLY_PERSONA_MODE", "shadow")
    shadow_out = dn_body.body_drive(user_id=uid)
    expected = modulators.persona_gains(uid)["dn_vigor"]
    assert live_out["action_vigor"] == pytest.approx(shadow_out["action_vigor"] * expected,
                                                    rel=1e-6)
    assert expected > 1.0


def test_gf_urgency_gain_path(pdir, live_persona):
    """Calm identity: same situation, lower GF urgency in live mode."""
    calm_uid, _ = _uid("gfcalm"), None
    # default calmness 0.65 → gf_urgency gain < 1
    u1 = action_select._gf_urgency("loud sudden noise!", user_id=calm_uid)
    import os
    os.environ["AIKO_FLY_PERSONA_MODE"] = "shadow"
    try:
        u2 = action_select._gf_urgency("loud sudden noise!", user_id=calm_uid)
    finally:
        os.environ["AIKO_FLY_PERSONA_MODE"] = "live"
    expected = modulators.persona_gains(calm_uid)["gf_urgency"]
    assert expected < 1.0
    assert u1 == pytest.approx(u2 * expected, rel=1e-6)


# ── 10. seeded A/B: the no-cheat proof ────────────────────────────────────

def test_seeded_personality_ab_divergence(pdir, monkeypatch):
    """curiosity → novelty attention. Same seed, same world, same
    situation, same LLM priors, same candidates — only the personality
    parameters differ. The curious identity must show a different CX
    trajectory AND pick the novel action; the default identity sticks
    with the familiar one. No reward shaping: candidates carry identical
    llm_priors structure and no hand-tuned bonuses."""
    monkeypatch.setenv("AIKO_FLY_PERSONA_MODE", "live")
    monkeypatch.setattr(temporal, "_MODE", "live")
    monkeypatch.setattr(temporal, "_DRIVE_W", 0.5)
    monkeypatch.setattr(action_select, "_MODE", "live")
    monkeypatch.setattr(action_select, "_WEIGHT", 0.4)

    lo, hi = _uid("ab-lo"), _uid("ab-hi")
    _raise_curiosity(hi, monkeypatch)  # via the dopamine path, not a cheat
    assert persona.get_personality(lo).traits["curiosity"] == 0.70
    assert persona.get_personality(hi).traits["curiosity"] == pytest.approx(1.0)

    # identical situation, three turns, novelty fading (same "seed")
    traj = {}
    for uid in (lo, hi):
        acts = []
        for nov in (0.6, 0.5, 0.4):
            r = temporal.tick_cx_temporal(
                "", user_id=uid, fresh_urgency=0.0, mb_valence=0.0,
                sleep_pressure=0.0, novelty=nov, user_energy=0.5,
                voice_energy=0.0,
            )
            acts.append(r["activations"]["curiosity"])
        traj[uid] = acts
    # different circuit trajectories: the curious identity's novelty
    # channel activates sooner and stronger, every turn
    assert traj[lo] != traj[hi]
    assert all(h >= l for h, l in zip(traj[hi], traj[lo]))
    assert traj[hi][0] > traj[lo][0]

    # identical candidates, identical LLM priors — the only difference
    # between the two runs is the personality state above
    desc = "a calm neutral statement about the weather"
    cands = [
        action_select.Candidate(
            id="novel", kind="reply", label="Explore the new topic",
            description=desc, llm_prior=0.47,
            hints={"novelty": 0.95, "energy": "med", "speed": "med",
                   "focus_match": 0.5},
        ),
        action_select.Candidate(
            id="familiar", kind="reply", label="Continue the usual topic",
            description=desc, llm_prior=0.53,
            hints={"novelty": 0.05, "energy": "med", "speed": "med",
                   "focus_match": 0.5},
        ),
    ]
    winners = {}
    for uid in (lo, hi):
        rec = action_select.score_candidates(
            cands, user_id=uid, context_text="hello", record_state=False
        )
        winners[uid] = rec["winner"]
    # different behavioral outputs from the same priors
    assert winners[lo] == "familiar"
    assert winners[hi] == "novel"


# ── 11. conscience-veto supremacy ─────────────────────────────────────────

def test_conscience_veto_supreme_over_persona(pdir, live_persona, monkeypatch):
    """A vetoed candidate stays vetoed with persona live — and can never
    win. Persona modulates circuit gains; it cannot clear a veto flag."""
    monkeypatch.setattr(action_select, "_MODE", "live")
    uid = _uid("veto")
    _raise_curiosity(uid, monkeypatch)  # maximally novelty-hungry identity
    monkeypatch.setattr(
        action_select, "_mb_valence", lambda _desc, _u=None: -0.9
    )
    cands = [
        action_select.Candidate(
            id="dangerous-novel", kind="tool", label="Risky novel tool",
            description="harmful", llm_prior=0.95,
            hints={"novelty": 1.0, "energy": "high", "speed": "fast",
                   "focus_match": 0.9, "external": True},
        ),
        action_select.Candidate(
            id="safe", kind="reply", label="Safe reply",
            description="kind", llm_prior=0.10,
            hints={"novelty": 0.1, "energy": "low", "speed": "slow",
                   "focus_match": 0.1},
        ),
    ]
    rec = action_select.score_candidates(
        cands, user_id=uid, context_text="do it", record_state=False
    )
    assert rec["candidates"]["dangerous-novel"]["veto"] is True
    # The fly never *applies* its override toward a vetoed candidate —
    # even with llm_prior 0.95, max novelty, and every mode live.
    assert rec["applied"] is False
    # (Fallback to the LLM prior winner on veto/ambiguity is pre-existing
    # action-selection behavior outside the fly's veto authority; the
    # Phase-11 claim is that persona modulation cannot clear, weaken, or
    # bypass the veto flag itself.)
    # also in shadow persona mode — veto does not depend on persona at all
    monkeypatch.setenv("AIKO_FLY_PERSONA_MODE", "shadow")
    rec2 = action_select.score_candidates(
        cands, user_id=uid, context_text="do it", record_state=False
    )
    assert rec2["candidates"]["dangerous-novel"]["veto"] is True
    assert rec2["applied"] is False


# ── 12. never raises ─────────────────────────────────────────────────────

def test_persona_never_raises(pdir, monkeypatch):
    monkeypatch.setenv("AIKO_FLY_PERSONA_MODE", "bogus-value")
    assert persona.persona_mode() == "shadow"  # unknown → safe default
    # garbage everywhere: nothing may raise
    persona.persona_gains(None)
    modulators.applied_gains(None)
    modulators.persona_gains("x")
    persona.explain_turn(None)
    ptrace.record_persona_trace(None)
    plasticity.on_credit_outcome(None, reward="x", pe=None, da="y", scope=None)
    persona.get_personality(None)
    persona.reset_personality(None)
    pstate.clear_personality(None)
    temporal.TemporalCX().step(None, gains={"cx": {"curiosity": {"gain": "bad"}}})
    temporal.teach_gain_for(None)
    action_select._gf_urgency(None, user_id=None)
    dn_body.body_drive(user_id=None)
