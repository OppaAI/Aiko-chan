"""Tests for Phase 7 — CX persistent temporal / navigation dynamics.

Covers the explicit update equations (determinism, boundedness, per-drive
decay rates), valence-modulated thresholds, drive-bias arbitration,
shadow-vs-live behavior, the CX→MB teaching gate, and a wired
apply_turn_priors integration run.
"""
from __future__ import annotations

import os

import cognition.centralcomplex.temporal as temporal
from cognition.centralcomplex.temporal import TemporalCX


def _fresh() -> TemporalCX:
    return TemporalCX()


# ── equation properties ───────────────────────────────────────────────────

def test_heading_update_is_deterministic():
    inputs = [
        {"cue_x": 0.8, "cue_y": 0.6, "fresh_urgency": 0.2, "novelty": 0.7,
         "engagement_in": 0.4, "rest_in": 0.1, "mb_valence": 0.3},
        {"cue_x": 0.0, "cue_y": 0.0, "fresh_urgency": 0.0, "novelty": 0.1,
         "engagement_in": 0.9, "rest_in": 0.5, "mb_valence": -0.4},
    ]
    a, b = _fresh(), _fresh()
    for inp in inputs:
        ra, rb = a.step(dict(inp)), b.step(dict(inp))
    assert ra == rb


def test_heading_decays_geometrically_without_cue():
    tcx = _fresh()
    tcx.step({"cue_x": 1.0, "cue_y": 0.0})
    h1x = tcx.heading[0]
    assert h1x == 0.5  # 0 * 0.92 + 1 * 0.5
    tcx.step({})
    assert tcx.heading[0] == h1x * 0.92
    assert tcx.heading[1] == 0.0


def test_heading_stays_bounded_under_sustained_cue():
    tcx = _fresh()
    for _ in range(50):
        tcx.step({"cue_x": 1.0, "cue_y": -1.0})
    hx, hy = tcx.heading
    assert -1.0 <= hx <= 1.0 and -1.0 <= hy <= 1.0
    # fixed point would be 0.5/0.08 = 6.25 → clipped to 1.0
    assert hx == 1.0 and hy == -1.0


def test_urgency_trace_spikes_then_decays():
    tcx = _fresh()
    tcx.step({"fresh_urgency": 1.0})
    assert tcx.urgency == 0.6  # 0 * 0.88 + 1 * 0.6
    u1 = tcx.urgency
    tcx.step({})
    assert tcx.urgency == u1 * 0.88
    for _ in range(100):
        tcx.step({})
    assert 0.0 <= tcx.urgency <= 1.0
    assert tcx.urgency < 1e-3  # decays away, never negative


def test_drives_have_their_own_decay_rates():
    tcx = _fresh()
    # Seed equal values so the decay rates — not the gains — are isolated.
    tcx.drives = {name: 1.0 for name in tcx.drives}
    tcx.step({})  # all inputs zero → pure decay
    d = tcx.drives
    # rest (0.95) > curiosity (0.90) > engagement (0.85)
    assert d["rest"] == 0.95
    assert d["curiosity"] == 0.90
    assert d["engagement"] == 0.85
    assert d["rest"] > d["curiosity"] > d["engagement"]


def test_valence_modulates_drive_thresholds():
    pos = _fresh()
    pos.step({"mb_valence": 1.0})
    neg = _fresh()
    neg.step({"mb_valence": -1.0})
    neu = _fresh()
    for name in ("curiosity", "engagement", "rest"):
        assert pos.threshold(name) < neu.threshold(name) < neg.threshold(name)
        assert 0.05 <= pos.threshold(name) <= 0.95
        assert 0.05 <= neg.threshold(name) <= 0.95


def test_step_with_empty_or_none_inputs_is_neutral():
    for inp in (None, {}, {"cue_x": "bogus"}):
        r = _fresh().step(inp)
        assert r["heading"] == [0.0, 0.0]
        assert r["urgency"] == 0.0
        assert all(v == 0.0 for v in r["drives"].values())


# ── drive bias arbitration ────────────────────────────────────────────────

def _charged() -> TemporalCX:
    tcx = _fresh()
    for _ in range(6):  # saturate drives toward their asymptotes
        tcx.step({"novelty": 1.0, "engagement_in": 1.0, "rest_in": 1.0,
                  "mb_valence": 0.0})
    return tcx


def test_drive_bias_bounded():
    tcx = _charged()
    for kind in ("reply", "tool", "route", ""):
        for energy in (-1.0, 0.0, 1.0):
            for novelty in (0.0, 0.5, 1.0):
                b = tcx.drive_bias(kind=kind, energy=energy, novelty=novelty)
                assert -1.0 <= b <= 1.0


def test_drive_bias_semantics():
    tcx = _charged()
    # engagement favors replying
    assert tcx.drive_bias(kind="reply") > tcx.drive_bias(kind="tool")
    # curiosity favors novel candidates, penalizes stale ones
    assert tcx.drive_bias(kind="tool", novelty=1.0) > \
        tcx.drive_bias(kind="tool", novelty=0.0)
    # rest damps high-energy candidates only
    assert tcx.drive_bias(kind="tool", energy=1.0) < \
        tcx.drive_bias(kind="tool", energy=-1.0)


def test_teach_gain_bounded_and_semantic():
    assert 0.5 <= _fresh().teach_gain() <= 1.25
    cur = _fresh()
    for _ in range(6):
        cur.step({"novelty": 1.0})
    assert cur.teach_gain() > 1.0  # curiosity up-regulates teaching
    rst = _fresh()
    for _ in range(6):
        rst.step({"rest_in": 1.0})
    assert rst.teach_gain() < 1.0  # rest down-regulates teaching
    assert rst.teach_gain() >= 0.5


# ── mode behavior ─────────────────────────────────────────────────────────

def _set_cx_mode(mode: str):
    old = temporal._MODE
    temporal._MODE = mode
    return old


def test_tick_off_mode_short_circuits():
    old = _set_cx_mode("off")
    try:
        out = temporal.tick_cx_temporal("hi", user_id="stage7-off")
        assert out == {"mode": "off", "ok": False}
    finally:
        temporal._MODE = old


def test_tick_never_raises_on_garbage():
    old = _set_cx_mode("shadow")
    try:
        out = temporal.tick_cx_temporal(
            None, user_id=None, fresh_urgency="bogus",
            mb_valence=float("nan"), novelty=-5.0,
        )
        assert isinstance(out, dict)
    finally:
        temporal._MODE = old


def _candidates():
    from cognition.fly_behavior.action_select import Candidate

    return [
        Candidate(id="a", kind="reply", label="A",
                  description="answer the question",
                  llm_prior=0.9, hints={"energy": "med"}),
        Candidate(id="b", kind="tool", label="B",
                  description="run a heavy external tool",
                  llm_prior=0.4,
                  hints={"energy": "high", "external": True, "novelty": 0.9}),
    ]


def test_shadow_mode_logs_drive_bias_without_steering():
    from cognition.fly_behavior import action_select as sel

    uid = "stage7-shadow"
    temporal.clear_temporal_cx(uid)
    old = _set_cx_mode("shadow")
    try:
        tcx = temporal.get_temporal_cx(uid)
        tcx.step({"novelty": 1.0, "engagement_in": 1.0, "mb_valence": 0.8})
        rec = sel.score_candidates(_candidates(), user_id=uid,
                                   context_text="hello")
        assert rec["winner"] == "a"  # shadow: LLM prior decides
        assert rec["cx_drive_live"] is False
        votes_b = rec["candidates"]["b"]["votes"]
        assert "cx_drive" in votes_b  # computed + logged
        assert votes_b["cx_drive"] != 0.0  # drives are genuinely active
    finally:
        temporal._MODE = old
        temporal.clear_temporal_cx(uid)


def test_live_mode_applies_only_a_bounded_nudge():
    from cognition.fly_behavior import action_select as sel

    uid = "stage7-live"
    temporal.clear_temporal_cx(uid)
    old = _set_cx_mode("shadow")
    try:
        tcx = temporal.get_temporal_cx(uid)
        tcx.step({"novelty": 1.0, "engagement_in": 1.0, "mb_valence": 0.8})
        rec_shadow = sel.score_candidates(_candidates(), user_id=uid,
                                          context_text="hello")
        temporal._MODE = "live"
        rec_live = sel.score_candidates(_candidates(), user_id=uid,
                                        context_text="hello")
        assert rec_live["cx_drive_live"] is True
        assert rec_live["winner"] == "a"  # bias never overrides the prior
        for cid in ("a", "b"):
            d = abs(rec_live["candidates"][cid]["final"]
                    - rec_shadow["candidates"][cid]["final"])
            # fly weight 0.25 × (drive_w 0.10 / 2 for the fly01 mapping)
            assert d <= 0.02, (cid, d)
    finally:
        temporal._MODE = old
        temporal.clear_temporal_cx(uid)


def test_live_mode_urgency_trace_warms_gf_vote():
    from cognition.fly_behavior import action_select as sel

    uid = "stage7-gftrace"
    temporal.clear_temporal_cx(uid)
    old = _set_cx_mode("live")
    try:
        temporal.get_temporal_cx(uid).step({"fresh_urgency": 1.0})
        u = sel._gf_urgency("a calm sentence", uid)
        assert u == temporal.get_urgency_trace(uid) == 0.6
        temporal._MODE = "shadow"
        assert sel._gf_urgency("a calm sentence", uid) == 0.0
    finally:
        temporal._MODE = old
        temporal.clear_temporal_cx(uid)


def test_clear_interrupt_resets_urgency_trace():
    from cognition.fly_behavior.gf_global import clear_interrupt

    uid = "stage7-clear"
    temporal.clear_temporal_cx(uid)
    try:
        temporal.get_temporal_cx(uid).step({"fresh_urgency": 1.0})
        assert temporal.get_urgency_trace(uid) > 0.0
        assert clear_interrupt(uid) is True
        assert temporal.get_urgency_trace(uid) == 0.0
    finally:
        temporal.clear_temporal_cx(uid)


# ── wired integration ─────────────────────────────────────────────────────

def test_apply_turn_priors_runs_cx_temporal_clean():
    from cognition.fly_behavior.turn import apply_turn_priors

    uid = "stage7-turn"
    temporal.clear_temporal_cx(uid)
    os.environ["MEMORY_FLYGF_MODE"] = "shadow"
    try:
        outs = [
            apply_turn_priors(t, user_id=uid)
            for t in ("hello there",
                      "STOP that right now",
                      "thanks, that worked")
        ]
        for o in outs:
            assert isinstance(o, dict)
            cx = o.get("cx_temporal")
            assert cx is not None and cx.get("ok") is True
            assert cx.get("mode") in ("shadow", "live")
            assert -1.0 <= cx["heading"][0] <= 1.0
            assert -1.0 <= cx["heading"][1] <= 1.0
            assert 0.0 <= cx["urgency"] <= 1.0
            assert all(0.0 <= v <= 1.0 for v in cx["drives"].values())
            assert 0.5 <= cx["teach_gain"] <= 1.25
        traces = [o["cx_temporal"]["urgency"] for o in outs]
        assert traces[1] > 0.0          # STOP spiked the trace
        assert traces[2] < traces[1]    # …then it decayed next turn
    finally:
        os.environ.pop("MEMORY_FLYGF_MODE", None)
        temporal.clear_temporal_cx(uid)
