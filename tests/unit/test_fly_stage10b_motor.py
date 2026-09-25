"""Phase 10B — DN → motor primitives → VNC coordination → actuator backends.

Covers the QA contract:
  - primitive emission from a scored action record (bounded, deterministic)
  - conscience invariant: vetoed / missing winner → no primitives
  - GF interrupt cancels in-flight primitives within the same turn
  - arbitration of competing primitives (priority; ties keep the earlier)
  - cooldowns, durations, in-flight cap
  - actuator clamping + per-actuator rate limits
  - shadow computes but applies nothing; off is a no-op; live applies
  - backend switching (vrm packet vs null vs physical stub)
  - on_scored hook feeds the latest record into drive()
  - per-turn cost < 1 ms
"""
from __future__ import annotations

import time

import pytest

from cognition.fly_behavior import motor_primitives as mp
from cognition.fly_behavior import vnc_coordinator as vnc
from cognition.fly_behavior import body
from cognition.fly_behavior import dn_body


@pytest.fixture
def uid():
    u = f"phase10b-{time.time_ns()}"
    yield u
    vnc.clear(u)
    body.clear(u)


@pytest.fixture
def shadow_mode(monkeypatch):
    monkeypatch.setenv("AIKO_FLY_BODY_MODE", "shadow")
    yield
    monkeypatch.delenv("AIKO_FLY_BODY_MODE", raising=False)


@pytest.fixture
def live_mode(monkeypatch):
    monkeypatch.setenv("AIKO_FLY_BODY_MODE", "live")
    yield
    monkeypatch.delenv("AIKO_FLY_BODY_MODE", raising=False)


def _neural(uid, *, valence=0.5, vigor=1.2, drive=0.8, urgency=0.0,
            interrupt=False):
    from cognition.neural_state import get_neural_state
    st = get_neural_state(uid)
    st.publish_mb(valence, source="test")
    st.publish_dn(arousal=drive, rate_mult=vigor, source="test")
    st.publish_gf(urgency, interrupt, source="test")
    return st


def _record(kind="reply", winner="c1", veto=False):
    return {
        "winner": winner,
        "mode": "live",
        "applied": True,
        "candidates": {
            "c1": {"kind": kind, "label": "reply", "llm_prior": 0.8,
                   "fly": 0.7, "final": 0.78, "veto": veto,
                   "votes": {"mb": 0.5, "cx": 0.3, "dn": 0.4, "gf": 0.0}},
        },
    }


# ── primitive emission ──────────────────────────────────────────────

def test_emit_reply_primitives(uid, shadow_mode):
    _neural(uid)
    prims = mp.emit_primitives(_record("reply"), user_id=uid)
    names = sorted(p.name for p in prims)
    assert names == ["expression", "gaze", "pose", "prosody"]
    for p in prims:
        assert p.actuator in mp.ACTUATORS
        d = p.as_dict()
        assert d["name"] == p.name
    # Deterministic: same inputs → identical output.
    again = mp.emit_primitives(_record("reply"), user_id=uid)
    assert [p.as_dict() for p in again] == [p.as_dict() for p in prims]


def test_emit_tool_primitives(uid, shadow_mode):
    _neural(uid)
    prims = mp.emit_primitives(_record("tool"), user_id=uid)
    names = sorted(p.name for p in prims)
    assert names == ["gaze", "gesture", "pose", "vigor"]
    gesture = next(p for p in prims if p.name == "gesture")
    assert 0.0 <= gesture.params["intensity"] <= 1.0


def test_emit_params_bounded(uid, shadow_mode):
    _neural(uid, valence=-0.9, vigor=1.5, drive=1.0, urgency=1.0)
    for p in mp.emit_primitives(_record("reply"), user_id=uid):
        for k, v in p.params.items():
            if isinstance(v, (int, float)):
                assert -2.0 <= v <= 2.0, (p.name, k, v)


def test_vetoed_winner_emits_nothing(uid, shadow_mode):
    _neural(uid)
    assert mp.emit_primitives(_record("reply", veto=True), user_id=uid) == []


def test_no_winner_emits_nothing(uid, shadow_mode):
    _neural(uid)
    rec = _record("reply", winner=None)
    assert mp.emit_primitives(rec, user_id=uid) == []
    assert mp.emit_primitives(None, user_id=uid) == []
    assert mp.emit_primitives("garbage", user_id=uid) == []


def test_off_mode_emits_nothing(uid, monkeypatch):
    monkeypatch.setenv("AIKO_FLY_BODY_MODE", "off")
    _neural(uid)
    assert mp.emit_primitives(_record("reply"), user_id=uid) == []
    assert mp.body_mode() == "off"


def test_gf_interrupt_blocks_emission(uid, shadow_mode, monkeypatch):
    monkeypatch.setenv("MEMORY_FLYGF_MODE", "live")
    _neural(uid, interrupt=True)
    assert mp.emit_primitives(_record("reply"), user_id=uid) == []


# ── VNC coordination ────────────────────────────────────────────────

def _prim(name, actuator, priority=10, duration=2, cooldown=0, params=None):
    return mp.MotorPrimitive(
        name=name, actuator=actuator, priority=priority,
        duration_turns=duration, cooldown_turns=cooldown,
        params=params or {})


def test_arbitration_higher_priority_wins(uid):
    r1 = vnc.coordinate([_prim("expression", "vrm.expression", priority=20,
                               duration=5)],
                        user_id=uid, tick=1)
    assert [a["name"] for a in r1["active"]] == ["expression"]
    # Lower priority rival on the same actuator is dropped; tie keeps the first.
    r2 = vnc.coordinate([_prim("expression", "vrm.expression", priority=10)],
                        user_id=uid, tick=2)
    assert r2["dropped"] and r2["dropped"][0]["reason"] == "arbitration"
    assert [a["name"] for a in r2["active"]] == ["expression"]
    # Higher priority supersedes.
    r3 = vnc.coordinate([_prim("expression", "vrm.expression", priority=30)],
                        user_id=uid, tick=3)
    assert any(d["reason"] == "superseded" for d in r3["dropped"])
    assert r3["started"][0]["priority"] == 30


def test_cooldown_blocks_refire(uid):
    p = _prim("gesture", "vrm.gesture", priority=15, duration=1, cooldown=2)
    vnc.coordinate([p], user_id=uid, tick=1)
    r = vnc.coordinate([p], user_id=uid, tick=2)
    assert r["dropped"] and r["dropped"][0]["reason"] == "cooldown"
    # After cooldown expiry it can fire again.
    r2 = vnc.coordinate([p], user_id=uid, tick=5)
    assert r2["started"], r2["dropped"]


def test_duration_expiry(uid):
    p = _prim("gaze", "vrm.gaze", priority=25, duration=2, cooldown=0)
    vnc.coordinate([p], user_id=uid, tick=1)
    assert len(vnc.in_flight(uid)) == 1
    # duration=2: alive through tick 2 (remaining 1), expired at tick 3.
    vnc.coordinate([], user_id=uid, tick=2)
    assert len(vnc.in_flight(uid)) == 1
    vnc.coordinate([], user_id=uid, tick=3)
    assert vnc.in_flight(uid) == []


def test_interrupt_cancels_in_flight(uid):
    vnc.coordinate([_prim("expression", "vrm.expression", priority=20,
                           duration=5)],
                   user_id=uid, tick=1)
    assert len(vnc.in_flight(uid)) == 1
    r = vnc.coordinate([], user_id=uid, tick=2, interrupt=True)
    assert r["cancelled"] and r["cancelled"][0]["name"] == "expression"
    assert vnc.in_flight(uid) == []
    # Nothing new is admitted while interrupted.
    r2 = vnc.coordinate([_prim("gaze", "vrm.gaze", priority=25)],
                        user_id=uid, tick=3, interrupt=True)
    assert r2["active"] == [] and r2["started"] == []


def test_in_flight_cap(uid):
    prims = [_prim(f"p{i}", f"act{i}", priority=i) for i in range(10)]
    r = vnc.coordinate(prims, user_id=uid, tick=1)
    assert len(r["active"]) <= 8
    assert any(d["reason"] == "capacity" for d in r["dropped"])


def test_cancel_api(uid):
    vnc.coordinate([_prim("pose", "vrm.pose", duration=5)], user_id=uid, tick=1)
    res = vnc.cancel(uid)
    assert res["n"] == 1
    assert vnc.in_flight(uid) == []


# ── body layer: translate / rate limits / modes ─────────────────────

def test_translate_clamps(uid):
    cmds = body.translate(
        [{"name": "expression", "actuator": "vrm.expression",
          "params": {"name": "happy", "intensity": 9.0}},
         {"name": "gaze", "actuator": "vrm.gaze",
          "params": {"target": "bogus", "speed": 99.0}}],
        user_id=uid)
    assert cmds["vrm.expression"]["intensity"] == 1.0
    assert cmds["vrm.gaze"]["target"] == "user"  # invalid → default
    assert cmds["vrm.gaze"]["speed"] == 1.4


def test_rate_limits(uid):
    first = body.translate(
        [{"name": "expression", "actuator": "vrm.expression",
          "params": {"name": "happy", "intensity": 0.5}}], user_id=uid)
    assert first["vrm.expression"]["intensity"] == 0.5
    # A jump of +0.5 is capped at the 0.25 per-step limit.
    second = body.translate(
        [{"name": "expression", "actuator": "vrm.expression",
          "params": {"name": "happy", "intensity": 1.0}}], user_id=uid)
    assert second["vrm.expression"]["intensity"] == pytest.approx(0.75)


def test_drive_shadow_applies_nothing(uid, shadow_mode):
    _neural(uid)
    res = body.drive(_record("reply"), user_id=uid, tick=1)
    assert res["mode"] == "shadow"
    assert res["applied"] is False
    assert res["primitives"], "shadow still computes primitives"
    assert res["backends"]["vrm"]["applied"] is False
    assert res["backends"]["tts"]["applied"] is False
    assert res["backends"]["agent"]["applied"] is False
    assert res["backends"]["null"]["logged"] is True


def test_drive_live_applies(uid, live_mode):
    _neural(uid)
    res = body.drive(_record("reply"), user_id=uid, tick=1)
    assert res["mode"] == "live"
    assert res["applied"] is True
    assert res["backends"]["vrm"]["applied"] is True
    vrm = res["backends"]["vrm"]
    assert vrm["expression_name"] in ("happy", "sad", "neutral")
    assert 0.0 <= vrm["expression_intensity"] <= 1.0


def test_drive_off_noop(uid, monkeypatch):
    monkeypatch.setenv("AIKO_FLY_BODY_MODE", "off")
    _neural(uid)
    res = body.drive(_record("reply"), user_id=uid, tick=1)
    assert res["applied"] is False and res["primitives"] == []


def test_drive_vetoed_record_no_actuators(uid, live_mode):
    _neural(uid)
    res = body.drive(_record("reply", veto=True), user_id=uid, tick=1)
    assert res["primitives"] == []
    assert res["actuators"] == {}


def test_physical_stub_never_selected(uid, live_mode):
    assert "physical" in body.backends()
    stub = body._BACKENDS["physical"].apply({"tts": {"rate": 1.2}},
                                            user_id=uid)
    assert stub["supported"] is False and stub["applied"] is False
    res = body.drive(_record("reply"), user_id=uid, tick=1)
    assert "physical" not in res["backends"]


def test_on_scored_hook_feeds_drive(uid, shadow_mode):
    _neural(uid)
    body.on_scored(_record("tool"), user_id=uid)
    assert body.latest_record(uid)["winner"] == "c1"
    res = body.drive(user_id=uid, tick=1)  # no record → uses latest
    names = sorted(p["name"] for p in res["primitives"])
    assert "gesture" in names  # tool-kind primitives


def test_tts_and_agent_accessors(uid, live_mode):
    _neural(uid)
    body.drive(_record("reply"), user_id=uid, tick=1)
    hints = body.tts_hints(uid)
    assert 0.85 <= hints["rate"] <= 1.15 and hints["applied"] is True
    cmd = body.agent_command(uid)
    assert 0.4 <= cmd["vigor"] <= 1.5 and cmd["applied"] is True


# ── dn_body integration ─────────────────────────────────────────────

def test_body_drive_packet_extended(uid, live_mode, monkeypatch):
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "live")
    _neural(uid)
    body.on_scored(_record("reply"), user_id=uid)
    pkt = dn_body.body_drive(user_id=uid)
    for key in ("body_mode", "body_applied", "primitives", "expression_name",
                "gaze_target", "gesture_name", "pose_name", "emphasis"):
        assert key in pkt, key
    assert pkt["body_mode"] == "live"
    assert pkt["body_applied"] is True
    assert pkt["expression_name"] in ("happy", "sad", "neutral")
    # Legacy keys unchanged in shape.
    assert 0.85 <= pkt["rate_mult"] <= 1.15


def test_body_drive_shadow_packet(uid, shadow_mode, monkeypatch):
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "live")
    _neural(uid)
    pkt = dn_body.body_drive(user_id=uid)
    assert pkt["body_mode"] == "shadow"
    assert pkt["body_applied"] is False


def test_agent_step_budget_body_live(uid, live_mode, monkeypatch):
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "live")
    _neural(uid, vigor=1.5, drive=1.0)
    body.on_scored(_record("tool"), user_id=uid)
    n = dn_body.agent_step_budget(user_id=uid, base=8)
    assert 1 <= n <= 12


def test_gf_interrupt_end_to_end(uid, live_mode, monkeypatch):
    monkeypatch.setenv("MEMORY_FLYGF_MODE", "live")
    monkeypatch.setenv("MEMORY_FLYDN_MODE", "live")
    _neural(uid)
    body.on_scored(_record("reply"), user_id=uid)
    # First a normal turn puts primitives in flight.
    r1 = body.drive(user_id=uid, tick=1)
    assert r1["primitives"]
    # Then the interrupt: same-turn cancel, nothing new admitted.
    _neural(uid, interrupt=True)
    r2 = body.drive(user_id=uid, tick=2)
    assert r2["cancelled"] is True
    assert r2["primitives"] == []
    assert vnc.in_flight(uid) == []


# ── performance ─────────────────────────────────────────────────────

def test_drive_cost_sub_ms(uid, shadow_mode):
    _neural(uid)
    rec = _record("reply")
    body.drive(rec, user_id=uid, tick=1)  # warm
    t0 = time.perf_counter()
    n = 50
    for i in range(n):
        body.drive(rec, user_id=uid, tick=10 + i)
    avg = (time.perf_counter() - t0) / n
    assert avg < 0.001, f"drive() avg {avg*1000:.3f} ms >= 1 ms"
