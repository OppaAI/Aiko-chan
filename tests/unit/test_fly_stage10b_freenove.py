"""Phase 10B+ — Freenove Robot Dog FNK0062 backend.

Covers:
  - exact wire-protocol encoding (F/O/A/B/E/C/D/M/R + trailing '#' + '\\n')
  - primitive → wire mapping (locomotion/gesture/pose/expression/gaze/buzzer)
  - no-spam: unchanged channels are not re-transmitted
  - auto-stop: F#0#0#0#5# when a locomotion intent disappears
  - GF interrupt → hardware e-stop F#0#0#0#5# (sent once, before apply)
  - shadow mode computes wire strings but never opens a socket
  - unreachable dog → soft fallback, no raise, bounded time
  - AIKO_FLY_BODY_BACKEND selection (default null; invalid → null)

A fake TCP server stands in for the dog and records exact bytes.
"""
from __future__ import annotations

import socketserver
import threading
import time

import pytest

from cognition.fly_behavior import body
from cognition.fly_behavior import freenove_dog as fd
from cognition.fly_behavior import motor_primitives as mp
from cognition.fly_behavior import vnc_coordinator as vnc


# ── fake dog ────────────────────────────────────────────────────────

class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(2.0)
        try:
            while True:
                data = self.request.recv(4096)
                if not data:
                    break
                self.server.received.append(data)
        except Exception:
            pass


@pytest.fixture
def dog_server():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    srv.received = []
    t = threading.Thread(target=srv.serve_forever,
                         kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def uid():
    u = f"phase10b-freenove-{time.time_ns()}"
    yield u
    vnc.clear(u)
    body.clear(u)
    body._BACKENDS["freenove"].reset(u)
    fd.reset_link()


@pytest.fixture
def live_mode(monkeypatch):
    monkeypatch.setenv("AIKO_FLY_BODY_MODE", "live")
    yield
    monkeypatch.delenv("AIKO_FLY_BODY_MODE", raising=False)


@pytest.fixture
def freenove_backend(monkeypatch):
    monkeypatch.setenv("AIKO_FLY_BODY_BACKEND", "freenove")
    yield
    monkeypatch.delenv("AIKO_FLY_BODY_BACKEND", raising=False)


@pytest.fixture
def dog_env(monkeypatch, dog_server):
    port = dog_server.server_address[1]
    monkeypatch.setenv("FREENOVE_DOG_HOST", "127.0.0.1")
    monkeypatch.setenv("FREENOVE_DOG_PORT", str(port))
    monkeypatch.setenv("FREENOVE_DOG_TIMEOUT", "1.0")
    fd.reset_link()
    yield port
    fd.reset_link()


def _neural(uid, *, valence=0.5, vigor=1.0, drive=0.8, urgency=0.0,
            interrupt=False):
    from cognition.neural_state import get_neural_state
    st = get_neural_state(uid)
    st.publish_mb(valence, source="test")
    st.publish_dn(arousal=drive, rate_mult=vigor, source="test")
    st.publish_gf(urgency, interrupt, source="test")
    return st


def _record(kind="tool", winner="c1", veto=False, locomotion=None):
    cand = {"kind": kind, "label": "move", "llm_prior": 0.8,
            "fly": 0.7, "final": 0.78, "veto": veto,
            "votes": {"mb": 0.5, "cx": 0.3, "dn": 0.4, "gf": 0.0}}
    if locomotion is not None:
        cand["locomotion"] = dict(locomotion)
    return {"winner": winner, "mode": "live", "applied": True,
            "candidates": {winner: cand}}


def _bytes(srv):
    # Give the server thread a beat to recv.
    time.sleep(0.15)
    return b"".join(srv.received)


# ── protocol encoding ───────────────────────────────────────────────

def test_encode_exact_wire_strings():
    assert fd.encode("F", 0, 0, 0) == "F#0#0#0#\n"
    assert fd.encode("F", 10, -20, 0, 50) == "F#10#-20#0#50#\n"
    assert fd.move(10, -20, 0, 50) == "F#10#-20#0#50#\n"
    assert fd.STOP == "F#0#0#0#5#\n"
    assert fd.trick(6) == "O#6#\n"
    assert fd.trick(99) == "O#6#\n"   # clamped to 1..6
    assert fd.trick(0) == "O#1#\n"
    assert fd.posture(2) == "A#2#\n"
    assert fd.twist(10, 0, 0) == "E#10#0#0#\n"
    assert fd.led(1, 0, 255, 0) == "C#1#0#255#0#\n"
    assert fd.buzzer(880) == "D#880#\n"
    assert fd.auto_walk(True) == "M#1#\n"
    assert fd.move_speed(60) == "R#60#\n"
    # Sense-back query builders exist (future afferent path, unused).
    assert fd.query_ultrasonic() == "H#\n"
    assert fd.query_battery() == "I#\n"


# ── locomotion → wire ───────────────────────────────────────────────

def test_locomotion_emits_primitive(uid, live_mode, freenove_backend):
    _neural(uid)
    prims = mp.emit_primitives(
        _record(locomotion={"x": 10, "y": -20, "rot": 0, "speed": 50}),
        user_id=uid)
    loco = [p for p in prims if p.name == "locomotion"]
    assert len(loco) == 1
    assert loco[0].params == {"x": 10, "y": -20, "rot": 0, "speed": 50}
    assert loco[0].actuator == "dog.locomotion"


def test_no_locomotion_intent_no_primitive(uid, live_mode, freenove_backend):
    _neural(uid)
    prims = mp.emit_primitives(_record(), user_id=uid)
    assert [p for p in prims if p.name == "locomotion"] == []


def test_locomotion_wire_exact(uid, live_mode, freenove_backend, dog_env,
                               dog_server):
    _neural(uid)
    res = body.drive(
        _record(locomotion={"x": 10, "y": -20, "rot": 0, "speed": 50}),
        user_id=uid, tick=1)
    fr = res["backends"]["freenove"]
    assert fr["applied"] is True
    assert b"F#10#-20#0#50#\n" in _bytes(dog_server)


def test_no_spam_identical_turns(uid, live_mode, freenove_backend, dog_env,
                                 dog_server):
    _neural(uid)
    rec = _record(locomotion={"x": 10, "y": 0, "rot": 0, "speed": 50})
    body.drive(rec, user_id=uid, tick=1)
    body.drive(rec, user_id=uid, tick=2)
    data = _bytes(dog_server)
    assert data.count(b"F#10#0#0#50#\n") == 1, data


def test_auto_stop_when_intent_disappears(uid, live_mode, freenove_backend,
                                          dog_env, dog_server):
    _neural(uid)
    body.drive(_record(locomotion={"x": 10, "y": 0, "rot": 0, "speed": 50}),
               user_id=uid, tick=1)
    assert b"F#10#0#0#50#\n" in _bytes(dog_server)
    # Next turn: no locomotion intent → the dog must be stopped explicitly.
    body.drive(_record(), user_id=uid, tick=2)
    assert b"F#0#0#0#5#\n" in _bytes(dog_server)


# ── e-stop ──────────────────────────────────────────────────────────

def test_gf_interrupt_sends_estop_once(uid, live_mode, freenove_backend,
                                       dog_env, dog_server, monkeypatch):
    monkeypatch.setenv("MEMORY_FLYGF_MODE", "live")
    _neural(uid)
    body.drive(_record(locomotion={"x": 10, "y": 0, "rot": 0, "speed": 50}),
               user_id=uid, tick=1)
    assert b"F#10#0#0#50#\n" in _bytes(dog_server)
    # Interrupt: same-turn cancel → exactly one hardware e-stop.
    _neural(uid, interrupt=True)
    res = body.drive(user_id=uid, tick=2)
    assert res["cancelled"] is True
    fr = res["backends"]["freenove"]
    assert fr["estop"]["estop_sent"] is True
    data = _bytes(dog_server)
    assert data.count(b"F#0#0#0#5#\n") == 1, data


def test_failed_estop_preserves_move_for_apply_retry(uid, monkeypatch):
    be = body._BACKENDS["freenove"]
    moving = fd.move(10, 0, 0, 50)
    be._last_wire[body._key(uid)] = {"F": moving}

    class Link:
        available = True
        def estop(self):
            return False
        def send(self, wire):
            assert wire == fd.STOP
            return True

    monkeypatch.setattr(fd, "get_link", lambda: Link())
    assert be.estop(user_id=uid, live=True)["estop_sent"] is False
    assert be._last_wire[body._key(uid)]["F"] == moving
    assert be.apply({}, user_id=uid, live=True)["sent"] == [fd.STOP]


def test_dog_link_reconnect_backoff_and_availability(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(fd.time, "monotonic", lambda: now[0])
    calls = []

    class Socket:
        def __init__(self, broken=False):
            self.broken = broken
        def settimeout(self, timeout):
            pass
        def sendall(self, data):
            if self.broken:
                raise OSError("send failed")
        def close(self):
            pass

    def connect(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("connect failed")
        return Socket(broken=len(calls) == 2)

    monkeypatch.setattr(fd.socket, "create_connection", connect)
    link = fd.DogLink("127.0.0.1", 5000)
    monkeypatch.setattr(link, "_warn_once", lambda msg: None)
    assert link.available is False
    assert link.send(fd.STOP) is False
    assert link.send(fd.STOP) is False
    assert len(calls) == 1
    now[0] += fd._RECONNECT_BACKOFF_S
    assert link.send(fd.STOP) is False
    assert link.available is False
    assert link.send(fd.STOP) is False
    assert len(calls) == 2
    now[0] += fd._RECONNECT_BACKOFF_S
    assert link.send(fd.STOP) is True
    assert link.available is True
    assert link.send(fd.STOP) is True
    assert len(calls) == 3  # the open socket remains the fast path


# ── mapping spot checks (unit-level, exact wires) ───────────────────

def _apply(cmds, uid, live=True):
    be = body._BACKENDS["freenove"]
    be.reset(uid)
    return be.apply(cmds, user_id=uid, live=live)


def test_gesture_maps_to_trick(uid):
    res = _apply({"vrm.gesture": {"name": "greet", "intensity": 0.8}},
                 uid, live=False)
    assert res["wire"] == ["O#1#\n"]
    res = _apply({"vrm.gesture": {"name": "emphasize", "intensity": 0.9}},
                 uid, live=False)
    assert res["wire"] == ["O#6#\n"]
    # "none" gesture → no trick sent.
    res = _apply({"vrm.gesture": {"name": "none", "intensity": 0.0}},
                 uid, live=False)
    assert res["wire"] == []


def test_expression_maps_to_led(uid):
    res = _apply({"vrm.expression": {"name": "happy", "intensity": 1.0}},
                 uid, live=False)
    assert res["wire"] == ["C#1#0#255#0#\n"]
    res = _apply({"vrm.expression": {"name": "sad", "intensity": 0.5}},
                 uid, live=False)
    assert res["wire"] == ["C#1#0#0#128#\n"]


def test_gaze_maps_to_lean_surrogate(uid):
    res = _apply({"vrm.gaze": {"target": "user", "speed": 1.0}},
                 uid, live=False)
    assert res["wire"] == ["E#10#0#0#\n"]


def test_pose_maps_to_posture(uid):
    res = _apply({"vrm.pose": {"name": "engaged", "intensity": 0.8}},
                 uid, live=False)
    assert res["wire"] == ["A#0#\n"]


def test_buzzer_edge_triggered(uid):
    be = body._BACKENDS["freenove"]
    be.reset(uid)
    r1 = be.apply({"tts": {"rate": 1.0, "volume": 1.0, "emphasis": 0.85}},
                  user_id=uid, live=False)
    assert r1["wire"] == ["D#880#\n"]
    # Still high → no repeat beep (edge-triggered).
    r2 = be.apply({"tts": {"rate": 1.0, "volume": 1.0, "emphasis": 0.9}},
                  user_id=uid, live=False)
    assert r2["wire"] == []
    # Drops low → re-arms.
    be.apply({"tts": {"rate": 1.0, "volume": 1.0, "emphasis": 0.2}},
             user_id=uid, live=False)
    r3 = be.apply({"tts": {"rate": 1.0, "volume": 1.0, "emphasis": 0.8}},
                  user_id=uid, live=False)
    assert r3["wire"] == ["D#880#\n"]


# ── shadow / fallback / selection ───────────────────────────────────

def test_shadow_computes_but_never_connects(uid, monkeypatch, dog_server):
    monkeypatch.setenv("AIKO_FLY_BODY_MODE", "shadow")
    monkeypatch.setenv("AIKO_FLY_BODY_BACKEND", "freenove")
    # Point at a port where nothing listens: any socket attempt would fail.
    monkeypatch.setenv("FREENOVE_DOG_HOST", "127.0.0.1")
    monkeypatch.setenv("FREENOVE_DOG_PORT", "9")
    fd.reset_link()
    _neural(uid)
    res = body.drive(
        _record(locomotion={"x": 10, "y": 0, "rot": 0, "speed": 50}),
        user_id=uid, tick=1)
    fr = res["backends"]["freenove"]
    assert fr["applied"] is False
    assert "F#10#0#0#50#\n" in fr["wire"]
    assert _bytes(dog_server) == b""


def _closed_port():
    """A localhost port nothing listens on (bind then close)."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_unreachable_dog_soft_fallback(uid, live_mode, freenove_backend,
                                       monkeypatch):
    # Localhost + closed port → connection refused, fast and deterministic.
    monkeypatch.setenv("FREENOVE_DOG_HOST", "127.0.0.1")
    monkeypatch.setenv("FREENOVE_DOG_PORT", str(_closed_port()))
    monkeypatch.setenv("FREENOVE_DOG_TIMEOUT", "1.0")
    fd.reset_link()
    _neural(uid)
    t0 = time.monotonic()
    res = body.drive(
        _record(locomotion={"x": 10, "y": 0, "rot": 0, "speed": 50}),
        user_id=uid, tick=1)
    dt = time.monotonic() - t0
    fr = res["backends"]["freenove"]
    assert fr["applied"] is False
    assert fr["reachable"] is False
    assert "F#10#0#0#50#\n" in fr["wire"]  # computed, not sent
    assert dt < 5.0, f"turn blocked too long: {dt:.2f}s"


def test_backend_selection_default_null(uid, live_mode, monkeypatch):
    monkeypatch.delenv("AIKO_FLY_BODY_BACKEND", raising=False)
    assert body.embodied_backend() == "null"
    _neural(uid)
    res = body.drive(_record(), user_id=uid, tick=1)
    assert "freenove" not in res["backends"]
    assert "vrm" not in res["backends"]
    assert res["backends"]["null"]["logged"] is True


def test_backend_selection_invalid_falls_back(uid, monkeypatch):
    monkeypatch.setenv("AIKO_FLY_BODY_BACKEND", "bogus")
    assert body.embodied_backend() == "null"


def test_backend_selection_vrm(uid, live_mode, monkeypatch):
    monkeypatch.setenv("AIKO_FLY_BODY_BACKEND", "vrm")
    assert body.embodied_backend() == "vrm"
    _neural(uid)
    res = body.drive(_record("reply"), user_id=uid, tick=1)
    assert res["backends"]["vrm"]["applied"] is True
    assert "freenove" not in res["backends"]


def test_freenove_in_backend_registry():
    assert "freenove" in body.backends()
    assert body._BACKENDS["freenove"].name == "freenove"
