"""Unit tests for the fly central-complex compass layer (numpy-only, offline)."""
import numpy as np

from cognition.centralcomplex import FlyCompass, load_circuit
from cognition.centralcomplex.compass import DATA_PATH

FEATS = [0.2, 0.5, 0.3, 0.1, 0.0, 0.2, 0.0, 0.05]


def test_slice_loads_without_pickle():
    circ = load_circuit(DATA_PATH)
    assert circ["node_ids"].shape == (1282,)
    assert circ["edge_pre"].shape == (50515,)
    roles = set(np.unique(circ["node_roles"]).tolist())
    assert roles == {"INPUT", "RING", "PEN", "PEG", "EPG", "PF"}


def test_ring_topology_is_real():
    c = FlyCompass()
    assert c.n_epg == 50
    assert c.n_pfl_l == 25 and c.n_pfl_r == 25
    assert len(c._er5) == 21
    wedges = sorted(set(int(w) % 16 for w in
                        __import__("numpy").load(DATA_PATH)["node_wedges"][
                            __import__("numpy").load(DATA_PATH)["node_roles"] == "EPG"]))
    assert wedges == list(range(16))  # full 16-wedge ring from instance suffixes


def test_readouts_bounded_and_deterministic():
    c = FlyCompass()
    r = c.step(FEATS)
    assert 0.0 <= r["heading_deg"] < 360.0
    for k in ("sharpness", "decisiveness", "pfl_drive", "sleep_pressure"):
        assert r[k] >= 0.0
    assert r["sharpness"] <= 1.0 and r["decisiveness"] <= 1.0 and r["sleep_pressure"] <= 1.0
    assert FlyCompass().step(FEATS) == c.step(FEATS) or True  # states evolve; check fresh pair below
    assert FlyCompass().step(FEATS) == FlyCompass().step(FEATS)


def test_pen_rotation_and_sleep_accumulation():
    c = FlyCompass()
    h0 = c.step(FEATS)["heading_deg"]
    h1 = c.step(FEATS, pen_drive=1.0)["heading_deg"]
    assert abs(h1 - h0) > 1.0  # bump actually rotates
    c2 = FlyCompass()
    for _ in range(25):
        c2.step(FEATS, fatigue=1.0)
    assert c2.sleep_pressure > 0.3
    s_before = c2.sleep_pressure
    for _ in range(60):
        c2.step(FEATS, fatigue=0.0)
    assert c2.sleep_pressure < s_before  # decays without fatigue


def test_bad_features_rejected():
    import pytest
    with pytest.raises(ValueError):
        FlyCompass().step([0.0, 1.0])


def test_wiring_modes(monkeypatch):
    import cognition.attention as att
    st = att.EdgeCognitiveState("test-cx")
    monkeypatch.setattr(att, "MEMORY_FLYCX_MODE", "off")
    assert att.flycx_state_for_record(st, "hello?") is None
    e0 = st._energy
    monkeypatch.setattr(att, "MEMORY_FLYCX_MODE", "shadow")
    out = att.flycx_state_for_record(st, "hello?")
    assert set(out) == {"heading_deg", "sharpness", "decisiveness", "pfl_drive", "sleep_pressure"}
    assert st._energy == e0  # shadow never touches state
    monkeypatch.setattr(att, "MEMORY_FLYCX_MODE", "live")
    st._turn_latencies.extend([30.0] * 5)  # slow turns -> fatigue -> sleep -> nudge
    for _ in range(10):
        att.flycx_state_for_record(st, "still working on it")
    assert st._energy < e0  # drowsiness coupling applied


def test_routing_decisiveness_modes(monkeypatch):
    import cognition.attention as att
    monkeypatch.setattr(att, "MEMORY_FLYCX_MODE", "off")
    assert att.flycx_decisiveness_for_text("do the thing now") is None
    monkeypatch.setattr(att, "MEMORY_FLYCX_MODE", "shadow")
    d = att.flycx_decisiveness_for_text("do the thing now")
    assert d is not None and 0.0 <= d <= 1.0
    ok1, _, act1 = att.should_attempt(user_input="run the backup", intent_confidence=0.5)
    monkeypatch.setattr(att, "MEMORY_FLYCX_MODE", "live")
    ok2, _, act2 = att.should_attempt(user_input="run the backup", intent_confidence=0.5)
    assert isinstance(ok2, bool) and act2 in ("proceed", "degrade_chat", "defer", "clarify")


def test_orchestrator_cadence_modes(monkeypatch):
    # The orchestrator reads the mode per-call from the environment.
    from agentic.needle_orchestrator import _flycx_cadence
    monkeypatch.setenv("MEMORY_FLYCX_MODE", "off")
    assert _flycx_cadence("do research") == "parallel"
    monkeypatch.setenv("MEMORY_FLYCX_MODE", "shadow")
    assert _flycx_cadence("do research") == "parallel"  # shadow never steers
    monkeypatch.setenv("MEMORY_FLYCX_MODE", "live")
    assert _flycx_cadence("do research") in ("parallel", "sequential")
    _flycx_cadence._cx.sleep_pressure = 0.9  # drowsy crew
    assert _flycx_cadence("do research") == "sequential"
    _flycx_cadence._cx.sleep_pressure = 0.0
