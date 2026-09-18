"""Unit tests for the fly mushroom-body biological layer (numpy-only, offline)."""
import numpy as np
import pytest

from cognition.flymemory import FlyMB, load_circuit
from cognition.flymemory.circuit import DATA_PATH

FEATS = [0.5, -1.0, 0.3, 0.0, 1.0, -0.4, 0.2, 0.8]


def test_slice_loads_without_pickle():
    circ = load_circuit(DATA_PATH)
    assert circ["node_ids"].shape == (4673,)
    assert circ["edge_pre"].shape == (121706,)
    roles = set(np.unique(circ["node_roles"]).tolist())
    assert roles == {"INPUT", "KC", "MBON", "DAN", "APLDPM"}


def test_mbon_opponent_split_matches_biology():
    m = FlyMB()
    assert m.n_mbon == 97
    assert int((m.mbon_sign > 0).sum()) + int((m.mbon_sign < 0).sum()) + int(
        (m.mbon_sign == 0).sum()) == 97
    sign_of = dict(zip(m.mbon_types.tolist(), m.mbon_sign.tolist()))
    # Ground truth spot-checks (Aso et al.; PAM=reward, PPL1=punishment):
    assert sign_of["MBON01"] > 0   # gamma5beta'2a, appetitive
    assert sign_of["MBON14"] < 0   # alpha3, aversive
    assert sign_of["MBON11"] < 0   # gamma1pedc, aversive


def test_sparse_deterministic_encoding():
    m = FlyMB()
    kc = m.encode(FEATS)
    assert kc.shape == (m.n_kc,)
    frac = float((kc > 0).mean())
    assert 0.03 < frac < 0.08  # ~5% sparseness like the fly
    assert bool((FlyMB().encode(FEATS) == kc).all())
    with pytest.raises(ValueError):
        m.encode([0.0, 1.0])


def test_reward_shifts_readout_toward_approach():
    m = FlyMB()
    kc = m.encode(FEATS)
    b0 = m.valence_bias(FEATS)
    m.reinforce(kc, +1.0)
    assert m.valence_bias(FEATS) > b0
    m2 = FlyMB()
    kc2 = m2.encode(FEATS)
    b0b = m2.valence_bias(FEATS)
    m2.reinforce(kc2, -1.0)
    assert m2.valence_bias(FEATS) < b0b
    assert -1.0 <= m.valence_bias(FEATS) <= 1.0


def test_plastic_overlay_stays_bounded_and_base_fixed():
    m = FlyMB()
    base = m._kcm[2].copy()
    for _ in range(50):
        m.reinforce(m.encode(FEATS), +1.0)
    assert bool((m._kcm[2] == base).all())  # base weights immutable
    assert float(np.abs(m._plastic).max()) <= 0.5 + 1e-9


def _turn(**kw):
    from cognition.memory.grasp import GraspTurn
    base = dict(user="hello", assistant="hi there", tokens=10, emotion=0.6,
                importance=0.5, relevance=0.5, novelty=0.5, question=0.0,
                entity=0.2, recall_count=0, created_turn=1)
    base.update(kw)
    return GraspTurn(**base)


def test_wiring_off_by_default_leaves_score_untouched(monkeypatch):
    import cognition.memory.grasp as grasp
    monkeypatch.setattr(grasp, "MEMORY_FLYMB_MODE", "off")
    t = _turn()
    assert grasp.compute_score(t, 1) == grasp.compute_score(t, 1)
    assert grasp.flymb_bias_for_turn(t, 1) is None


def test_wiring_shadow_does_not_change_score(monkeypatch):
    import cognition.memory.grasp as grasp
    monkeypatch.setattr(grasp, "MEMORY_FLYMB_MODE", "off")
    t = _turn()
    base = grasp.compute_score(t, 1)
    monkeypatch.setattr(grasp, "MEMORY_FLYMB_MODE", "shadow")
    assert grasp.compute_score(t, 1) == base
    assert grasp.flymb_bias_for_turn(t, 1) is not None


def test_wiring_live_nudges_score_and_learns_once(monkeypatch):
    import cognition.memory.grasp as grasp
    monkeypatch.setattr(grasp, "MEMORY_FLYMB_MODE", "off")
    t = _turn()
    base = grasp.compute_score(t, 1)
    monkeypatch.setattr(grasp, "MEMORY_FLYMB_MODE", "live")
    monkeypatch.setattr(grasp, "MEMORY_FLYMB_W", 0.05)
    live = grasp.compute_score(t, 1)
    assert live != base
    assert abs(live - base) <= 0.05 + 1e-9
    buf = grasp.GraspBuffer(journal_enabled=False)
    buf.fill("I loved this", "glad to hear it")
    plastic = grasp._flymb()._plastic
    assert float(abs(plastic).sum()) > 0.0  # exactly one teaching event
