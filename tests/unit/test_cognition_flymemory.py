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


def test_promote_wiring_modes(monkeypatch):
    # promote.py reads the mode per-call from the environment (unlike grasp's
    # import-time constant), so drive it via setenv like production does.
    import cognition.consolidate.promote as promo
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "off")
    base = promo.score_journal_fragment("I love this wonderful result, thanks!")
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "shadow")
    assert promo.score_journal_fragment("I love this wonderful result, thanks!") == base
    # Naive (unteached) biases are arbitrary-signed; meaning comes from DAN
    # teaching, so teach this exact pattern once before comparing.
    from cognition.flymemory import FlyMB, text_features
    mb = promo._flymb()
    feats = text_features("I love this wonderful result, thanks!")
    for _ in range(5):
        mb.reinforce(mb.encode(feats), +1.0)
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    assert promo.score_journal_fragment("I love this wonderful result, thanks!") > base
    assert promo.score_journal_fragment("random filler words here ok") <= base + 0.11


def test_recall_rerank_keeps_all_hits_and_orders(monkeypatch):
    # episode.py also reads the mode per-call from the environment.
    import cognition.memory.episode as ep
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "shadow")
    hits = [
        {"id": 1, "_recall_score": 0.03, "trace": "the cat sat on the mat"},
        {"id": 2, "_recall_score": 0.02, "trace": "I love this wonderful birthday party, thanks!"},
    ]
    out = ep._flymb_rerank("wonderful birthday party love", hits)
    assert [h["id"] for h in out] == [1, 2]  # shadow: order untouched
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    out = ep._flymb_rerank("wonderful birthday party love", [dict(h) for h in hits])
    assert sorted(h["id"] for h in out) == [1, 2]  # never drops hits
    assert all("_flymb_overlap" in h for h in out)


def test_grasp_context_gist_line(monkeypatch):
    import cognition.memory.grasp as grasp
    monkeypatch.setattr(grasp, "MEMORY_FLYMB_MODE", "off")
    buf = grasp.GraspBuffer(journal_enabled=False)
    buf.fill("I love this wonderful result", "great!")
    assert "[fly valence" not in (buf.get_context_block(touch=False) or "")
    monkeypatch.setattr(grasp, "MEMORY_FLYMB_MODE", "live")
    buf2 = grasp.GraspBuffer(journal_enabled=False)
    for _ in range(3):
        buf2.fill("I love this wonderful result, thanks so much!", "wonderful!")
    block = buf2.get_context_block(touch=False) or ""
    assert "[fly valence" in block and "advisory only" in block


def _teach_promote_mb(text, times=5):
    import cognition.consolidate.promote as promo
    from cognition.flymemory import text_features
    mb = promo._flymb()
    feats = text_features(text)
    for _ in range(times):
        mb.reinforce(mb.encode(feats), +1.0)
    return mb


def test_ltm_rank_helpers(monkeypatch):
    # LTM rank/dream share one helper: bounded bias, None-safe, mode-aware.
    import cognition.memory.memorize as mem
    assert mem._flymb_bias_for_text("") is None
    b = mem._flymb_bias_for_text("I love this wonderful result, thanks!")
    assert b is not None and -1.0 <= b <= 1.0
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "off")
    assert mem._flymb_mode() == "off"


def test_dream_boost_wiring_modes(monkeypatch):
    import cognition.memory.memorize as mem
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "shadow")
    assert mem._flymb_mode() == "shadow"
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    assert mem._flymb_mode() == "live"
    assert mem._flymb_float("MEMORY_FLYMB_LTM_W", 0.01) == 0.01
    assert mem._flymb_float("MEMORY_FLYMB_DREAM_W", 0.2) == 0.2


def test_forget_decay_wire(monkeypatch):
    import cognition.memory.forget as fg
    kw = dict(access_count=3, last_accessed_iso="2026-01-01T00:00:00Z",
              valence_tag="pos", valence_score=2,
              memory_text="I love this wonderful result, thanks!")
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "off")
    base = fg.compute_weighted_score(**kw)
    assert base > 0
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "shadow")
    import pytest as _pt
    assert fg.compute_weighted_score(**kw) == _pt.approx(base)  # shadow never changes scores
    assert fg._flymb_mode() == "shadow"
    from cognition.flymemory import text_features as _tf
    _mb = fg._flymb_fg()
    _f = _tf(kw["memory_text"])
    for _ in range(5):
        _mb.reinforce(_mb.encode(_f), +1.0)  # teach: praise predicts reward
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    assert fg.compute_weighted_score(**kw) >= base  # approach lingers


def test_plasticity_roundtrip(tmp_path, monkeypatch):
    import numpy as np
    from cognition.flymemory import FlyMB
    from cognition.flymemory.store import PlasticityStore, apply_mb
    db = tmp_path / "fly.db"
    store = PlasticityStore(db)
    assert store.load_mb(4064, 97) is None  # nothing stored yet
    assert store.load_cx() is None
    mb = FlyMB()
    feats = [0.5, -1.0, 0.3, 0.0, 1.0, -0.4, 0.2, 0.8]
    for _ in range(3):
        mb.reinforce(mb.encode(feats), +1.0)
    ind, idx, _ = mb._kcm
    import numpy as _np
    counts = _np.diff(ind).astype(_np.int64)
    pre = _np.repeat(_np.arange(len(ind) - 1, dtype=_np.int64), counts)
    rows = store.save_mb(pre, idx, mb._plastic)
    assert rows > 0
    store.save_cx(0.42)
    assert store.load_cx() == 0.42
    mb2 = FlyMB()
    assert mb2.valence_bias(feats) != mb.valence_bias(feats)  # taught vs naive differ
    n = apply_mb(mb2, store.load_mb(4064, 97))
    assert n == rows
    assert mb2.valence_bias(feats) == mb.valence_bias(feats)  # restored exactly
    # Shape mismatch refuses force-fit instead of corrupting.
    assert store.load_mb(10, 10) is None


def test_plasticity_debounce_and_env(tmp_path, monkeypatch):
    from cognition.flymemory.store import PlasticityStore
    monkeypatch.setenv("FLY_PLASTICITY_EVERY", "3")
    store = PlasticityStore(tmp_path / "fly.db")
    assert store.save_if_due_cx(0.1) is False
    assert store.save_if_due_cx(0.2) is False
    assert store.save_if_due_cx(0.3) is True
    assert store.load_cx() == 0.3
