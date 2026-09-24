"""Phase 10A — rate-based dopamine + eligibility traces.

Covers the QA contract:
  - exact trace-decay values (0.5 ** (age / halflife))
  - dopamine rise/decay dynamics
  - prediction-error sign and baseline adaptation
  - delayed reward at t+5 credits action t, not unrelated actions
  - long-run boundedness (trace store, da, baseline)
  - shadow computes but writes nothing; off is a no-op
  - legacy pulse() compatibility (signature + result keys)
  - scope separation (real vs flyworld-sim)
  - per-event cost stays far below a millisecond
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from cognition.flymemory import credit
from cognition.flymemory import dopamine as dopamine_mod


@pytest.fixture
def live_credit(monkeypatch):
    """Both flags live + a fresh user id with clean credit state."""
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("AIKO_FLY_DOPAMINE_MODE", "live")
    uid = f"phase10a-{time.time_ns()}"
    credit.clear(uid, scope=None)
    yield uid
    credit.clear(uid, scope=None)


def _fake_kc(i: int, n: int = 32) -> np.ndarray:
    rng = np.random.default_rng(1000 + i)
    v = rng.random(n)
    v[v < 0.7] = 0.0  # sparse, KC-like
    v = v / (np.linalg.norm(v) or 1.0)
    return v


class _FakeMB:
    """Deterministic stand-in: records (kc, reward), returns |reward|."""

    def __init__(self):
        self.calls: list[tuple[bytes, float]] = []

    def reinforce(self, kc, reward):
        r = float(reward)
        self.calls.append((np.asarray(kc).tobytes(), r))
        return abs(r)


@pytest.fixture
def fake_mb(monkeypatch):
    mb = _FakeMB()
    monkeypatch.setattr("cognition.fly_registry.get_flymb", lambda _u: mb)
    return mb


# ── trace decay ──────────────────────────────────────────────────────────

def test_trace_decay_exact_values(live_credit, fake_mb):
    uid = live_credit
    kcs = [_fake_kc(i) for i in range(6)]
    for k in kcs:
        credit.mark_trace(uid, k, scope="real")
    st = credit.stats(uid)
    assert st["n_traces"] == 6
    assert st["clock"] == 6
    d = 0.5 ** (1.0 / 6.0)
    # stats() sorts by eligibility desc: newest (age 0) first.
    expected = sorted((round(d ** age, 4) for age in range(6)), reverse=True)
    assert [t["e"] for t in st["traces"]] == expected


def test_identical_mark_refreshes_without_tick(live_credit, fake_mb):
    uid = live_credit
    k = _fake_kc(0)
    t1 = credit.mark_trace(uid, k, scope="real")
    t2 = credit.mark_trace(uid, k, scope="real")  # bit-identical KC
    assert t1 == t2
    st = credit.stats(uid)
    assert st["n_traces"] == 1
    assert st["clock"] == 1  # no tick on refresh
    assert st["traces"][0]["e"] == 1.0


def test_near_duplicate_marks_stay_separate(live_credit, fake_mb):
    uid = live_credit
    a = _fake_kc(0)
    b = a.copy()
    b[1] += 1e-3  # close but not bit-identical
    assert credit.mark_trace(uid, a, scope="real") != credit.mark_trace(uid, b, scope="real")
    assert credit.stats(uid)["n_traces"] == 2


# ── dopamine dynamics ────────────────────────────────────────────────────

def test_dopamine_rise_and_decay(live_credit, fake_mb):
    uid = live_credit
    r1 = credit.credit_event(uid, 0.5, kc=None, scope="real")
    assert r1["pe"] == pytest.approx(0.5)
    assert r1["da"] == pytest.approx(0.5)      # rise: 0 * 0.6 + 1.0 * 0.5
    assert r1["baseline"] == pytest.approx(0.1)  # 0 + 0.2 * 0.5
    # A neutral-ish outcome after the baseline adapted: pure decay.
    r2 = credit.credit_event(uid, 0.1, kc=None, scope="real")
    assert r2["pe"] == pytest.approx(0.0)
    assert r2["da"] == pytest.approx(0.3)      # 0.5 * 0.6 + 0
    assert -1.0 <= r2["da"] <= 1.0


def test_prediction_error_sign_and_baseline_adaptation(live_credit, fake_mb):
    uid = live_credit
    pos = credit.credit_event(uid, 1.0, kc=None, scope="real")
    assert pos["pe"] > 0 and pos["da"] > 0
    # Repeating the same reward shrinks the prediction error.
    pos2 = credit.credit_event(uid, 1.0, kc=None, scope="real")
    assert 0 < pos2["pe"] < pos["pe"]
    # A surprising punishment flips the signal negative.
    neg = credit.credit_event(uid, -1.0, kc=None, scope="real")
    assert neg["pe"] < 0 and neg["da"] < 0


def test_dopamine_signal_stays_bounded(live_credit, fake_mb):
    uid = live_credit
    for i in range(200):
        r = 1.0 if i % 2 == 0 else -1.0
        out = credit.credit_event(uid, r, kc=None, scope="real")
        assert -1.0 <= out["da"] <= 1.0
        assert -1.0 <= out["baseline"] <= 1.0
    st = credit.stats(uid)
    assert -1.0 <= st["da"] <= 1.0
    assert -1.0 <= st["baseline"] <= 1.0


# ── delayed credit ───────────────────────────────────────────────────────

def test_delayed_reward_credits_action_not_unrelated(live_credit, fake_mb):
    uid = live_credit
    mb = fake_mb
    kcs = [_fake_kc(i) for i in range(6)]
    for k in kcs:
        credit.mark_trace(uid, k, scope="real")
    stranger = _fake_kc(99)  # never marked

    res = credit.credit_event(uid, 1.0, kc=None, source="t", scope="real")
    assert res["reason"] == "ok"
    assert res["applied"] is True
    assert res["n_applied_traces"] == 6

    by_kc = {b: r for b, r in mb.calls}
    d = 0.5 ** (1.0 / 6.0)
    # da = 1.0 (pe = 1.0 − 0), lr = 1.0 → update_i = eligibility_i.
    for i, k in enumerate(kcs):
        age = 5 - i
        assert by_kc[k.tobytes()] == pytest.approx(d ** age), i
    # The unrelated pattern learned nothing.
    assert stranger.tobytes() not in by_kc


def test_old_trace_below_cutoff_gets_nothing(live_credit, fake_mb, monkeypatch):
    uid = live_credit
    mb = fake_mb
    monkeypatch.setenv("FLY_TRACE_MAX", "10000")  # isolate the cutoff
    old = _fake_kc(0)
    credit.mark_trace(uid, old, scope="real")
    for i in range(1, 60):
        credit.mark_trace(uid, _fake_kc(i), scope="real")
    # e_old = 0.5 ** (59/6) ≈ 0.001 < cutoff 0.02 → pruned long ago.
    res = credit.credit_event(uid, 1.0, kc=None, scope="real")
    assert res["applied"] is True  # younger traces still learn
    assert old.tobytes() not in {b for b, _ in mb.calls}


def test_trace_store_bounded(live_credit, fake_mb):
    uid = live_credit
    for i in range(500):
        credit.mark_trace(uid, _fake_kc(i), scope="real")
    st = credit.stats(uid)
    assert st["n_traces"] <= 32  # FLY_TRACE_MAX default
    assert st["clock"] == 500


# ── modes ────────────────────────────────────────────────────────────────

def test_shadow_computes_but_writes_nothing(monkeypatch, fake_mb):
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    # AIKO_FLY_DOPAMINE_MODE left at default "shadow".
    uid = f"phase10a-shadow-{time.time_ns()}"
    credit.clear(uid, scope=None)
    mb = fake_mb
    credit.mark_trace(uid, _fake_kc(0), scope="real")
    res = credit.credit_event(uid, 0.8, kc=_fake_kc(1), scope="real")
    assert res["reason"] == "shadow"
    assert res["applied"] is False
    assert res["pe"] == pytest.approx(0.8)   # computed...
    assert res["da"] == pytest.approx(0.8)   # ...and tracked...
    assert mb.calls == []                     # ...but nothing written.
    assert res["n_applied_traces"] == 0
    credit.clear(uid, scope=None)


def test_dopamine_off_is_noop(monkeypatch, fake_mb):
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    monkeypatch.setenv("AIKO_FLY_DOPAMINE_MODE", "off")
    uid = f"phase10a-off-{time.time_ns()}"
    assert credit.mark_trace(uid, _fake_kc(0), scope="real") is None
    res = credit.credit_event(uid, 0.8, kc=_fake_kc(1), scope="real")
    assert res["reason"] == "dopamine_off"
    assert res["applied"] is False
    assert fake_mb.calls == []
    assert credit.stats(uid)["n_traces"] == 0


def test_mb_off_is_noop(monkeypatch, fake_mb):
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "off")
    monkeypatch.setenv("AIKO_FLY_DOPAMINE_MODE", "live")
    uid = f"phase10a-mboff-{time.time_ns()}"
    res = credit.credit_event(uid, 0.8, kc=_fake_kc(1), scope="real")
    assert res["reason"] == "mb_off"
    assert res["applied"] is False
    assert fake_mb.calls == []


# ── scope separation ─────────────────────────────────────────────────────

def test_sim_scope_never_touches_real_traces(live_credit, fake_mb):
    uid = live_credit
    mb = fake_mb
    real_kc = _fake_kc(0)
    sim_kc = _fake_kc(1)
    credit.mark_trace(uid, real_kc, scope="real")
    credit.mark_trace(uid, sim_kc, scope="flyworld-sim")
    res = credit.credit_event(uid, 1.0, kc=None, scope="flyworld-sim",
                              source="flyworld-sim")
    assert res["simulated"] is True
    by_kc = {b: r for b, r in mb.calls}
    assert sim_kc.tobytes() in by_kc
    assert real_kc.tobytes() not in by_kc  # real credit untouched
    assert credit.stats(uid, scope="real")["n_traces"] == 1


# ── legacy pulse() compatibility ─────────────────────────────────────────

def test_pulse_shim_keeps_signature_and_keys(live_credit, fake_mb):
    uid = live_credit
    kc = _fake_kc(2)
    res = dopamine_mod.pulse(0.5, user_id=uid, kc=kc, weight=1.0,
                             source="compat")
    for key in ("mode", "applied", "delta", "pam", "ppl1", "source",
                "reason", "drive", "pe", "da", "baseline", "n_traces"):
        assert key in res, key
    assert res["mode"] == "live"
    assert res["applied"] is True
    assert res["reason"] == "ok"
    assert res["pam"] == pytest.approx(0.5) and res["ppl1"] == 0.0
    assert res["source"] == "compat"


def test_pulse_shim_negative_reward_ppl1(live_credit, fake_mb):
    uid = live_credit
    res = dopamine_mod.pulse(-0.7, user_id=uid, kc=_fake_kc(3))
    assert res["applied"] is True
    assert res["ppl1"] == pytest.approx(0.7) and res["pam"] == 0.0
    assert res["da"] < 0


def test_pulse_shim_no_kc_no_text(live_credit, fake_mb):
    res = dopamine_mod.pulse(0.5, user_id=live_credit)
    assert res["reason"] == "no_kc"
    assert res["applied"] is False


def test_pulse_shim_shadow_by_default(monkeypatch, fake_mb):
    monkeypatch.setenv("MEMORY_FLYMB_MODE", "live")
    uid = f"phase10a-pulseshadow-{time.time_ns()}"
    res = dopamine_mod.pulse(0.5, user_id=uid, kc=_fake_kc(4))
    assert res["mode"] == "shadow"
    assert res["applied"] is False
    assert res["reason"] == "shadow"
    assert fake_mb.calls == []


# ── performance ──────────────────────────────────────────────────────────

def test_per_event_cost_well_below_millisecond(live_credit, fake_mb):
    uid = live_credit
    for i in range(32):  # fill the trace store to its cap
        credit.mark_trace(uid, _fake_kc(i), scope="real")
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        credit.credit_event(uid, 0.5, kc=None, scope="real")
    dt_ms = (time.perf_counter() - t0) / n * 1000.0
    assert dt_ms < 1.0, f"{dt_ms:.3f} ms per event"
