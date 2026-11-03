"""Phase 10A — rate-based dopamine + eligibility traces (delayed credit).

Replaces the one-shot pulse model with a continuous modulatory signal:

    reward event
      → prediction error  pe = clip(r − E, −1, 1)      (E = running baseline)
      → dopamine signal   da ← clip(da·δ_da + RISE·pe, −1, 1)
      → eligibility traces e_i decay per salient event: e ← e·δ_tr
      → weight update     Δw_i ∝ da · e_i               (credit flows backward)

Action now → consequence later → credit assigned backward: a trace marked
at action time is still alive (decayed) when the much-later reward arrives,
and the dopamine signal at that moment scales its update. Traces that had
nothing to do with the outcome have decayed to ~0 and learn nothing.

Equations (all bounded, deterministic given the event sequence; the clock
is event-stepped, never wall-clock):

    baseline:  E  ← clip(E + α·(r − E), −1, 1)        α = FLY_DOPAMINE_BASELINE_ALPHA (0.2)
    dopamine:  da ← clip(da·δ_da + RISE·pe, −1, 1)     δ_da = FLY_DOPAMINE_DECAY (0.6)
    trace:     e  ← e · 0.5**(1/halflife)              halflife = FLY_TRACE_HALFLIFE_EVENTS (6.0)
    update:    Δ  = clip(da · e · LR, −1, 1)           LR = FLY_DOPAMINE_LR (1.0)

Reversal boost (kept from the pulse model): when the MB's current valence
for the event text disagrees in sign with the reward (bias·r < −0.05),
the prediction error is surprising, so pe ← clip(pe·ρ, −1, 1) with
ρ = FLY_REVERSAL_MULT (1.6).

Scopes: traces and dopamine state are keyed by (user_id, scope). The real
turn path uses scope="real"; the FlyWorld simulator uses
scope="flyworld-sim" so synthetic experience never pollutes real credit
(Phase 9 invariant). The "replay" scope holds real offline replay.

Mode AIKO_FLY_DOPAMINE_MODE=off|shadow|live (defaults to live when MB is live,
otherwise shadow):
  off    : no-op, zero overhead, no state changes.
  shadow : full computation + trail logging, zero weight changes.
  live   : the rate path applies — but still only when MEMORY_FLYMB_MODE
           is also live. Both flags must agree for a write.

Fail-soft: never raises. No ML weights. Per-event cost is bounded
(≤ FLY_TRACE_MAX traces, ≤ FLY_TRACE_UPDATE_MAX reinforce calls).
"""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger("aiko.flymemory.credit")

# ── env ──────────────────────────────────────────────────────────────────


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _str_env(name: str, default: str) -> str:
    try:
        from system.config import env_str
        return env_str(name, default)
    except Exception:
        return os.getenv(name, default) or default


def dopamine_mode() -> str:
    """AIKO_FLY_DOPAMINE_MODE: off | shadow | live (defaults from MB mode)."""
    default = "live" if _mb_mode() == "live" else "shadow"
    m = (_str_env("AIKO_FLY_DOPAMINE_MODE", default) or default).strip().lower()
    return m if m in ("off", "shadow", "live") else "shadow"


def _mb_mode() -> str:
    try:
        from system.config import env_str
        return env_str("MEMORY_FLYMB_MODE", "off").strip().lower()
    except Exception:
        return (os.getenv("MEMORY_FLYMB_MODE") or "off").strip().lower()


def _rise() -> float:
    return max(0.0, min(2.0, _float_env("FLY_DOPAMINE_RISE", 1.0)))


def _da_decay() -> float:
    return max(0.0, min(1.0, _float_env("FLY_DOPAMINE_DECAY", 0.6)))


def _baseline_alpha() -> float:
    return max(0.0, min(1.0, _float_env("FLY_DOPAMINE_BASELINE_ALPHA", 0.2)))


def _lr() -> float:
    return max(0.0, min(2.0, _float_env("FLY_DOPAMINE_LR", 1.0)))


def _trace_decay() -> float:
    halflife = max(0.5, _float_env("FLY_TRACE_HALFLIFE_EVENTS", 6.0))
    return 0.5 ** (1.0 / halflife)


def _trace_max() -> int:
    return max(1, min(256, _int_env("FLY_TRACE_MAX", 32)))


def _update_max() -> int:
    return max(1, min(64, _int_env("FLY_TRACE_UPDATE_MAX", 16)))


def _trace_cutoff() -> float:
    return max(0.0, min(0.5, _float_env("FLY_TRACE_CUTOFF", 0.02)))


def _mark_dedup_cos() -> float:
    # Default 1.0: only bit-identical KC patterns merge. Measured: unrelated
    # sentences can reach cosine 0.99 (sparse KC overlap), so any looser
    # default would misattribute credit across different experiences.
    # Bit-identical patterns are indistinguishable to the MB anyway, so
    # merging them is honest. A lower env value allows looser merging.
    return max(0.0, min(1.0, _float_env("FLY_TRACE_MARK_DEDUP_COS", 1.0)))


def _pulse_dedup_cos() -> float:
    return max(0.0, min(1.0, _float_env("FLY_DOPAMINE_DEDUP_COS", 0.9)))


def _pulse_dedup_tol() -> float:
    return max(0.0, _float_env("FLY_DOPAMINE_DEDUP_REWARD_TOL", 0.15))


def _pulse_dedup_seconds() -> float:
    return max(0.0, _float_env("FLY_DOPAMINE_DEDUP_SECONDS", 2.0))


def _reversal_mult() -> float:
    return max(1.0, _float_env("FLY_REVERSAL_MULT", 1.6))


# ── state ────────────────────────────────────────────────────────────────
# Keyed by (user_id, scope). In-memory only; plastic deltas persist via the
# normal flush_mb path. Bounded: ≤ FLY_TRACE_MAX traces per key.

_lock = threading.RLock()
_STATE: dict[tuple[str, str], dict] = {}
# Duplicate-event debounce (Phase 2 "stop the waste", kept from dopamine.py):
# near-identical consecutive teaching events ("thanks" / "thank you") teach
# nothing new. Records the last APPLIED event per key.
_LAST_EVENT: dict[tuple[str, str], tuple[np.ndarray, float, float]] = {}


def _uid(user_id: str | None) -> str:
    return (user_id or "").strip() or "default"


def _scope_key(user_id: str | None, scope: str | None) -> tuple[str, str]:
    return (_uid(user_id), (scope or "real").strip() or "real")


def _state_for(key: tuple[str, str]) -> dict:
    st = _STATE.get(key)
    if st is None:
        st = {
            "da": 0.0,          # dopamine signal ∈ [-1, 1]
            "baseline": 0.0,    # expected-reward baseline E ∈ [-1, 1]
            "clock": 0,         # event-stepped clock (marks only)
            "traces": {},       # tid -> {"kc": np.ndarray, "e": float, "birth": int}
            "next_tid": 1,
            "last_mark": None,  # (kc, tid) for mark dedup
        }
        _STATE[key] = st
    return st


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    try:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


# ── trace marking ────────────────────────────────────────────────────────


def mark_trace(
    user_id: str | None,
    kc,
    value: float = 1.0,
    *,
    scope: str | None = "real",
) -> int | None:
    """Mark an eligibility trace for a KC pattern. Returns trace id (or None).

    Ticks the event clock once and decays all existing traces — UNLESS the
    KC is bit-identical to the last mark (cosine ≥ 1−1e-9), in which case
    the existing trace is refreshed with no tick. Only the MB-indistinguishable
    case merges; near-duplicates stay separate traces, because merging them
    would misattribute credit across different experiences.
    """
    if dopamine_mode() == "off":
        return None
    try:
        arr = np.asarray(kc, dtype=np.float64).reshape(-1).copy()
    except Exception:
        return None
    if float(np.linalg.norm(arr)) <= 0.0:
        return None
    v0 = max(0.0, min(1.0, float(value)))
    key = _scope_key(user_id, scope)
    with _lock:
        st = _state_for(key)
        last = st["last_mark"]
        if last is not None:
            last_kc, last_tid = last
            thr = _mark_dedup_cos()
            if thr >= 1.0:
                thr = 1.0 - 1e-9  # float noise on bit-identical vectors
            if _cosine(arr, last_kc) >= thr:
                tr = st["traces"].get(last_tid)
                if tr is not None:
                    tr["e"] = min(1.0, max(float(tr["e"]), v0))
                    return last_tid
        # New trace: tick the clock, decay + prune the old ones.
        st["clock"] += 1
        dtr = _trace_decay()
        cutoff = _trace_cutoff()
        dead = [tid for tid, tr in st["traces"].items()
                if float(tr["e"]) * dtr < cutoff]
        for tid in dead:
            del st["traces"][tid]
        for tr in st["traces"].values():
            tr["e"] = float(tr["e"]) * dtr
        while len(st["traces"]) >= _trace_max():
            victim = min(st["traces"], key=lambda t: float(st["traces"][t]["e"]))
            del st["traces"][victim]
        tid = st["next_tid"]
        st["next_tid"] += 1
        st["traces"][tid] = {"kc": arr, "e": v0, "birth": st["clock"]}
        st["last_mark"] = (arr, tid)
        return tid


# ── reward event ─────────────────────────────────────────────────────────


def _is_duplicate_event(key: tuple[str, str], kc_arr, pe: float) -> bool:
    """Phase 2 debounce: same content + same PE sign + close magnitude."""
    try:
        last = _LAST_EVENT.get(key)
        if last is None:
            return False
        last_kc, last_pe, last_at = last
        if time.monotonic() - last_at > _pulse_dedup_seconds():
            return False
        if float(np.sign(pe)) != float(np.sign(last_pe)):
            return False
        if abs(pe - last_pe) > _pulse_dedup_tol():
            return False
        return _cosine(kc_arr, last_kc) >= _pulse_dedup_cos()
    except Exception:
        return False


def credit_event(
    user_id: str | None,
    reward: float,
    *,
    kc=None,
    mark_value: float = 1.0,
    text: str | None = None,
    source: str = "credit",
    scope: str | None = "real",
    apply: bool | None = None,
    broadcast: bool = True,
) -> dict:
    """One reward event through the rate-based credit path. Never raises.

    kc (optional): mark a trace for this pattern first (the "action now").
    kc=None: pure outcome event — credit flows to already-marked traces
      (the "consequence later").
    apply: None → decided by modes; True/False forces the caller's intent,
      but AIKO_FLY_DOPAMINE_MODE=live AND MEMORY_FLYMB_MODE=live are still
      required for any write.
    broadcast=False: credit only the trace marked by this event.
    """
    da_mode = dopamine_mode()
    mb_mode = _mb_mode()
    scope_name = (scope or "real").strip() or "real"
    simulated = scope_name not in ("real", "replay")
    out: dict = {
        "ran": False,
        "mode": "shadow",
        "dopamine_mode": da_mode,
        "applied": False,
        "delta": 0.0,
        "reward": 0.0,
        "pe": 0.0,
        "da": 0.0,
        "baseline": 0.0,
        "pam": 0.0,
        "ppl1": 0.0,
        "drive": 0.0,
        "n_traces": 0,
        "n_applied_traces": 0,
        "applications": [],
        "reversal": False,
        "source": source,
        "scope": scope_name,
        "simulated": simulated,
        "reason": "",
    }
    if da_mode == "off":
        out["reason"] = "dopamine_off"
        return out
    if mb_mode == "off":
        out["reason"] = "mb_off"
        return out
    try:
        r = max(-1.0, min(1.0, float(reward)))
    except Exception:
        out["reason"] = "bad_reward"
        return out
    if abs(r) < 1e-9:
        out["reason"] = "zero_drive"
        return out
    out["reward"] = r
    out["pam"] = max(0.0, r)
    out["ppl1"] = max(0.0, -r)

    key = _scope_key(user_id, scope_name)
    try:
        from cognition.fly_registry import get_flymb

        mb = get_flymb(user_id)
        if mb is None or not hasattr(mb, "reinforce"):
            out["reason"] = "mb_unavailable"
            return out

        # Resolve the pattern: explicit KC wins, else encode(text).
        kc_arr = None
        if kc is not None:
            try:
                kc_arr = np.asarray(kc, dtype=np.float64).reshape(-1)
                if float(np.linalg.norm(kc_arr)) <= 0.0:
                    kc_arr = None
            except Exception:
                kc_arr = None
        elif text:
            try:
                from cognition.flymemory.circuit import text_features
                kc_arr = mb.encode(text_features(text))
                if float(np.linalg.norm(kc_arr)) <= 0.0:
                    kc_arr = None
            except Exception as exc:
                out["reason"] = f"no_kc: {exc}"
                return out
        # kc_arr is None and no text: pure outcome event — credit flows to
        # already-marked traces (the assign_credit path).

        with _lock:
            st = _state_for(key)

            # 1. Mark the current pattern's trace (the "action now").
            target_tid = None
            if kc_arr is not None:
                target_tid = mark_trace(user_id, kc_arr, value=mark_value, scope=scope_name)

            # 2. Prediction error against the running baseline.
            E = float(st["baseline"])
            pe = max(-1.0, min(1.0, r - E))

            # 3. Reversal boost: reward disagrees in sign with the MB's
            #    current valence for this text → surprising → amplify.
            prior_bias = None
            if text:
                try:
                    from cognition.flymemory.circuit import text_features
                    prior_bias = float(mb.valence_bias(text_features(text)))
                    out["prior_bias"] = round(prior_bias, 4)
                    if prior_bias * r < -0.05:
                        pe = max(-1.0, min(1.0, pe * _reversal_mult()))
                        out["reversal"] = True
                except Exception:
                    pass

            # 4. Duplicate-event debounce (Phase 2).
            if kc_arr is not None and _is_duplicate_event(key, kc_arr, pe):
                out["reason"] = "dedup"
                out["pe"] = round(pe, 4)
                out["da"] = round(float(st["da"]), 4)
                out["baseline"] = round(E, 4)
                return out

            # 5. Dopamine dynamics: decay, then rise on the prediction error.
            da = max(-1.0, min(1.0, float(st["da"]) * _da_decay() + _rise() * pe))
            st["da"] = da
            # 6. Baseline learns the outcome AFTER the error is computed.
            st["baseline"] = max(-1.0, min(1.0,
                E + _baseline_alpha() * (r - E)))

            out["pe"] = round(pe, 4)
            out["da"] = round(da, 4)
            out["baseline"] = round(float(st["baseline"]), 4)
            out["drive"] = round(pe * max(0.0, min(1.0, float(mark_value))), 4)

            live = (apply if apply is not None else True) and da_mode == "live" and mb_mode == "live"
            out["mode"] = "live" if live else "shadow"

            # 7. Credit backward: each live trace gets da·e_i, strongest first.
            items = sorted(st["traces"].items(),
                           key=lambda kv: float(kv[1]["e"]), reverse=True)
            if not broadcast:
                items = [(tid, tr) for tid, tr in items if tid == target_tid]
            out["n_traces"] = len(items)
            lr = _lr()
            total = 0.0
            applied_n = 0
            apps: list[dict] = []
            if abs(da) >= 1e-9:
                for tid, tr in items[:_update_max()]:
                    e = float(tr["e"])
                    if e < _trace_cutoff():
                        break  # sorted desc: the rest are even smaller
                    update = max(-1.0, min(1.0, da * e * lr))
                    age = int(st["clock"]) - int(tr["birth"])
                    app: dict = {
                        "trace": int(tid),
                        "age": age,
                        "e": round(e, 4),
                        "update": round(update, 4),
                        "applied": False,
                        "delta": 0.0,
                        "reason": "shadow",
                    }
                    if live:
                        try:
                            d = float(mb.reinforce(tr["kc"], update) or 0.0)
                        except Exception:
                            d = 0.0
                        app["delta"] = round(d, 4)
                        app["applied"] = d > 0.0
                        app["reason"] = "ok" if d > 0.0 else "no_delta"
                        if d > 0.0:
                            total += d
                            applied_n += 1
                    apps.append(app)
            out["applications"] = apps
            out["n_applied_traces"] = applied_n
            out["delta"] = round(total, 4)
            out["applied"] = total > 0.0
            out["ran"] = True
            out["reason"] = "ok" if out["applied"] else "shadow"
            if live and out["applied"] and kc_arr is not None:
                _LAST_EVENT[key] = (kc_arr.copy(), pe, time.monotonic())

        try:
            from cognition.neural_state import get_neural_state
            get_neural_state(user_id).record_influence({
                "kind": "dopamine",
                "mode": out["mode"],
                "source": source,
                "scope": scope_name,
                "simulated": simulated,
                "reward": r,
                "pe": out["pe"],
                "da": out["da"],
                "baseline": out["baseline"],
                "pam": round(out["pam"], 4),
                "ppl1": round(out["ppl1"], 4),
                "drive": out["drive"],
                "delta": out["delta"],
                "n_traces": out["n_traces"],
                "n_applied_traces": out["n_applied_traces"],
                "reversal": bool(out["reversal"]),
                "applied": out["applied"],
            })
        except Exception:
            pass
        # Phase 11: trait plasticity — the ONLY path that mutates the
        # neural personality state. Guarded inside on_credit_outcome:
        # live persona mode AND scope="real" only.
        if out.get("ran"):
            try:
                from cognition.fly_persona.plasticity import on_credit_outcome
                out["persona"] = on_credit_outcome(
                    user_id,
                    reward=r,
                    pe=out.get("pe", 0.0),
                    da=out.get("da", 0.0),
                    scope=scope_name,
                )
            except Exception as exc:
                log.debug("credit persona plasticity skipped: %s", exc)
    except Exception as exc:
        log.debug("credit event skipped: %s", exc)
        out["reason"] = f"error: {exc}"
    return out


# ── observability / tests ────────────────────────────────────────────────


def stats(user_id: str | None = None, scope: str | None = "real") -> dict:
    """Dopamine state + trace inventory (harness/Studio)."""
    key = _scope_key(user_id, scope)
    with _lock:
        st = _STATE.get(key)
        if st is None:
            return {"da": 0.0, "baseline": 0.0, "clock": 0, "n_traces": 0,
                    "traces": []}
        return {
            "da": round(float(st["da"]), 4),
            "baseline": round(float(st["baseline"]), 4),
            "clock": int(st["clock"]),
            "n_traces": len(st["traces"]),
            "traces": [
                {"e": round(float(tr["e"]), 4),
                 "age": int(st["clock"]) - int(tr["birth"])}
                for tr in sorted(st["traces"].values(),
                                 key=lambda t: float(t["e"]), reverse=True)
            ],
        }


def clear(user_id: str | None = None, scope: str | None = None) -> None:
    """Drop credit state (tests / identity switch). scope=None clears all."""
    with _lock:
        if scope is None:
            for key in [k for k in _STATE if k[0] == _uid(user_id)]:
                _STATE.pop(key, None)
                _LAST_EVENT.pop(key, None)
        else:
            key = _scope_key(user_id, scope)
            _STATE.pop(key, None)
            _LAST_EVENT.pop(key, None)
