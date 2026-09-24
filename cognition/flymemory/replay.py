"""Offline experience replay — computational analogue of sleep consolidation (Phase 8).

FRAMING — read before extending: "replay" here is a *computational analogue*
of sleep-consolidation motifs (re-activate eligible experiences offline so
plasticity compounds). It is NOT a claim about biological Drosophila sleep
physiology, sleep stages, or replay neurobiology. Keep it that way in any
comment, doc, or UI string you add.

Causal chain this closes:
    experience -> MB/fly valence -> eligibility + plasticity
        -> sleep/replay -> Memory Bank consolidation
        -> future retrieval -> future fly state

Pipeline (nightly system job, default 23:30 local — ahead of the 00:00
dream/consolidation job so replayed plasticity is visible to it):
  1. SELECT: recent episodes from emc_storage (last FLY_REPLAY_LOOKBACK_H
     hours). Each is scored:
         recency   = 0.5 ** (age_h / halflife_h)          # in (0, 1]
         salience  = clip(|valence| + 0.5*salience_score, 0, 1)
         elig      = recency * salience                    # in [0, 1]
     Items with elig < 0.05 are skipped (same cutoff convention as
     eligibility.py). Top FLY_REPLAY_MAX_ITEMS by (elig desc, id asc) —
     deterministic ordering, bounded work.
  2. REPLAY (live mode only): encode the episode trace -> KC, read valence
     before/after, and apply one dopamine.pulse(reward=valence, kc=kc,
     weight=elig, source="replay") per item — the same canonical teaching
     path online_teach() uses, including its dedup guard.
  3. CONSOLIDATE: one fly_store.flush_mb(mb) at the end (Phase 2 convention:
     one flush per teaching event, not per item).

How this wires into the REAL Memory Bank lifecycle (no parallel toy store):
  grasp.py, consolidate/promote.py, memory/forget.py and memorize.py's
  dream-boost all read mb.valence_bias() LIVE at scoring time, scaled by the
  existing MEMORY_FLYMB_* config weights. Replay re-teaches the MB, so the
  plastic weights shift, future valence_bias() outputs shift, and grasp /
  promotion / decay-half-life / dream-boost scores move through the real
  code paths with the real weights. The trail below documents the predicted
  per-item effect (valence before -> after) for observability.

Mode: AIKO_FLY_REPLAY_MODE=off|shadow|live (default shadow).
  off    : return immediately, zero overhead.
  shadow : select + score + trail everything, skip pulses ("would_replay").
  live   : apply pulses and flush.

Fail-soft: every external call is wrapped; exceptions log and skip. The
replay path must never break the scheduler or the memory lifecycle.
No ML weights are loaded here; per-item cost is one sparse KC reinforce.
"""
from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

log = logging.getLogger("aiko.flymemory.replay")

_lock = threading.RLock()
_LAST_RUN: dict[str, dict] = {}

# Sentinel: replay is a no-op unless the MB layer itself is on.
_MB_LIVE_MODES = ("live",)


def _uid(user_id: str | None) -> str:
    return (user_id or "").strip() or "default"


def _env_str(name: str, default: str) -> str:
    try:
        from system.config import env_str
        return env_str(name, default)
    except Exception:
        return os.getenv(name, default) or default


def replay_mode() -> str:
    """Current replay mode: off | shadow | live (default shadow)."""
    m = (_env_str("AIKO_FLY_REPLAY_MODE", "shadow") or "shadow").strip().lower()
    return m if m in ("off", "shadow", "live") else "shadow"


def _mb_mode() -> str:
    m = (_env_str("MEMORY_FLYMB_MODE", "off") or "off").strip().lower()
    return m


def _float_env(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(os.getenv(name, str(default)))))
    except Exception:
        return default


def _int_env(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.getenv(name, str(default)))))
    except Exception:
        return default


def _lookback_h() -> float:
    return _float_env("FLY_REPLAY_LOOKBACK_H", 20.0, 1.0, 72.0)


def _max_items() -> int:
    return _int_env("FLY_REPLAY_MAX_ITEMS", 50, 1, 500)


def _halflife_h() -> float:
    return _float_env("FLY_REPLAY_HALFLIFE_H", 6.0, 0.5, 48.0)


def _min_elig() -> float:
    return _float_env("FLY_REPLAY_MIN_ELIG", 0.05, 0.0, 1.0)


def score_candidate(age_h: float, valence: float, salience: float | None) -> float:
    """Eligibility score in [0, 1]. Pure function — deterministic, testable.

    recency  = 0.5 ** (age_h / halflife_h)          # exponential decay
    salience = clip(|valence| + 0.5 * salience, 0, 1)
    elig     = recency * salience
    """
    try:
        age = max(0.0, float(age_h))
        v = max(-1.0, min(1.0, float(valence)))
        s = float(salience) if salience is not None else 0.0
        s = max(0.0, s)
    except Exception:
        return 0.0
    recency = 0.5 ** (age / _halflife_h())
    sal = max(0.0, min(1.0, abs(v) + 0.5 * s))
    return max(0.0, min(1.0, recency * sal))


def _parse_ts(value) -> float | None:
    """ISO timestamp text -> epoch seconds. None when unparseable."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _db_path_for_user(user_id: str) -> str | None:
    """Episodic DB path for a user. None when unresolvable (fail-soft).

    Kept as a module-level seam so tests can stub it without importing
    cognition.memory.schema (which needs sqlite_vec, absent in some envs).
    """
    try:
        from cognition.memory.schema import _memory_db_path_for_user
        return str(_memory_db_path_for_user(_uid(user_id)))
    except Exception as exc:
        log.debug("replay: db path unavailable: %s", exc)
        return None


def _recent_episodes(user_id: str, now: float) -> list[dict]:
    """Read recent emc_storage episodes via a short-lived read connection.

    Returns dicts: {id, trace, salience, age_h}. Never raises; [] on any
    failure (missing DB, missing table, locked, ...). Superseded episodes
    are skipped — they have been replaced by a newer version.
    """
    out: list[dict] = []
    db_path = _db_path_for_user(user_id)
    if not db_path:
        return out
    try:
        import os as _os
        if not _os.path.exists(str(db_path)):
            return out
        cutoff = datetime.fromtimestamp(
            now - _lookback_h() * 3600.0, tz=timezone.utc
        ).isoformat()
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            rows = conn.execute(
                "SELECT id, trace, salience_score, timestamp, valence_tag FROM emc_storage"
                " WHERE user_id = ? AND timestamp >= ?"
                " AND (superseded_by IS NULL)"
                " ORDER BY id ASC",
                (_uid(user_id), cutoff),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        log.debug("replay: episode read skipped: %s", exc)
        return out
    for row in rows:
        try:
            eid, trace, sal, ts = row[0], row[1], row[2], row[3]
            if not (trace or "").strip():
                continue
            ts_epoch = _parse_ts(ts)
            if ts_epoch is None:
                continue
            out.append({
                "id": eid,
                "trace": trace,
                "salience": sal,
                "age_h": max(0.0, (now - ts_epoch) / 3600.0),
            })
        except Exception:
            continue
    return out


def select_candidates(
    episodes: list[dict],
    *,
    valence_of,
    now: float | None = None,
) -> list[dict]:
    """Score + rank episode dicts. Pure apart from the valence_of callback.

    valence_of(trace) -> float in [-1, 1] (live MB valence, same signal the
    lifecycle knobs read). Returns at most FLY_REPLAY_MAX_ITEMS dicts, each
    with {id, trace, valence, salience, age_h, elig}, ordered by
    (elig desc, id asc) — deterministic for identical inputs.
    """
    _ = now  # reserved: age_h is precomputed by the caller.
    scored: list[dict] = []
    min_elig = _min_elig()
    for ep in episodes:
        try:
            v = max(-1.0, min(1.0, float(valence_of(ep["trace"]))))
        except Exception:
            continue
        elig = score_candidate(ep.get("age_h", 0.0), v, ep.get("salience"))
        if elig < min_elig:
            continue
        scored.append({
            "id": ep["id"],
            "trace": ep["trace"],
            "valence": round(v, 4),
            "salience": ep.get("salience"),
            "age_h": round(float(ep.get("age_h", 0.0)), 3),
            "elig": round(elig, 4),
        })
    scored.sort(key=lambda d: (-d["elig"], d["id"]))
    return scored[: _max_items()]


def _mb_valence_of(mb):
    def _v(trace: str) -> float:
        from cognition.flymemory.circuit import text_features
        return float(mb.valence_bias(text_features(trace or "")))
    return _v


def run_replay(user_id: str | None = None, *, force: bool = False) -> dict:
    """Run one offline replay pass. Never raises.

    force=True bypasses the mode gate for tests/harness (still fail-soft).
    """
    mode = replay_mode()
    out: dict = {
        "ran": False, "mode": mode, "n_candidates": 0, "n_replayed": 0,
        "n_skipped_dedup": 0, "plastic_before": 0.0, "plastic_after": 0.0,
        "items": [], "reason": "",
    }
    if mode == "off" and not force:
        out["reason"] = "mode_off"
        return out
    if _mb_mode() not in _MB_LIVE_MODES and not force:
        out["reason"] = "mb_not_live"
        return out
    live = (mode == "live") or force
    try:
        now = time.time()
        episodes = _recent_episodes(_uid(user_id), now)
        out["n_candidates"] = len(episodes)
        if not episodes:
            out["reason"] = "no_episodes"
            _record_trail(user_id, out)
            return out

        from cognition.fly_registry import get_flymb, get_fly_store
        mb = get_flymb(user_id)
        if mb is None or not hasattr(mb, "_plastic"):
            out["reason"] = "mb_unavailable"
            _record_trail(user_id, out)
            return out

        import numpy as np
        out["plastic_before"] = round(float(np.abs(mb._plastic).sum()), 4)

        candidates = select_candidates(episodes, valence_of=_mb_valence_of(mb), now=now)
        items: list[dict] = []
        replayed = 0
        skipped_dedup = 0
        for cand in candidates:
            item = {
                "id": cand["id"], "valence": cand["valence"], "elig": cand["elig"],
                "age_h": cand["age_h"], "applied": False, "delta": 0.0,
                "valence_before": cand["valence"], "valence_after": cand["valence"],
            }
            if live:
                try:
                    from cognition.flymemory.circuit import text_features
                    from cognition.flymemory.dopamine import pulse
                    kc = mb.encode(text_features(cand["trace"]))
                    before = float(mb.valence_bias(text_features(cand["trace"])))
                    res = pulse(
                        cand["recorded_valence"], user_id=user_id, kc=kc,
                        weight=cand["elig"], source="replay",
                    )
                    if res.get("reason") == "dedup":
                        skipped_dedup += 1
                        item["reason"] = "dedup"
                    elif res.get("applied"):
                        replayed += 1
                        item["applied"] = True
                        item["delta"] = float(res.get("delta") or 0.0)
                        after = float(mb.valence_bias(text_features(cand["trace"])))
                        item["valence_before"] = round(before, 4)
                        item["valence_after"] = round(after, 4)
                    else:
                        item["reason"] = str(res.get("reason") or "not_applied")
                except Exception as exc:
                    item["reason"] = f"error: {exc}"
                    log.debug("replay item %s skipped: %s", cand["id"], exc)
            else:
                item["reason"] = "would_replay"
            items.append(item)

        if live and replayed:
            try:
                store = get_fly_store(user_id)
                if store is not None:
                    store.flush_mb(mb)
                    out["flushed"] = True
            except Exception as exc:
                log.debug("replay flush skipped: %s", exc)
        out["plastic_after"] = round(float(np.abs(mb._plastic).sum()), 4)
        out["items"] = items
        out["n_replayed"] = replayed
        out["n_skipped_dedup"] = skipped_dedup
        out["ran"] = True
        out["reason"] = "ok"
        log.info(
            "fly replay user=%s mode=%s candidates=%d replayed=%d dedup=%d "
            "plastic %.4f->%.4f",
            _uid(user_id)[:12], mode, out["n_candidates"], replayed,
            skipped_dedup, out["plastic_before"], out["plastic_after"],
        )
    except Exception as exc:
        log.debug("fly replay failed: %s", exc)
        out["reason"] = f"error: {exc}"
    _record_trail(user_id, out)
    return out


def _record_trail(user_id: str | None, out: dict) -> None:
    """Observability: neural-state influence event + last-run cache."""
    try:
        with _lock:
            _LAST_RUN[_uid(user_id)] = {
                "ts": time.time(), "mode": out.get("mode"),
                "n_candidates": out.get("n_candidates"),
                "n_replayed": out.get("n_replayed"),
                "reason": out.get("reason"),
            }
        from cognition.neural_state import get_neural_state
        get_neural_state(user_id).record_influence({
            "kind": "fly_replay",
            "mode": out.get("mode"),
            "n_candidates": out.get("n_candidates"),
            "n_replayed": out.get("n_replayed"),
            "n_skipped_dedup": out.get("n_skipped_dedup"),
            "plastic_before": out.get("plastic_before"),
            "plastic_after": out.get("plastic_after"),
            "reason": out.get("reason"),
            # Bounded: at most FLY_REPLAY_MAX_ITEMS entries, small dicts.
            "items": [
                {k: it[k] for k in (
                    "id", "valence", "elig", "applied", "delta",
                    "valence_before", "valence_after",
                ) if k in it}
                for it in (out.get("items") or [])
            ],
        })
    except Exception:
        pass


def last_run(user_id: str | None = None) -> dict:
    """Last replay run summary (for harness/Studio)."""
    with _lock:
        return dict(_LAST_RUN.get(_uid(user_id), {}))


def clear_last_run(user_id: str | None = None) -> None:
    with _lock:
        _LAST_RUN.pop(_uid(user_id), None)
