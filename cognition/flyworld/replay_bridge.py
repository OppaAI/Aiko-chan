"""Phase 9 — bridge: FlyWorld sim episodes -> Phase 8 replay machinery.

Sim episodes live in flyworld's own bounded in-memory trail — NEVER in
the real episodic store. Mixing synthetic experience into real episodic
memory would make it indistinguishable from reality; the bridge keeps
simulated=True on every candidate, pulse (source="flyworld-replay"), and
trail item, end to end.

Reuse, not duplication: replay.select_candidates (same eligibility math,
cutoffs, deterministic ordering) + dopamine.pulse + one flush_mb — i.e.
the Phase 8 machinery, not a copy of it.

The nightly fly_replay job does NOT auto-include sim episodes; opt in with
FLY_REPLAY_INCLUDE_SIM=1 (default 0) — see replay.run_replay(). The
direct entry point here is replay_sim(), for harness/tests/explicit runs.

Fail-soft: never raises.
"""
from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger("aiko.flyworld.replay_bridge")

_lock = threading.RLock()
_LAST_BRIDGE: dict[str, dict] = {}


def _uid(user_id: str | None) -> str:
    return (user_id or "").strip() or "default"


def _max_items() -> int:
    try:
        return max(1, min(200, int(os.getenv("FLYWORLD_BRIDGE_MAX_ITEMS", "50"))))
    except Exception:
        return 50


def collect_sim_episodes(user_id: str | None = None, max_n: int | None = None) -> list[dict]:
    """Sim steps reshaped for replay.select_candidates.

    Returns dicts {id, trace, salience, age_h, simulated: True}, newest
    episodes first, capped. Sim candidates have no recorded_valence, so
    replay falls back to live valence for the pulse (documented there).
    """
    from .loop import past_episodes

    cap = max_n or _max_items()
    out: list[dict] = []
    for ep in past_episodes(user_id, n=cap):
        seed = ep.get("seed", 0)
        for st in ep.get("steps", []) or []:
            if len(out) >= cap:
                break
            try:
                r = float(st.get("reward", 0.0) or 0.0)
            except Exception:
                continue
            out.append({
                "id": f"flyworld:{seed}:{st.get('step', 0)}",
                "trace": str(st.get("trace", "")),
                "salience": round(min(1.0, abs(r)), 4),
                "age_h": 0.05,  # fresh by construction
                "simulated": True,
            })
        if len(out) >= cap:
            break
    return out[:cap]


def replay_sim(user_id: str | None = None, *, force: bool = False) -> dict:
    """Replay sim episodes through the Phase 8 machinery. Never raises.

    Mode is AIKO_FLYWORLD_MODE (off|shadow|live, default shadow); force
    bypasses the gate for tests/harness. Pulses use
    source="flyworld-replay" and every trail item carries simulated=True.
    """
    from .loop import flyworld_mode
    from cognition.flymemory import replay as _replay
    from cognition.fly_registry import get_flymb, get_fly_store
    from cognition.flymemory.circuit import text_features
    from cognition.flymemory.dopamine import pulse
    import numpy as np

    mode = flyworld_mode()
    out: dict = {
        "ran": False, "mode": mode, "simulated": True,
        "n_candidates": 0, "n_replayed": 0, "n_skipped_dedup": 0,
        "items": [], "reason": "",
    }
    if mode == "off" and not force:
        out["reason"] = "mode_off"
        return out
    live = (mode == "live") or force
    try:
        episodes = collect_sim_episodes(user_id)
        out["n_candidates"] = len(episodes)
        if not episodes:
            out["reason"] = "no_sim_episodes"
            _record_bridge_trail(user_id, out)
            return out
        mb = get_flymb(user_id)
        if mb is None or not hasattr(mb, "encode"):
            out["reason"] = "mb_unavailable"
            _record_bridge_trail(user_id, out)
            return out

        def _valence_of(trace: str) -> float:
            return float(mb.valence_bias(text_features(trace or "")))

        candidates = _replay.select_candidates(episodes, valence_of=_valence_of)
        items: list[dict] = []
        replayed = 0
        skipped_dedup = 0
        for cand in candidates:
            item = {
                "id": cand["id"], "valence": cand["valence"], "elig": cand["elig"],
                "simulated": True, "applied": False, "delta": 0.0,
                "reward_source": "live",
            }
            if live:
                try:
                    kc = mb.encode(text_features(cand["trace"]))
                    res = pulse(
                        cand["valence"], user_id=user_id, kc=kc,
                        weight=cand["elig"], source="flyworld-replay",
                    )
                    if res.get("reason") == "dedup":
                        skipped_dedup += 1
                        item["reason"] = "dedup"
                    elif res.get("applied"):
                        replayed += 1
                        item["applied"] = True
                        item["delta"] = float(res.get("delta") or 0.0)
                    else:
                        item["reason"] = str(res.get("reason") or "not_applied")
                except Exception as exc:
                    item["reason"] = f"error: {exc}"
                    log.debug("flyworld replay item %s skipped: %s", cand["id"], exc)
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
                log.debug("flyworld replay flush skipped: %s", exc)
        out["items"] = items
        out["n_replayed"] = replayed
        out["n_skipped_dedup"] = skipped_dedup
        out["ran"] = True
        out["reason"] = "ok"
        log.info(
            "flyworld replay user=%s mode=%s candidates=%d replayed=%d",
            _uid(user_id)[:12], mode, out["n_candidates"], replayed,
        )
    except Exception as exc:
        log.debug("flyworld replay failed: %s", exc)
        out["reason"] = f"error: {exc}"
    _record_bridge_trail(user_id, out)
    return out


def _record_bridge_trail(user_id: str | None, out: dict) -> None:
    try:
        with _lock:
            _LAST_BRIDGE[_uid(user_id)] = {
                "ts": time.time(), "mode": out.get("mode"),
                "n_candidates": out.get("n_candidates"),
                "n_replayed": out.get("n_replayed"),
                "reason": out.get("reason"),
            }
        from cognition.neural_state import get_neural_state
        get_neural_state(user_id).record_influence({
            "kind": "flyworld_replay",
            "simulated": True,
            "mode": out.get("mode"),
            "n_candidates": out.get("n_candidates"),
            "n_replayed": out.get("n_replayed"),
            "reason": out.get("reason"),
            "items": [
                {k: it[k] for k in (
                    "id", "valence", "elig", "applied", "delta",
                    "reward_source", "simulated",
                ) if k in it}
                for it in (out.get("items") or [])
            ],
        })
    except Exception:
        pass


def last_bridge_run(user_id: str | None = None) -> dict:
    """Last bridge run summary (for harness/Studio)."""
    with _lock:
        return dict(_LAST_BRIDGE.get(_uid(user_id), {}))
