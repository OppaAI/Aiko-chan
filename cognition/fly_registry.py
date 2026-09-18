"""Identity-scoped registry for fly biological layers (MB + CX + plasticity).

Before this module, FlyMB / FlyCompass / PlasticityStore were process-global
singletons. In multi-user deployments one user's reinforce() could leak into
another user's GRASP ranking, and the agent orchestrator held a separate CX
instance that never shared sleep pressure with attention.

Usage:
    from cognition.fly_registry import get_flymb, get_flycx, get_fly_store, flush_all

    mb = get_flymb(user_id)          # per-identity FlyMB, loads plastic deltas
    cx = get_flycx(user_id)          # shared CX for attention + agent cadence
    store = get_fly_store(user_id)   # per-identity SQLite path
    flush_all(user_id)               # write pending deltas on shutdown / identity switch
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

try:
    from system.log import get_logger as _get_logger
    log = _get_logger(__name__)
except Exception:
    import logging as _logging
    log = _logging.getLogger("aiko.fly_registry")

_lock = threading.RLock()
_stores: dict[str, Any] = {}
_mbs: dict[str, Any] = {}
_cxs: dict[str, Any] = {}
_mb_ok: dict[str, bool] = {}
_cx_ok: dict[str, bool] = {}


def _norm_id(user_id: str | None) -> str:
    uid = (user_id or "").strip() or "default"
    # Keep path-safe; identities are github_123 / patreon_456 style.
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in uid)[:128]


def _default_db_for(user_id: str) -> Path:
    """Prefer per-user state dir when present; fall back to repo data/."""
    override = (os.getenv("FLY_PLASTICITY_DB") or "").strip()
    if override:
        # Explicit path: if it looks like a directory, nest by identity.
        p = Path(override).expanduser()
        if p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
            # Single shared file requested — still namespace with a table key
            # is handled by PlasticityStore identity column; path stays as-is
            # only when FLY_PLASTICITY_SHARED=1.
            if (os.getenv("FLY_PLASTICITY_SHARED") or "").strip() in ("1", "true", "yes"):
                return p
            return p.parent / f"fly_plasticity_{_norm_id(user_id)}.db"
        return p / f"fly_plasticity_{_norm_id(user_id)}.db"

    # Prefer USER_STATE_ROOT / AIKO_USER_STATE_ROOT / ~/.aiko/<id>/
    for key in ("USER_STATE_ROOT", "AIKO_USER_STATE_ROOT", "USER_SPACE_ROOT"):
        root = (os.getenv(key) or "").strip()
        if root:
            return Path(root).expanduser() / _norm_id(user_id) / "fly_plasticity.db"

    home = Path.home() / ".aiko" / _norm_id(user_id) / "fly_plasticity.db"
    if home.parent.parent.exists() or os.getenv("HOME"):
        return home

    # Last resort: repo-relative (dev / single-user)
    return Path(__file__).resolve().parents[1] / "data" / f"fly_plasticity_{_norm_id(user_id)}.db"


def get_fly_store(user_id: str | None = None):
    """Return (and cache) a PlasticityStore for this identity."""
    key = _norm_id(user_id)
    with _lock:
        if key not in _stores:
            try:
                from cognition.flymemory.store import PlasticityStore
                path = _default_db_for(key)
                _stores[key] = PlasticityStore(path=path, identity=key)
            except Exception as exc:
                log.debug("fly store unavailable for %s: %s", key, exc)
                _stores[key] = False
        store = _stores[key]
        return store or None


def get_flymb(user_id: str | None = None):
    """Per-identity FlyMB with plastic deltas restored from that identity's store."""
    key = _norm_id(user_id)
    with _lock:
        if key not in _mb_ok:
            try:
                from cognition.flymemory import FlyMB
                from cognition.flymemory.store import apply_mb
                mb = FlyMB()
                store = get_fly_store(key)
                if store is not None:
                    try:
                        ind, _, _ = mb._kcm
                        n = apply_mb(mb, store.load_mb(len(ind) - 1, mb.n_mbon))
                        if n:
                            log.debug("flymb[%s] restored %d plastic deltas", key, n)
                    except Exception as exc:
                        log.debug("flymb[%s] restore skipped: %s", key, exc)
                _mbs[key] = mb
                _mb_ok[key] = True
            except Exception as exc:
                log.debug("flymb[%s] unavailable: %s", key, exc)
                _mbs[key] = None
                _mb_ok[key] = False
        return _mbs.get(key) if _mb_ok.get(key) else None


def get_flycx(user_id: str | None = None):
    """Shared per-identity FlyCompass (attention + agent orchestrator must share this)."""
    key = _norm_id(user_id)
    with _lock:
        if key not in _cx_ok:
            try:
                from cognition.centralcomplex import FlyCompass
                cx = FlyCompass()
                store = get_fly_store(key)
                if store is not None:
                    try:
                        saved = store.load_cx()
                        if saved is not None:
                            cx.sleep_pressure = max(0.0, min(1.0, float(saved)))
                            log.debug("flycx[%s] restored sleep_pressure=%.3f", key, cx.sleep_pressure)
                    except Exception as exc:
                        log.debug("flycx[%s] restore skipped: %s", key, exc)
                _cxs[key] = cx
                _cx_ok[key] = True
            except Exception as exc:
                log.debug("flycx[%s] unavailable: %s", key, exc)
                _cxs[key] = None
                _cx_ok[key] = False
        return _cxs.get(key) if _cx_ok.get(key) else None


def flush_all(user_id: str | None = None) -> dict[str, int | bool]:
    """Force-write pending MB plastic deltas and CX sleep pressure for one identity.

    Call on clean shutdown, identity switch, or before process exit.
    Returns a small status dict for logging.
    """
    key = _norm_id(user_id)
    out: dict[str, int | bool] = {"mb_rows": 0, "cx": False}
    with _lock:
        store = _stores.get(key) or None
        if store is False or store is None:
            store = get_fly_store(key)
        mb = _mbs.get(key)
        cx = _cxs.get(key)
        if store is None:
            return out
        try:
            if mb is not None:
                out["mb_rows"] = int(store.flush_mb(mb) or 0)
        except Exception as exc:
            log.debug("fly flush mb[%s] failed: %s", key, exc)
        try:
            if cx is not None:
                store.flush_cx(float(getattr(cx, "sleep_pressure", 0.0)))
                out["cx"] = True
        except Exception as exc:
            log.debug("fly flush cx[%s] failed: %s", key, exc)
    return out


def flush_everything() -> dict[str, dict]:
    """Flush every cached identity (process shutdown)."""
    with _lock:
        keys = set(_stores) | set(_mbs) | set(_cxs)
    return {k: flush_all(k) for k in keys}
