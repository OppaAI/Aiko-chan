"""SQLite persistence for the living tissue (plastic overlays, not base weights).

What persists (and why only this):
  - MB KC->MBON plastic deltas (nonzero only): learned associations must
    survive restarts, or praise never compounds across days.
  - Compass sleep_pressure scalar: sleep need is homeostatic (persists in
    the fly too). The compass bump itself is NOT stored — attention starts
    fresh every boot, which is the correct behavior.

Design: synchronous debounced writes (every FLY_PLASTICITY_EVERY events,
default 25; a few-ms SQLite write of tiny tables). Base connectome NPZ
stays read-only and version-pinned: on shape mismatch stored deltas are
ignored, never force-fit. stdlib sqlite3 + numpy only.

Identity isolation: each PlasticityStore instance is bound to one identity
(path or logical key). Multi-user deployments must use one store per user
via cognition.fly_registry — never share a single process-global DB.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2

_DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "fly_plasticity.db"


def _db_path() -> Path:
    try:
        raw = (os.getenv("FLY_PLASTICITY_DB") or "").strip()
    except Exception:
        raw = ""
    return Path(raw).expanduser() if raw else _DEFAULT_DB


def _save_every() -> int:
    try:
        return max(1, int(os.getenv("FLY_PLASTICITY_EVERY", "25")))
    except Exception:
        return 25


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class PlasticityStore:
    """Debounced store for one identity (or the legacy single-user path).

    Prefer constructing via cognition.fly_registry.get_fly_store(user_id) so
    the SQLite path is identity-scoped under the user state directory.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        identity: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser() if path else _db_path()
        self.identity = (identity or "default").strip() or "default"
        self._mb_events = 0
        self._cx_events = 0

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS mb_plastic
               (pre INTEGER NOT NULL, post INTEGER NOT NULL, delta REAL NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY (pre, post))"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS cx_state
               (key TEXT PRIMARY KEY, value REAL NOT NULL, updated_at TEXT NOT NULL)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS meta
               (key TEXT PRIMARY KEY, value TEXT NOT NULL)"""
        )
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('identity', ?)",
            (self.identity,),
        )
        conn.commit()
        return conn

    def save_mb(self, pre, post, delta) -> int:
        pre = np.asarray(pre).reshape(-1)
        post = np.asarray(post).reshape(-1)
        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        mask = np.abs(delta) > 1e-9
        now = _now()
        rows = [
            (int(a), int(b), float(d), now)
            for a, b, d in zip(
                pre[mask].tolist(), post[mask].tolist(), delta[mask].tolist()
            )
        ]
        conn = self._connect()
        try:
            conn.execute("DELETE FROM mb_plastic")
            conn.executemany(
                "INSERT INTO mb_plastic(pre, post, delta, updated_at) VALUES (?,?,?,?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        self._mb_events = 0
        return len(rows)

    def load_mb(self, n_pre: int, n_post: int) -> dict | None:
        if not self.path.exists():
            return None
        conn = self._connect()
        try:
            rows = conn.execute("SELECT pre, post, delta FROM mb_plastic").fetchall()
        finally:
            conn.close()
        out = {}
        for a, b, d in rows:
            if 0 <= a < n_pre and 0 <= b < n_post:
                out[(int(a), int(b))] = float(d)
            else:
                return None
        return out

    def save_if_due_mb(self, mb) -> int | None:
        self._mb_events += 1
        if self._mb_events < _save_every():
            return None
        return self.flush_mb(mb)

    def flush_mb(self, mb) -> int:
        ind, idx, _ = mb._kcm
        counts = np.diff(ind).astype(np.int64)
        pre = np.repeat(np.arange(len(ind) - 1, dtype=np.int64), counts)
        return self.save_mb(pre, idx, mb._plastic)

    def save_cx(self, sleep_pressure: float) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cx_state(key, value, updated_at) VALUES ('sleep_pressure', ?, ?)",
                (float(sleep_pressure), _now()),
            )
            conn.commit()
        finally:
            conn.close()
        self._cx_events = 0

    def load_cx(self) -> float | None:
        if not self.path.exists():
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT value FROM cx_state WHERE key='sleep_pressure'"
            ).fetchone()
        finally:
            conn.close()
        return float(row[0]) if row else None

    def save_if_due_cx(self, sleep_pressure: float) -> bool:
        self._cx_events += 1
        if self._cx_events < _save_every():
            return False
        self.save_cx(sleep_pressure)
        return True

    def flush_cx(self, sleep_pressure: float) -> None:
        self.save_cx(sleep_pressure)


def apply_mb(mb, deltas: dict | None) -> int:
    if not deltas:
        return 0
    ind, idx, _ = mb._kcm
    n = 0
    for i in range(len(ind) - 1):
        s, e = ind[i], ind[i + 1]
        for p, t in zip(range(s, e), idx[s:e].tolist()):
            d = deltas.get((i, int(t)))
            if d:
                mb._plastic[p] = float(d)
                n += 1
    return n
