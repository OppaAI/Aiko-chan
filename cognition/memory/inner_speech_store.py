"""Inner-speech store: SQLite/sqlite-vec persistent backing for Aiko's inner speech.

This is the third tier of the mind, next to the subconscious (SubliminalLayer,
felt but never verbalized) and speech (memory.db, verbalized outward):

- subconscious: felt, never verbalized — affect, impulses, priming
- inner speech (THIS store): verbal but inward — reflections, improvement
  notes, the day's thinking
- speech: verbal and outward — grounded facts she is willing to say

CONTAMINATION BOUNDARY: rows in this store are recalled ONLY into the
<inner_speech> prompt block, never into chat context. The chat-context
builder must never import this module. The physical file boundary (a
separate .db next to memory.db) enforces what convention alone would not.

Schema mirrors cognition/memory/schema.py: a base table, FTS5, and a vec0
vector table. Recall is vector KNN re-ranked by recency — thoughts evaporate
like human thoughts do (half-life ~2 days); the midnight dream distills what
matters into memory.db before they fade.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

log = logging.getLogger("aiko.inner_speech_store")

EMBED_DIMS = int(os.getenv("EMBED_DIMS", "640"))

# Thoughts fade: a 2-day half-life means a thought from a week ago scores
# ~9% of its raw similarity. Facts persist in memory.db; thoughts evaporate.
RECENCY_HALF_LIFE_HOURS = float(os.getenv("INNER_SPEECH_HALF_LIFE_H", "48"))

_DDL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS thoughts (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    text             TEXT NOT NULL,
    kind             TEXT NOT NULL DEFAULT 'reflection',
    source           TEXT NOT NULL DEFAULT 'turn',
    created_at       TEXT NOT NULL,
    consolidated     INTEGER NOT NULL DEFAULT 0,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT NOT NULL DEFAULT 'never'
);

CREATE INDEX IF NOT EXISTS idx_thoughts_user ON thoughts(user_id);
CREATE INDEX IF NOT EXISTS idx_thoughts_created ON thoughts(created_at);
CREATE INDEX IF NOT EXISTS idx_thoughts_consolidated ON thoughts(consolidated);

CREATE VIRTUAL TABLE IF NOT EXISTS thoughts_fts USING fts5(
    text,
    id UNINDEXED,
    content='thoughts',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS thoughts_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dims}] distance_metric=cosine
);

CREATE TRIGGER IF NOT EXISTS thoughts_ai AFTER INSERT ON thoughts BEGIN
    INSERT INTO thoughts_fts(rowid, text, id)
    VALUES (new.rowid, new.text, new.id);
END;

CREATE TRIGGER IF NOT EXISTS thoughts_ad AFTER DELETE ON thoughts BEGIN
    INSERT INTO thoughts_fts(thoughts_fts, rowid, text, id)
    VALUES ('delete', old.rowid, old.text, old.id);
END;

CREATE TRIGGER IF NOT EXISTS thoughts_au AFTER UPDATE OF text ON thoughts BEGIN
    INSERT INTO thoughts_fts(thoughts_fts, rowid, text, id)
    VALUES ('delete', old.rowid, old.text, old.id);
    INSERT INTO thoughts_fts(rowid, text, id)
    VALUES (new.rowid, new.text, new.id);
END;
"""


def _ddl(dims: int = EMBED_DIMS) -> str:
    return _DDL_TEMPLATE.format(dims=dims)


def _db_path(user_id: str | None = None) -> str:
    from cognition.memory.vecstore import resolve_user_db_path

    return str(resolve_user_db_path("memory/inner_speech.db", user_id=user_id))


class InnerSpeechStore:
    """Persistent inner-thought store. Inner recall only — never chat context."""

    def __init__(self, user_id: str | None = None, embedder=None) -> None:
        self._lock = threading.RLock()
        self._user_id = user_id or "oppa"
        self._path = _db_path(self._user_id)
        self._embedder = embedder  # lazy HarrierEmbedder if None
        self._conn = self._connect()

    # ── setup ──────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            import sqlite_vec  # type: ignore

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception as e:
            log.warning("inner_speech: sqlite-vec unavailable: %s", e)
        with conn:
            conn.executescript(_ddl())
        return conn

    def _embed(self, text: str) -> list[float] | None:
        try:
            if self._embedder is None:
                from cognition.memory.vecstore import HarrierEmbedder

                self._embedder = HarrierEmbedder()
            vec = self._embedder.embed([text])
            import numpy as _np

            arr = _np.asarray(vec[0] if hasattr(vec, "__len__") else vec)
            return [float(x) for x in arr.ravel().tolist()]
        except Exception as e:
            log.debug("inner_speech: embed failed: %s", e)
            return None

    # ── writes ─────────────────────────────────────────────────────────

    def write(
        self,
        text: str,
        *,
        kind: str = "reflection",
        source: str = "turn",
    ) -> str | None:
        """Persist one inner utterance. Returns the row id, or None on failure."""
        text = (text or "").strip()
        if not text:
            return None
        tid = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO thoughts(id, user_id, text, kind, source, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (tid, self._user_id, text, kind, source, now),
                )
                vec = self._embed(text)
                if vec:
                    import sqlite_vec  # type: ignore

                    blob = sqlite_vec.serialize_float32(vec)
                    self._conn.execute(
                        "INSERT INTO thoughts_vec(id, embedding) VALUES (?, ?)",
                        (tid, blob),
                    )
            return tid
        except Exception as e:
            log.warning("inner_speech: write failed: %s", e)
            return None

    # ── recall (inner only) ────────────────────────────────────────────

    def recall(self, query: str, *, k: int = 3) -> list[dict]:
        """Top-k past thoughts by similarity × recency decay.

        Returns dicts {id, text, kind, source, created_at, score}.
        NEVER feed these into chat context — inner prompt block only.
        """
        query = (query or "").strip()
        if not query:
            return []
        vec = self._embed(query)
        if not vec:
            return self._recent_fallback(k)
        try:
            import sqlite_vec  # type: ignore

            blob = sqlite_vec.serialize_float32(vec)
            overscan = max(k * 4, 12)
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT t.id, t.text, t.kind, t.source, t.created_at,
                           t.consolidated, v.distance AS dist
                    FROM thoughts_vec v
                    JOIN thoughts t ON t.id = v.id
                    WHERE v.embedding MATCH ?
                      AND v.k = ?
                      AND t.user_id = ?
                    ORDER BY v.distance ASC
                    LIMIT ?
                    """,
                    (blob, overscan, self._user_id, k),
                ).fetchall()
        except Exception as e:
            log.debug("inner_speech: knn recall failed: %s", e)
            return self._recent_fallback(k)

        now = time.time()
        scored = []
        for r in rows:
            try:
                created = datetime.fromisoformat(r["created_at"]).timestamp()
            except Exception:
                created = now
            age_h = max(0.0, (now - created) / 3600.0)
            decay = 0.5 ** (age_h / RECENCY_HALF_LIFE_HOURS)
            similarity = max(0.0, 1.0 - float(r["dist"] or 0.0))
            # Consolidated thoughts already live on in memory.db as facts;
            # down-weight the raw row so she stops "remembering thinking" it.
            if r["consolidated"]:
                similarity *= 0.4
            scored.append((similarity * decay, dict(r)))
        scored.sort(key=lambda p: p[0], reverse=True)
        out = []
        for score, r in scored[:k]:
            r["score"] = round(score, 4)
            out.append(r)
        self._touch([r["id"] for r in out])
        return out

    def _recent_fallback(self, k: int) -> list[dict]:
        """When embeddings are unavailable: most recent thoughts, decayed."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, text, kind, source, created_at, consolidated"
                    " FROM thoughts WHERE user_id = ?"
                    " ORDER BY created_at DESC LIMIT ?",
                    (self._user_id, k),
                ).fetchall()
            now = time.time()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    age_h = max(0.0, (now - datetime.fromisoformat(d["created_at"]).timestamp()) / 3600.0)
                except Exception:
                    age_h = 0.0
                d["score"] = round(0.5 ** (age_h / RECENCY_HALF_LIFE_HOURS), 4)
                out.append(d)
            return out
        except Exception:
            return []

    def _touch(self, ids: list[str]) -> None:
        if not ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock, self._conn:
                for tid in ids:
                    self._conn.execute(
                        "UPDATE thoughts SET access_count = access_count + 1,"
                        " last_accessed_at = ? WHERE id = ?",
                        (now, tid),
                    )
        except Exception:
            pass

    # ── midnight join ──────────────────────────────────────────────────

    def unconsolidated_since(self, since_iso: str) -> list[dict]:
        """Rows since `since_iso` not yet folded into the dream. For dream.py."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, text, kind, source, created_at FROM thoughts"
                    " WHERE user_id = ? AND consolidated = 0 AND created_at >= ?"
                    " ORDER BY created_at ASC",
                    (self._user_id, since_iso),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            log.warning("inner_speech: unconsolidated_since failed: %s", e)
            return []

    def mark_consolidated(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            with self._lock, self._conn:
                for tid in ids:
                    self._conn.execute(
                        "UPDATE thoughts SET consolidated = 1 WHERE id = ?", (tid,)
                    )
        except Exception as e:
            log.warning("inner_speech: mark_consolidated failed: %s", e)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
