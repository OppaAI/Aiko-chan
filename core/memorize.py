"""
core/memorize.py
Aiko's persistent memory — custom backend via sqlite-vec + fastembed + Ollama.
Abstracts all memory calls so think.py stays clean.

Memory lifecycle:
  - Every search() call increments access_count and updates last_accessed_at
    in the memories table, enabling Ebbinghaus-style exponential decay scoring.
  - dream() runs nightly (00:00) as a consolidation pass — no new vectors
    are written. It boosts salient memories, merges near-duplicates, then
    prunes decayed entries. Order matters: boost before prune so boosted
    memories aren't immediately swept.
  - cleanup() deletes memories below decay threshold, with grace period
    protection for newly created entries.
  - Decay logic lives in core/forget.py (pure math, no I/O).
  - Pinned memories (created via pin()) are permanently immune to decay
    cleanup and dream pruning. The pinned flag lives in the memories table.

Dream pass overview:
  1. Boost  — increment access_count on memories matching salience heuristics
              (keyword signals, high prior access, recency) so they survive decay.
  2. Merge  — cosine-similarity search per memory; near-duplicates above
              threshold are collapsed: keep the higher access_count copy,
              delete the redundant one to stay in sync.
              Pinned memories are never chosen as the loser in a merge.
  3. Prune  — standard cleanup() pass; runs after boost so newly protected
              memories aren't caught in the sweep.
              Pinned memories are skipped entirely.

Storage layout (single .db file):
  memories        — canonical record: id, user_id, memory, metadata
  memories_fts    — FTS5 virtual table for lexical search (BM25)
  memories_vec    — vec0 virtual table for KNN cosine search

Recall strategy — Reciprocal Rank Fusion (RRF):
  score = 1/(k + rank_knn) + 1/(k + rank_fts)
  k=60 (standard RRF constant — dampens outlier ranks)

  KNN catches semantic similarity ("I love cats" ↔ "I adore cats")
  FTS5 catches exact token matches ("Max", "birthday", proper nouns)
  RRF fuses both without weighting either arbitrarily.

Custom backend (replaces Qdrant + mem0):
  - _MemoryBackend handles LLM-based fact extraction, fastembed embeddings,
    and direct sqlite-vec upsert/search/delete/scroll.
  - Extraction prompt is tuned for small models: asks for a JSON array of
    atomic facts, strips <think> blocks for CoT models, skips trivial turns.
  - All schema fields (memory, user_id, created_at, access_count,
    last_accessed_at, pinned) are owned by this module — no hidden schema.

Dependencies:
  pip install sqlite-vec fastembed
"""
from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import sqlite3
import struct
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import sqlite_vec
from fastembed import TextEmbedding

from core.forget import compute_weighted_score, should_cleanup, CLEANUP_THRESHOLD
from core.log import get_logger

log = get_logger(__name__)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'mem_sqlite':  'Opening sqlite-vec memory store...',
    'mem_cleanup': 'Running memory cleanup...',
    'mem_ready':   'Memory backend ready',
}

# ── constants ─────────────────────────────────────────────────────────────────

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_DIMS  = 768
RRF_K       = 60          # standard RRF constant — dampens outlier ranks
KNN_LIMIT   = 20          # candidates fetched before RRF re-rank
FTS_LIMIT   = 20          # candidates fetched before RRF re-rank

USER_ID = os.getenv("USER_ID", "OppaAI")

# Cosine similarity threshold for near-duplicate detection during dream pass.
# 0.92 is conservative — only collapses near-identical phrasings.
# Lower (e.g. 0.85) catches more semantic duplicates but risks false merges.
DREAM_MERGE_THRESHOLD = float(os.getenv("DREAM_MERGE_THRESHOLD", 0.92))

# access_count boost applied to salient memories during dream pass.
DREAM_BOOST_AMOUNT = int(os.getenv("DREAM_BOOST_AMOUNT", 2))

# Salience keywords — memories containing these are boosted during dream pass.
_SALIENCE_KEYWORDS = frozenset([
    "name", "called", "likes", "loves", "hates", "dislikes", "always", "never",
    "important", "remember", "favourite", "favorite", "birthday", "works",
    "lives", "studying", "job", "afraid", "dream", "goal",
])

# Minimum conversation size (chars) worth sending to LLM for extraction.
# Skips trivial turns (greetings, one-word replies) to save inference time.
_EXTRACT_MIN_CHARS = int(os.getenv("MEMORY_EXTRACT_MIN_CHARS", 80))

# Extraction prompt — tuned for small models.
# Asks for a flat JSON array of atomic fact strings. Nothing else.
_EXTRACT_PROMPT = """\
Extract memorable facts about the user from this conversation.
Return ONLY a JSON array of short strings. Each string is one atomic fact.
Facts should be about the user's preferences, identity, life, or goals.
If nothing is worth remembering, return an empty array: []
Do NOT include assistant statements. Do NOT explain. No markdown.

Conversation:
{conversation}"""

def _sanitize_fts_query(query: str) -> str:
    """
    Strip characters that break FTS5 query parsing.
    FTS5 treats , " ( ) * ^ : - ' as syntax tokens — remove them all.
    Args:
        query (str): The query string to sanitize.
    Returns:
        str: The sanitized query string.
    """
    cleaned = re.sub(r'[^\w\s]', ' ', query)   # keep only word chars and spaces
    cleaned = ' '.join(cleaned.split())          # collapse whitespace
    return cleaned or "*"

# ── schema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- canonical memory records
CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    memory           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT NOT NULL DEFAULT 'never',
    pinned           INTEGER NOT NULL DEFAULT 0   -- 0=false 1=true
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);

-- FTS5 for lexical BM25 search — mirrors memories.memory column
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory,
    id UNINDEXED,
    content='memories',
    content_rowid='rowid'
);

-- vec0 for KNN cosine search — one embedding per memory
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dims}]
);

-- keep FTS5 in sync with memories via triggers
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, memory, id)
    VALUES (new.rowid, new.memory, new.id);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, memory, id)
    VALUES ('delete', old.rowid, old.memory, old.id);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF memory ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, memory, id)
    VALUES ('delete', old.rowid, old.memory, old.id);
    INSERT INTO memories_fts(rowid, memory, id)
    VALUES (new.rowid, new.memory, new.id);
END;
""".format(dims=EMBED_DIMS)


# ── sqlite payload helpers ────────────────────────────────────────────────────
# These replace the self._qdrant.retrieve / set_payload / scroll calls
# that AikoMemorize used to make directly against Qdrant.

def _sqlite_get_payload(conn: sqlite3.Connection, mem_id: str) -> dict:
    """
    Fetch the full memories row for a single id.
    Equivalent to qdrant.retrieve(ids=[mem_id], with_payload=True).
    Returns {} if not found.
    """
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()
    return dict(row) if row else {}


def _sqlite_set_payload(
    conn: sqlite3.Connection,
    mem_id: str,
    payload: dict,
) -> None:
    """
    Update arbitrary column subset for a single memory row.
    Equivalent to qdrant.set_payload(payload=..., points=[mem_id]).
    Only valid memories column names should be passed as keys.
    """
    if not payload:
        return
    cols = ", ".join(f"{k} = ?" for k in payload)
    vals = list(payload.values()) + [mem_id]
    conn.execute(f"UPDATE memories SET {cols} WHERE id = ?", vals)
    conn.commit()


def _sqlite_batch_get_payloads(
    conn: sqlite3.Connection,
    mem_ids: list[str],
) -> dict:
    """
    Batch-fetch access_count + last_accessed_at in a single query.
    Equivalent to the Qdrant _batch_get_payloads() round-trip.
    Returns {mem_id: (access_count, last_accessed_at)}.
    """
    if not mem_ids:
        return {}
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(mem_ids))
    rows = conn.execute(
        f"SELECT id, access_count, last_accessed_at FROM memories WHERE id IN ({placeholders})",
        mem_ids,
    ).fetchall()
    return {
        r["id"]: (r["access_count"] or 0, r["last_accessed_at"] or "never")
        for r in rows
    }


def _sqlite_get_vector(conn: sqlite3.Connection, mem_id: str) -> list[float]:
    """
    Retrieve the raw embedding for one memory from the vec0 table.
    Equivalent to qdrant.retrieve(ids=[mem_id], with_vectors=True).
    Returns [] on miss or error — callers should skip on empty.
    """
    row = conn.execute(
        "SELECT embedding FROM memories_vec WHERE id = ?", (mem_id,)
    ).fetchone()
    if row and row[0]:
        raw = row[0]
        n   = len(raw) // 4
        return list(struct.unpack(f"{n}f", raw))   # deserialise float32 blob
    return []


def _sqlite_is_pinned(conn: sqlite3.Connection, mem_id: str) -> bool:
    """
    Return True if memories.pinned == 1 for this id.
    Equivalent to checking qdrant payload pinned=True.
    Defaults to False on any error — safe: a miss leaves memory subject to
    normal decay rather than silently deleting it.
    """
    row = conn.execute(
        "SELECT pinned FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()
    return bool(row and row[0])


def _sqlite_knn_search(
    conn: sqlite3.Connection,
    vector: list[float],
    user_id: str,
    limit: int,
    threshold: Optional[float] = None,
) -> list[sqlite3.Row]:
    """
    KNN cosine search against memories_vec, filtered by user_id.
    Used by both _MemoryBackend.search() and _dream_merge() similarity lookup.
    When threshold is supplied, only rows with dist <= (1 - threshold) are
    returned (cosine distance = 1 - cosine similarity).
    """
    vec_blob = sqlite_vec.serialize_float32(vector)
    if threshold is not None:
        # convert similarity threshold to distance ceiling
        dist_ceil = 1.0 - threshold
        rows = conn.execute(
            """
            SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
            FROM memories_vec v
            JOIN memories m ON m.id = v.id
            WHERE m.user_id = ?
              AND vec_distance_cosine(v.embedding, ?) <= ?
            ORDER BY dist ASC
            LIMIT ?
            """,
            (vec_blob, user_id, vec_blob, dist_ceil, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
            FROM memories_vec v
            JOIN memories m ON m.id = v.id
            WHERE m.user_id = ?
            ORDER BY dist ASC
            LIMIT ?
            """,
            (vec_blob, user_id, limit),
        ).fetchall()
    return rows


# ── memory backend ────────────────────────────────────────────────────────────

class _MemoryBackend:
    """
    sqlite-vec + FTS5 + RRF replacement for the Qdrant _MemoryBackend.

    Public API is identical to the Qdrant version:
        add(), search(), get_all(), delete(), delete_all()

    AikoMemorize calls these and nothing else — all Qdrant-specific helpers
    (_batch_get_payloads, _get_vector, _is_pinned, set_payload equivalents)
    are re-implemented as module-level sqlite helper functions above.

    The .db file path is read from env var SQLITE_MEMORY_PATH,
    defaulting to ~/.aiko/memory.db.
    """

    def __init__(
        self,
        db_path:         str,
        ollama_base_url: str,
        model:           str,
        fastembed_cache: Optional[str] = None,
    ) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path  = db_path
        self._ollama   = ollama_base_url.rstrip("/")
        self._model    = model
        self._embedder = TextEmbedding(
            model_name=EMBED_MODEL,
            cache_dir=fastembed_cache,
        )
        self._conn = self._connect()
        self._apply_schema()

    # ── connection ────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open WAL-mode connection and load sqlite-vec extension."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        sqlite_vec.load(conn)   # registers vec0 + vec_distance etc.
        return conn

    def _apply_schema(self) -> None:
        """Create tables/triggers if they don't exist yet."""
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ── embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Embed a single string with fastembed. Returns a plain float list."""
        return list(self._embedder.embed([text]))[0].tolist()

    # ── extraction ────────────────────────────────────────────────────────────

    def _should_extract(self, messages: list[dict]) -> bool:
        """
        Return False for trivial turns not worth extracting.
        Skips LLM call entirely for greetings, one-word replies, etc.
        """
        total = sum(len(m.get("content") or "") for m in messages)
        return total >= _EXTRACT_MIN_CHARS

    def _extract_facts(self, messages: list[dict]) -> list[str]:
        """
        Send conversation to Ollama LLM and parse the returned JSON fact array.

        Strips <think>…</think> blocks before parsing (CoT models like
        Ministral/Qwen emit these and break JSON parsing otherwise).

        Returns a list of fact strings, or [] on any failure.
        """
        if not self._should_extract(messages):
            return []

        convo = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
            if (m.get("content") or "").strip()
        )
        prompt = _EXTRACT_PROMPT.format(conversation=convo)

        try:
            resp = httpx.post(
                f"{self._ollama}/api/chat",
                json={
                    "model":    self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {"temperature": 0.1, "num_predict": 512},
                },
                timeout=45,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()
        except Exception as e:
            log.warning(f"Extraction LLM call failed: {e}")
            return []

        # strip CoT think blocks before JSON parsing
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
        except json.JSONDecodeError:
            log.warning(f"Failed to parse extraction JSON: {raw[:200]!r}")

        return []

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str) -> list[str]:
        """
        Extract facts and persist each as a row in memories + memories_vec.
        FTS5 triggers handle memories_fts automatically.

        Returns list of new memory IDs (UUIDs). Empty list if nothing extracted
        or extraction fails — callers treat this as a no-op, not an error.
        """
        facts = self._extract_facts(messages)
        if not facts:
            return []

        now = datetime.now(timezone.utc).isoformat()
        ids = []

        for fact in facts:
            mem_id = str(uuid.uuid4())
            try:
                vector = self._embed(fact)

                # insert canonical record — FTS5 trigger fires automatically
                self._conn.execute(
                    """
                    INSERT INTO memories
                        (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
                    VALUES (?, ?, ?, ?, 0, 'never', 0)
                    """,
                    (mem_id, user_id, fact, now),
                )

                # insert embedding into vec0 table
                self._conn.execute(
                    "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                    (mem_id, sqlite_vec.serialize_float32(vector)),
                )

                self._conn.commit()
                ids.append(mem_id)
            except Exception as e:
                log.warning(f"Failed to upsert fact {mem_id!r}: {e}")
                self._conn.rollback()

        return ids

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: str, user_id: str, limit: int = 5) -> list[dict]:
        """
        KNN + FTS5 → RRF fusion search.

        1. KNN: top-KNN_LIMIT by cosine distance from memories_vec
           (user_id filter applied in JOIN — vec0 doesn't support standalone WHERE)
        2. FTS5: top-FTS_LIMIT by BM25 from memories_fts
           (user_id filter in JOIN)
        3. RRF: score = 1/(k+rank_knn) + 1/(k+rank_fts)
           ids appearing in only one source get 0 for the missing rank
        4. Return top `limit` by RRF score as payload dicts

        Access tracking (access_count, last_accessed_at) is handled by
        AikoMemorize.search() — not here, matching original contract.
        """
        vector = self._embed(query)

        # ── KNN candidates ────────────────────────────────────────────────────
        knn_rows = _sqlite_knn_search(self._conn, vector, user_id, KNN_LIMIT)

        # rank_knn: {id: 1-based rank}
        rank_knn = {row["id"]: i + 1 for i, row in enumerate(knn_rows)}

        # ── FTS5 candidates ───────────────────────────────────────────────────
        fts_rows = self._conn.execute(
            """
            SELECT f.id
            FROM memories_fts f
            JOIN memories m ON m.id = f.id
            WHERE memories_fts MATCH ?
            AND m.user_id = ?
            ORDER BY rank
            LIMIT ?
            """,
            (_sanitize_fts_query(query), user_id, FTS_LIMIT),
        ).fetchall()

        rank_fts = {row["id"]: i + 1 for i, row in enumerate(fts_rows)}

        # ── RRF fusion ────────────────────────────────────────────────────────
        all_ids = set(rank_knn) | set(rank_fts)
        if not all_ids:
            return []

        def rrf(mem_id: str) -> float:
            knn = rank_knn.get(mem_id, 0)
            fts = rank_fts.get(mem_id, 0)
            score = 0.0
            if knn:
                score += 1.0 / (RRF_K + knn)
            if fts:
                score += 1.0 / (RRF_K + fts)
            return score

        ranked = sorted(all_ids, key=rrf, reverse=True)[:limit]

        # ── fetch full payloads ───────────────────────────────────────────────
        placeholders = ",".join("?" * len(ranked))
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            ranked,
        ).fetchall()

        # preserve RRF order
        order       = {mid: i for i, mid in enumerate(ranked)}
        rows_sorted = sorted(rows, key=lambda r: order.get(r["id"], 999))

        return [dict(r) for r in rows_sorted]

    def get_all(self, user_id: str) -> list[dict]:
        """Return all memory records for a user."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── delete ────────────────────────────────────────────────────────────────

    def delete(self, memory_id: str) -> None:
        """
        Delete a memory from all three tables.
        FTS5 trigger on memories handles memories_fts automatically.
        memories_vec must be deleted explicitly (no FK cascade on virtual tables).
        """
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
        self._conn.commit()

    def delete_all(self, user_id: str) -> None:
        """Delete every memory for a user from all three tables."""
        ids = [
            r["id"] for r in self._conn.execute(
                "SELECT id FROM memories WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})", ids
        )
        self._conn.execute(
            f"DELETE FROM memories_vec WHERE id IN ({placeholders})", ids
        )
        self._conn.commit()


# ── memorize ──────────────────────────────────────────────────────────────────

class AikoMemorize:
    """
    Persistent memory with Ebbinghaus decay lifecycle and nightly dream() pass.

    Uses a custom _MemoryBackend (Ollama extraction + fastembed + sqlite-vec)
    instead of Qdrant + mem0. Public API and all lifecycle behaviour are unchanged.

    Boot sequence (called by wakeup.py in order):
        memorize = AikoMemorize()   # opens sqlite-vec store + loads fastembed
        memorize.cleanup()          # prune decayed memories on startup

    Access tracking:
        Every search() call updates the memories table (access_count,
        last_accessed_at) so the decay formula has fresh data.

    Pinned memories:
        Created via pin() — the pinned=1 column flag makes them
        immune to cleanup(), dream prune, and dream merge (as the loser).

    Dream pass (call nightly at 00:00):
        1. Boost salient memories' access_count so they survive decay.
        2. Merge near-duplicate vectors — keeps higher-access copy.
           Pinned memories are never deleted as a merge loser.
        3. Prune decayed memories via cleanup().
           Pinned memories are skipped entirely.

    Cleanup:
        Also available standalone — deletes memories below decay threshold,
        with grace period protection for newly created entries.
        Pinned memories are always kept regardless of score.
    """

    def __init__(self, silent: bool = False) -> None:
        db_path = os.getenv(
            "SQLITE_MEMORY_PATH",
            str(Path.home() / ".aiko" / "memory.db"),
        )

        if not silent:
            log.info("Opening sqlite-vec memory store...")

        self._mem = _MemoryBackend(
            db_path=db_path,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("MEM0_MODEL") or os.getenv("OLLAMA_MODEL"),
            fastembed_cache=os.getenv("FASTEMBED_CACHE_PATH"),
        )
        # direct connection handle for payload operations (access tracking,
        # pinning, vector retrieval) that bypass the backend abstraction
        self._conn = self._mem._conn

        if not silent:
            log.info("Ready.")

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str = USER_ID) -> bool:
        """
        Store a conversation turn (or batch) into long-term memory.

        Runs synchronously — LLM extraction completes before returning.
        Callers that need non-blocking writes should enqueue via their own
        worker (e.g. think.py's _mem_write_loop).

        Returns True on success, False on failure so callers can log/alert.
        """
        try:
            t       = time.perf_counter()
            ids     = self._mem.add(messages, user_id=user_id)
            elapsed = time.perf_counter() - t
            if ids:
                log.info(f"Saved {len(ids)} memories in {elapsed:.2f}s")
            else:
                log.debug(f"No facts extracted ({elapsed:.2f}s) — nothing saved.")
            return True
        except Exception as e:
            log.error(f"Save failed: {e}")
            return False

    def pin(self, messages: list[dict], user_id: str = USER_ID) -> bool:
        """
        Store messages and immediately mark all resulting memories as pinned.

        Pinned memories are permanently immune to:
          - decay cleanup (cleanup() skips them regardless of score)
          - dream pruning (dream() prune stage skips them)
          - dream merging (never chosen as the loser in a duplicate collapse)

        Uses before/after snapshot of get_all() to identify new memory IDs,
        then sets pinned=1 in their memories row.

        Returns True on success, False on any failure (check logs).
        """
        try:
            before  = {str(m["id"]) for m in self.get_all(user_id=user_id)}
            ok      = self.add(messages, user_id=user_id)
            if not ok:
                return False
            after   = {str(m["id"]) for m in self.get_all(user_id=user_id)}
            pin_ids = list(after - before)

            if not pin_ids:
                # fallback: search for the closest matching memories
                query = "\n".join(
                    (m.get("content") or "").strip()
                    for m in messages
                    if (m.get("content") or "").strip()
                )
                pin_ids = [
                    str(m.get("id"))
                    for m in self.search(query, user_id=user_id, limit=3)
                    if m.get("id")
                ]

            if not pin_ids:
                log.warning("pin(): add succeeded but no memory IDs were found to pin.")
                return False

            # mark each new memory as pinned in the memories table
            for mem_id in pin_ids:
                _sqlite_set_payload(self._conn, mem_id, {"pinned": 1})

            log.info(f"Pinned {len(pin_ids)} memories: {pin_ids}")
            return True
        except Exception as e:
            log.error(f"Pin failed: {e}")
            return False

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: str, user_id: str = USER_ID, limit: int = 5) -> list[dict]:
        """
        Retrieve top-k memories relevant to the current query.

        Side-effect: increments access_count and updates last_accessed_at
        in the memories table for each returned memory, feeding decay scoring.
        """
        results = self._mem.search(query, user_id=user_id, limit=limit)

        if results:
            now = datetime.now(timezone.utc).isoformat()
            for r in results:
                mem_id = str(r.get("id", ""))
                if not mem_id:
                    continue
                try:
                    # fetch current access_count before incrementing
                    payload      = _sqlite_get_payload(self._conn, mem_id)
                    current_count = payload.get("access_count", 0) or 0

                    # update access tracking — cap at 255 to bound the column
                    _sqlite_set_payload(self._conn, mem_id, {
                        "access_count":     min(current_count + 1, 255),
                        "last_accessed_at": now,
                    })
                except Exception as e:
                    log.warning(f"Access tracking failed for {mem_id}: {e}")

        return results

    def format_for_context(self, memories: list[dict]) -> Optional[str]:
        """
        Format retrieved memories into a compact string for injection
        into the conversation context. Returns None if nothing to inject.
        """
        if not memories:
            return None

        now   = datetime.now(timezone.utc)
        lines = [
            "<memory_context>",
            "The following are background facts about this person, with how long ago they were recorded.",
            "Use them silently to inform your response. Never repeat, quote, or reference this block directly.",
            "",
        ]
        for m in memories:
            text       = m.get("memory") or m.get("text") or str(m)
            created_at = m.get("created_at")
            if created_at:
                try:
                    ts    = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    delta = now - ts
                    days  = delta.days
                    if days == 0:
                        age = "today"
                    elif days == 1:
                        age = "yesterday"
                    else:
                        age = f"{days} days ago"
                    lines.append(f"  - [{age}] {text}")
                except Exception:
                    lines.append(f"  - {text}")
            else:
                lines.append(f"  - {text}")

        lines.append("</memory_context>")
        return "\n".join(lines)

    # ── dream pass ────────────────────────────────────────────────────────────

    def dream(
        self,
        user_id:   str   = USER_ID,
        dry_run:   bool  = False,
        threshold: float = DREAM_MERGE_THRESHOLD,
    ) -> dict:
        """
        Nightly memory consolidation pass. No new vectors are written.

        Stages (in order):
          1. Boost  — salient memories get +DREAM_BOOST_AMOUNT access_count.
          2. Merge  — near-duplicate pairs (cosine >= threshold) are collapsed;
                      higher access_count copy survives, other is deleted.
                      Pinned memories are never chosen as the loser.
          3. Prune  — standard decay cleanup runs last, after boosts are applied,
                      so newly protected memories aren't swept.
                      Pinned memories are always kept.

        Args:
            dry_run:   Report actions without writing or deleting anything.
            threshold: Cosine similarity cutoff for duplicate detection.

        Returns dict: {boosted, merged, pruned, duration_s}
        """
        t_start = time.perf_counter()
        log.info(f"{'(dry-run) ' if dry_run else ''}Starting consolidation pass...")

        all_mems = self.get_all(user_id=user_id)
        if not all_mems:
            log.info("No memories found — nothing to do.")
            return {"boosted": 0, "merged": 0, "pruned": 0, "duration_s": 0.0}

        mem_ids     = [str(m.get("id", "")) for m in all_mems if m.get("id")]
        payload_map = self._batch_get_payloads(mem_ids)

        boosted      = self._dream_boost(all_mems, payload_map, dry_run=dry_run)
        merged       = self._dream_merge(mem_ids, user_id=user_id, threshold=threshold, dry_run=dry_run)
        prune_result = self.cleanup(user_id=user_id, dry_run=dry_run)
        pruned       = prune_result.get("deleted", 0)

        duration = round(time.perf_counter() - t_start, 2)
        log.info(
            f"{'(dry-run) ' if dry_run else ''}"
            f"Done — boosted={boosted}, merged={merged}, pruned={pruned}, "
            f"duration={duration}s"
        )
        return {"boosted": boosted, "merged": merged, "pruned": pruned, "duration_s": duration}

    def _dream_boost(
        self,
        all_mems:    list[dict],
        payload_map: dict,
        dry_run:     bool = False,
    ) -> int:
        """
        Increment access_count on memories that match salience heuristics.

        Salience criteria (any one triggers boost):
          - Text contains a keyword from _SALIENCE_KEYWORDS
          - access_count >= 3 (user has repeatedly surfaced this memory)
          - created_at within the last 7 days (recency grace boost)

        Pinned memories pass through the boost unchanged — they don't need it.

        Returns count of memories boosted.
        """
        now     = datetime.now(timezone.utc)
        boosted = 0

        for m in all_mems:
            mem_id = str(m.get("id", ""))
            if not mem_id:
                continue

            if _sqlite_is_pinned(self._conn, mem_id):
                continue

            text     = (m.get("memory") or "").lower()
            ac, _la  = payload_map.get(mem_id, (0, "never"))

            # check recency — grace boost for memories within 7 days
            is_recent  = False
            created_at = m.get("created_at", "")
            if created_at:
                try:
                    ts        = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    is_recent = (now - ts).days <= 7
                except Exception:
                    pass

            is_salient = (
                any(kw in text for kw in _SALIENCE_KEYWORDS)
                or ac >= 3
                or is_recent
            )

            if not is_salient:
                continue

            if not dry_run:
                try:
                    # increment access_count, capped at 255
                    _sqlite_set_payload(self._conn, mem_id, {
                        "access_count": min(ac + DREAM_BOOST_AMOUNT, 255)
                    })
                except Exception as e:
                    log.warning(f"Boost failed for {mem_id}: {e}")
                    continue

            boosted += 1

        if boosted:
            log.info(f"{'(dry-run) ' if dry_run else ''}Boosted {boosted} memories.")
        return boosted

    def _dream_merge(
        self,
        mem_ids:   list[str],
        user_id:   str,
        threshold: float = DREAM_MERGE_THRESHOLD,
        dry_run:   bool  = False,
    ) -> int:
        """
        Detect and collapse near-duplicate memory vectors.

        For each memory, performs a KNN cosine search filtered by user_id and
        threshold. When a duplicate pair is found, the lower access_count copy
        is deleted. Tracks already-deleted IDs to avoid double-deletes in the
        same pass.

        Pinned memories are skipped as query origins and are never chosen as
        the loser in _resolve_duplicate() — a pinned memory always survives.

        Returns count of memories deleted as duplicates.
        """
        deleted_ids: set[str] = set()
        merged = 0

        for mem_id in mem_ids:
            if mem_id in deleted_ids:
                continue

            if _sqlite_is_pinned(self._conn, mem_id):
                continue

            # retrieve the embedding for this memory to use as query vector
            vector = _sqlite_get_vector(self._conn, mem_id)
            if not vector:
                continue

            try:
                # search for neighbors above the similarity threshold
                neighbor_rows = _sqlite_knn_search(
                    self._conn, vector, user_id, limit=4, threshold=threshold
                )
            except Exception as e:
                log.warning(f"Similarity search failed for {mem_id}: {e}")
                continue

            for row in neighbor_rows:
                neighbor_id = row["id"]
                if neighbor_id == mem_id:
                    continue
                if neighbor_id in deleted_ids:
                    continue

                # cosine distance → similarity score for logging
                similarity = 1.0 - row["dist"]

                n_merged = self._resolve_duplicate(
                    mem_id, neighbor_id, similarity, dry_run=dry_run
                )
                if n_merged:
                    deleted_ids.add(neighbor_id)
                    merged += 1

        if merged:
            log.info(f"{'(dry-run) ' if dry_run else ''}Merged {merged} duplicate memories.")
        return merged

    def _resolve_duplicate(
        self,
        id_a:    str,
        id_b:    str,
        score:   float,
        dry_run: bool = False,
    ) -> bool:
        """
        Compare two near-duplicate memories and delete the weaker one.

        Keeps the copy with higher access_count. On a tie, keeps id_a
        (the query origin) and deletes id_b.

        If either memory is pinned, the merge is aborted — a pinned memory
        is never deleted regardless of access_count comparison.

        Returns True if a deletion occurred (or would occur in dry_run).
        """
        if _sqlite_is_pinned(self._conn, id_a) or _sqlite_is_pinned(self._conn, id_b):
            log.info(f"Skipping merge: one or both of ({id_a}, {id_b}) is pinned.")
            return False

        payload_map = self._batch_get_payloads([id_a, id_b])
        ac_a, _     = payload_map.get(id_a, (0, "never"))
        ac_b, _     = payload_map.get(id_b, (0, "never"))
        loser       = id_b if ac_a >= ac_b else id_a   # tie goes to id_a (query origin)

        if dry_run:
            log.info(
                f"(dry-run) Would merge: score={score:.3f} "
                f"ac_a={ac_a} ac_b={ac_b} → delete {loser}"
            )
            return True

        try:
            self._mem.delete(memory_id=loser)
            log.info(
                f"Merged duplicate (score={score:.3f}, "
                f"ac_a={ac_a}, ac_b={ac_b}) → deleted {loser}"
            )
            return True
        except Exception as e:
            log.warning(f"Merge delete failed for {loser}: {e}")
            return False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def cleanup(
        self,
        user_id:   str   = USER_ID,
        threshold: float = CLEANUP_THRESHOLD,
        dry_run:   bool  = False,
    ) -> dict:
        """
        Prune decayed memories below threshold score.

        Fetches all memories, batch-retrieves payloads (single round-trip),
        evaluates decay score via should_cleanup(), and deletes candidates
        directly to keep the vector store in sync.

        Grace period (14 days) protects newly created memories from deletion
        even if they score below threshold.

        Pinned memories are unconditionally kept — the pinned flag overrides
        all decay scoring.

        Args:
            threshold: Override decay threshold (default: CLEANUP_THRESHOLD = 0.05).
            dry_run:   If True, report candidates without deleting.

        Returns dict with counts: deleted, kept, failed, candidates (dry_run only).
        """
        all_mems = self.get_all(user_id=user_id)
        if not all_mems:
            return {"deleted": 0, "kept": 0, "failed": 0}

        mem_ids     = [str(m.get("id", "")) for m in all_mems if m.get("id")]
        payload_map = self._batch_get_payloads(mem_ids)

        candidates = []
        kept       = 0

        for m in all_mems:
            mem_id     = str(m.get("id", ""))
            ac, la     = payload_map.get(mem_id, (0, "never"))
            created_at = m.get("created_at", "")

            # pinned memories are immune to all decay paths
            if _sqlite_is_pinned(self._conn, mem_id):
                kept += 1
                continue

            if should_cleanup(ac, la, created_at):
                w = compute_weighted_score(ac, la)
                candidates.append({
                    "id":               mem_id,
                    "memory":           m.get("memory", "")[:120],
                    "access_count":     ac,
                    "weighted_score":   round(w, 4),
                    "last_accessed_at": la,
                })
            else:
                kept += 1

        # sort by weakest score first so logs are most-pruneable-first
        candidates.sort(key=lambda x: x["weighted_score"])

        if dry_run:
            log.info(f"Dry run: {len(candidates)} candidates for deletion, {kept} kept.")
            return {"deleted": 0, "kept": kept, "failed": 0, "candidates": candidates}

        deleted = []
        failed  = []
        for c in candidates:
            try:
                self._mem.delete(memory_id=c["id"])
                deleted.append(c["id"])
            except Exception as e:
                failed.append({"id": c["id"], "error": str(e)})

        log.info(f"Cleanup: deleted={len(deleted)}, kept={kept}, failed={len(failed)}")
        return {"deleted": len(deleted), "kept": kept, "failed": len(failed)}

    # ── debug ─────────────────────────────────────────────────────────────────

    def get_all(self, user_id: str = USER_ID) -> list[dict]:
        """Return all stored memories for a user (for debugging / dream pass)."""
        return self._mem.get_all(user_id=user_id)

    def clear(self, user_id: str = USER_ID) -> None:
        """Wipe all memories for a user. Use carefully."""
        self._mem.delete_all(user_id=user_id)
        log.info(f"Cleared all memories for user '{user_id}'.")

    # ── internal ──────────────────────────────────────────────────────────────

    def _batch_get_payloads(self, mem_ids: list[str]) -> dict:
        """
        Batch retrieve access_count + last_accessed_at in a single query.
        Single round-trip — eliminates N+1 query problem for cleanup/stats.
        Returns dict: {mem_id: (access_count, last_accessed_at)}
        """
        return _sqlite_batch_get_payloads(self._conn, mem_ids)

    def _get_vector(self, mem_id: str) -> list[float]:
        """
        Retrieve the raw embedding vector for a single memory.
        Used by _dream_merge() to run similarity searches.
        Returns empty list on failure — callers should skip on empty.
        """
        return _sqlite_get_vector(self._conn, mem_id)

    def _is_pinned(self, mem_id: str) -> bool:
        """
        Return True if memories.pinned == 1 for this id.

        Used as a guard in cleanup(), _dream_merge(), and _resolve_duplicate()
        to make pinned memories permanently immune to all deletion paths.
        Defaults to False on any error — safe because a False miss at worst
        leaves a memory subject to normal decay, not silently deletes it.
        """
        return _sqlite_is_pinned(self._conn, mem_id)
