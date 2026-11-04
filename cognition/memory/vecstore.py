"""
memory/vecstore.py

Shared database helpers and text embedder for Aiko's local RAG stores.

Memory, learned knowledge, and experience all use local SQLite/sqlite-vec
stores, optionally encrypted through system.secure. Keep common connection,
schema, FTS query, and ranking helpers here so store modules own their domain
schemas/queries but not repeated database bootstrap code.

The HarrierEmbedder class provides HTTP-based text embeddings via llama-server.
"""
from __future__ import annotations

from cognition.memory.redaction import redact_text, redact_data

import os
import json
import tempfile
import contextlib
import uuid
from itertools import islice
import re
import sqlite3
import struct
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Generator, Iterable

import numpy as np
import requests

from system.secure import connect_sqlite
from system.config import env_float, env_int, env_str
from system.userspace import current_user_id, user_state_path
from system.log import get_logger

log = get_logger(__name__)

# Fly antennal-lobe layer: divisive normalization on every embedding.
# Applied inside _embed_texts (the single funnel for store + query paths)
# so geometry stays consistent. Modes off | shadow | live (default off);
# shadow computes + logs gain stats and returns raw vectors.
def _flyal_mode() -> str:
    try:
        return env_str("MEMORY_FLYAL_MODE", "off").strip().lower()
    except Exception:
        return "off"


_FLYAL = None
_FLYAL_OK = None


def _flyal():
    global _FLYAL, _FLYAL_OK
    if _FLYAL_OK is None:
        try:
            from cognition.flysense import FlyAL
            _FLYAL = FlyAL()
            _FLYAL_OK = True
        except Exception as exc:
            log.debug(f"flyal unavailable: {exc}")
            _FLYAL_OK = False
    return _FLYAL if _FLYAL_OK else None



# ═══════════════════════════════════════════════════════════════════════════════
#  Embedder
# ═══════════════════════════════════════════════════════════════════════════════

_EMBED_BASE_URL   = os.getenv("EMBED_BASE_URL", "http://127.0.0.1:8080")
_EMBED_MODEL      = os.getenv("EMBED_MODEL", "harrier")
_EMBED_DIMS       = env_int("EMBED_DIMS", 640)
_BATCH_SIZE       = env_int("EMBED_BATCH_SIZE", 32)
_EMBED_TIMEOUT    = env_float("EMBED_TIMEOUT_S", 30)
# Circuit breaker: consecutive HTTP failures before failing fast, and how
# long to fail fast. Env-tunable for slow/loaded backends.
_CB_FAILURES     = max(1, env_int("EMBED_CB_FAILURES", 3))
_CB_COOLDOWN_S   = max(1.0, env_float("EMBED_CB_COOLDOWN_S", 60.0))
_QUERY_INSTRUCT   = os.getenv(
    "EMBED_QUERY_INSTRUCT",
    "Retrieve relevant memories that answer the query",
)
# Query-side cap: llama-server's /embedding endpoint 400s oversized inputs
# (a scheduled-job prompt with the full worker skill attached is ~4KB).
# Queries should be short by construction; this truncates the head (task
# text comes first) instead of hard-failing the whole turn. Documents are
# pre-chunked at ingest and never pass through this cap.
_EMBED_QUERY_MAX_CHARS = max(256, env_int("EMBED_QUERY_MAX_CHARS", 2000))


def instruct_query_text(query: str, instruct: str) -> str:
    """Format a search query with the instruction prefix the embedding model expects.

    Single source of truth for the ``Instruct: …\\nQuery: …`` format — used by
    HarrierEmbedder.embed_query/embed_queries and by reason.cached_embed_batch
    so batched vectors are bit-identical to per-query ones.
    """
    return f"Instruct: {instruct}\nQuery: {(query or '')[:_EMBED_QUERY_MAX_CHARS]}"

# vec0 MATCH KNN oversampling — see user_scoped_vec_knn. Defaults mirror the
# memory-domain constants in cognition.memory.schema.
KNN_MATCH_OVERSCAN = env_int("KNN_MATCH_OVERSCAN", 16)
KNN_MATCH_K_MIN = env_int("KNN_MATCH_K_MIN", 32)


class HarrierEmbedder:
    """
    HTTP-based text embedder for harrier-oss-v1-270m via llama-server.

    Talks to a running llama-server instance (started with ``embedding = true``,
    ``pooling-type = last``) over its /embedding endpoint. Connection is lazy —
    the first call just hits the HTTP endpoint, no local model loading happens
    in this process.

    Includes a small TTL-based LRU cache on _embed_texts so that repeated
    calls with the same texts (e.g. the same user query embedded by routing,
    memory search, and knowledge search in a single turn) skip the HTTP
    round-trip.

    Optional persistent disk cache (EMBED_CACHE_PATH) survives restarts.
    """

    _CACHE_MAX: int = 256  # retained vectors, not batches
    # Nano default 300s: same greeting/query must not re-embed after 30s.
    # Override with EMBED_CACHE_TTL seconds when needed.
    _CACHE_TTL: float = env_float("EMBED_CACHE_TTL", 300.0)
    _DISK_CACHE_MAX: int = 2048  # cap on retained vectors
    _DISK_CACHE_FILE_MAX_BYTES: int = 8 * 1024 * 1024  # compact JSONL past this

    def __init__(
        self,
        base_url: str   = _EMBED_BASE_URL,
        model: str      = _EMBED_MODEL,
        dims: int       = _EMBED_DIMS,
        batch_size: int = _BATCH_SIZE,
        timeout: float  = _EMBED_TIMEOUT,
        cache_path: str | None = None,
    ) -> None:
        self.base_url   = base_url.rstrip("/")
        self.model      = model
        self.dims       = dims
        self.batch_size = int(batch_size)
        if self.batch_size < 1 or int(dims) < 1:
            raise ValueError("embedding dimensions and batch size must be positive")
        self.timeout    = timeout
        self._session   = requests.Session()
        self._cache: OrderedDict[tuple[str, ...], tuple[float, np.ndarray]] = OrderedDict()
        self._cache_lock = threading.Lock()

        # Persistent disk cache for embeddings across restarts
        self._disk_cache_path = Path(cache_path).expanduser() if cache_path else None
        self._disk_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._disk_cache_lock = threading.RLock()
        if self._disk_cache_path:
            self._load_disk_cache()

        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=requests.adapters.Retry(
                total=2,
                connect=2,
                read=2,
                backoff_factor=0.2,
                status_forcelist=[502, 503, 504],
                allowed_methods=frozenset(["GET", "POST"]),
            ),
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        # Consecutive-failure circuit-breaker state (guarded by _cache_lock).
        self._cb_failures: int = 0
        self._cb_open_until: float = 0.0

    def _geometry(self, mode=None):
        return (self.base_url, self.model, self.dims, _flyal_mode() if mode is None else mode)

    def _embedding_key(self, texts, mode=None):
        return (*self._geometry(mode), tuple(texts))

    def _validated_batch(self, values, count):
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape != (count, self.dims) or not np.isfinite(arr).all():
            raise ValueError("embedding response has invalid dimensions, count, or values")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
            raise ValueError("embedding response contains zero or invalid vectors")
        return arr

    @staticmethod
    def _trim_vector_cache(cache, limit):
        # Limits count vectors, not batches, so a larger batch setting cannot
        # multiply cache RAM usage on an 8 GB device.
        count = sum(len(value[1] if isinstance(value, tuple) else value) for value in cache.values())
        while cache and count > limit:
            oldest = next(iter(cache))
            value = cache.pop(oldest)
            count -= len(value[1] if isinstance(value, tuple) else value)

    @contextlib.contextmanager
    def _disk_guard(self):
        # The sidecar lock also coordinates separate instances/processes.
        import fcntl
        self._disk_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._disk_cache_lock, self._disk_cache_path.with_suffix(self._disk_cache_path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load_disk_cache(self) -> None:
        """Stream validated, geometry-tagged batches with bounded retention."""
        if not self._disk_cache_path or not self._disk_cache_path.exists():
            return
        try:
            with self._disk_guard(), self._disk_cache_path.open("rb") as stream:
                self._read_disk_entries(stream, self._disk_cache)
        except Exception as exc:
            log.warning("Failed to load embedding cache: %s", exc)

    def _read_disk_entries(self, stream, cache):
        # Corrupt or obsolete cache lines are disposable; never allocate an
        # unbounded line from an old cache or accept untagged model geometry.
        line_limit = 1024 * 1024
        while line := stream.readline(line_limit + 1):
            if len(line) > line_limit:
                while line and not line.endswith(b"\n"):
                    line = stream.readline(line_limit + 1)
                continue
            try:
                entry = json.loads(line)
                if entry.get("geometry") != list(self._geometry()):
                    continue
                texts = entry["texts"]
                if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
                    continue
                if any(redact_text(text) != text for text in texts):
                    continue  # old unredacted cache entries must not be reused
                vec = self._validated_batch(entry["vector"], len(texts))
                key = self._embedding_key(texts)
                cache.pop(key, None)
                cache[key] = vec
                self._trim_vector_cache(cache, self._DISK_CACHE_MAX)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue

    @staticmethod
    def _disk_entry(key, vec):
        return {"geometry": list(key[:4]), "texts": list(key[4]), "vector": vec.tolist()}

    def _save_to_disk_cache(self, key, vec) -> None:
        if not self._disk_cache_path:
            return
        try:
            with self._disk_guard():
                with self._disk_cache_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(self._disk_entry(key, vec), ensure_ascii=False) + "\n")
                self._maybe_compact_disk_cache()
        except Exception as exc:
            log.debug("Failed to write embedding cache: %s", exc)

    def _maybe_compact_disk_cache(self) -> None:
        # Called under _disk_guard so appends cannot be lost during replace.
        if not self._disk_cache_path or self._disk_cache_path.stat().st_size <= self._DISK_CACHE_FILE_MAX_BYTES:
            return
        entries = {}
        with self._disk_cache_path.open("rb") as stream:
            self._read_disk_entries(stream, entries)
        fd, temporary = tempfile.mkstemp(prefix=".embed-cache-", dir=self._disk_cache_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                for key, vec in entries.items():
                    stream.write(json.dumps(self._disk_entry(key, vec), ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._disk_cache_path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of raw texts via llama-server's /embedding endpoint.
        Returns np.ndarray of shape (len(texts), dims), L2-normalised.

        Results are cached in a small TTL-based LRU cache keyed by the tuple
        of input texts, so duplicate calls within _CACHE_TTL seconds skip the
        HTTP round-trip. Also checks persistent disk cache.
        """
        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)
        texts = [redact_text(text) for text in texts]
        now = time.monotonic()
        mode = _flyal_mode()
        mkey = self._embedding_key(texts, mode)

        # Check memory cache first
        with self._cache_lock:
            cached = self._cache.get(mkey)
            if cached is not None and now - cached[0] <= self._CACHE_TTL:
                self._cache.move_to_end(mkey)
                return cached[1].copy()

        # Check disk cache
        with self._disk_cache_lock:
            disk_cached = self._disk_cache.get(mkey)
            if disk_cached is not None:
                # Promote to memory cache
                with self._cache_lock:
                    self._cache[mkey] = (now, disk_cached)
                    self._trim_vector_cache(self._cache, self._CACHE_MAX)
                return disk_cached.copy()

        # Circuit breaker: after _CB_FAILURES consecutive HTTP failures,
        # fail fast for _CB_COOLDOWN_S instead of burning timeout×retries
        # (≈90s) per batch. Callers already degrade (persist without
        # vectors), so fast failure only converts stall time, not outcomes.
        if self._cb_open_until and now >= self._cb_open_until:
            self._cb_open_until = 0.0
        if self._cb_open_until:
            raise RuntimeError(f"embedding circuit open ({self._cb_failures} consecutive failures)")

        # HTTP call to embedding server
        try:
            resp = self._session.post(
                f"{self.base_url}/embedding",
                json={"model": self.model, "content": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or len(data) != len(texts):
                raise ValueError("embedding response count mismatch")
            vecs = []
            for item in data:
                emb = item["embedding"]
                if emb and isinstance(emb[0], list):
                    if len(emb) != 1:
                        raise ValueError("embedding response has multiple vectors per input")
                    emb = emb[0]
                vecs.append(emb)
            arr = self._validated_batch(vecs, len(texts))
        except Exception as exc:
            if isinstance(exc, requests.exceptions.HTTPError) and getattr(exc.response, "status_code", None) == 400:
                raise  # oversized input is not an unavailable backend
            with self._cache_lock:
                self._cb_failures += 1
                if self._cb_failures >= _CB_FAILURES:
                    import time as _time
                    self._cb_open_until = _time.monotonic() + _CB_COOLDOWN_S
                    log.warning("embedding circuit OPEN after %d consecutive failures — failing fast for %.0fs.",
                                self._cb_failures, _CB_COOLDOWN_S)
            raise
        with self._cache_lock:
            if self._cb_failures:
                log.info("embedding circuit closed — server recovered.")
            self._cb_failures = 0
            self._cb_open_until = 0.0

        result = arr / np.linalg.norm(arr, axis=1, keepdims=True)

        # Fly AL layer: divisive gain control (shadow logs, live replaces).
        if mode in ("shadow", "live"):
            try:
                al = _flyal()
                if al is not None:
                    if mode == "live":
                        result = al.normalize_batch(result).astype(np.float32)
                    log.debug(f"flyal mode={mode} glomeruli={al.summary()['glomeruli']} "
                              f"inh_med={al.summary()['inhibition_median']}")
            except Exception as exc:
                log.debug(f"flyal transform skipped: {exc}")

        # Store in both caches (both capped LRU-style). The disk-cache dict
        # mutation holds the lock; file appends and compaction are separately
        # serialized across instances/processes by the sidecar file lock.
        with self._cache_lock:
            self._cache[mkey] = (time.monotonic(), result.copy())
            self._trim_vector_cache(self._cache, self._CACHE_MAX)

        if self._disk_cache_path:
            with self._disk_cache_lock:
                self._disk_cache[mkey] = result.copy()
                self._trim_vector_cache(self._disk_cache, self._DISK_CACHE_MAX)
            self._save_to_disk_cache(mkey, result)

        return result

    def embed(self, texts: Iterable[str]) -> Generator[np.ndarray, None, None]:
        """Embed documents (no instruction prefix). Yields one np.ndarray(dim,) per text."""
        iterator = iter(texts)
        while batch := list(islice(iterator, self.batch_size)):
            vecs = self._embed_texts(batch)
            for v in vecs:
                yield v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed documents and return all vectors as np.ndarray (N, dims)."""
        all_vecs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_vecs.append(self._embed_texts(batch))
        return np.vstack(all_vecs) if all_vecs else np.empty((0, self.dims), dtype=np.float32)

    def embed_query(self, query: str, instruct: str = _QUERY_INSTRUCT) -> np.ndarray:
        """
        Embed a single search query with the instruction prefix.
        Returns np.ndarray(dims,).
        """
        text = instruct_query_text(query, instruct)
        for _ in range(4):
            try:
                return self._embed_texts([text])[0]
            except requests.exceptions.HTTPError as e:
                # llama-server's /embedding 400s past its token ctx and the
                # limit depends on content density, not chars — halve and
                # retry instead of failing the whole turn.
                if getattr(getattr(e, "response", None), "status_code", None) != 400 or len(text) <= 256:
                    raise
                text = text[: max(256, len(text) // 2)]
                log.debug("embed_query 400 — retrying with %d chars", len(text))
        return self._embed_texts([text])[0]

    def embed_queries(self, queries: list[str], instruct: str = _QUERY_INSTRUCT) -> np.ndarray:
        """Embed multiple search queries with the instruction prefix. Returns np.ndarray (N, dims)."""
        prefixed = [instruct_query_text(q, instruct) for q in queries]
        return self.embed_batch(prefixed)

    @staticmethod
    def serialize(vector: np.ndarray) -> bytes:
        """Serialise a float32 vector for sqlite-vec INSERT."""
        v = vector.astype(np.float32)
        return struct.pack(f"{len(v)}f", *v)


# ═══════════════════════════════════════════════════════════════════════════════
#  Database helpers
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_user_db_path(path_value: str | os.PathLike[str], *, user_id: str | None = None) -> Path:
    """Resolve an absolute or per-user relative database path.

    Relative paths live under <USER_SPACE_ROOT>/<user_id>/, matching memory,
    knowledge, and experience storage conventions. ":memory:" is SQLite's
    special in-memory sentinel — it must pass through untouched, never
    joined onto a user directory, or it becomes a literal on-disk file
    named ":memory:" instead of an ephemeral RAM-only database.
    """
    raw = str(path_value)
    if raw == ":memory:":
        return Path(":memory:")

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return user_state_path(str(path), user_id)


def connect_sqlite_db(path: str | os.PathLike[str], *, user_id: str | None = None, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open an optionally encrypted SQLite connection without sqlite-vec."""
    uid = user_id or current_user_id()
    # Resolve relative paths under <USER_SPACE_ROOT>/<user_id>/ so that
    # a bare "memory/memory.db" argument never lands in the process's
    # CWD (e.g. the repo root when launched via main.py).
    resolved = resolve_user_db_path(path, user_id=uid)
    conn = connect_sqlite(resolved, user_id=uid)
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn
    except BaseException:
        conn.close()
        raise


def initialize_sqlite_db(path: str | os.PathLike[str], ddl: str, *, user_id: str | None = None) -> sqlite3.Connection:
    """Open a standard SQLite DB, apply schema DDL, commit, and return connection."""
    conn = connect_sqlite_db(path, user_id=user_id)
    try:
        conn.executescript(ddl)
        conn.commit()
        return conn
    except BaseException:
        conn.close()
        raise


def connect_sqlite_vec(path: str | os.PathLike[str], *, user_id: str | None = None, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open an optionally encrypted SQLite connection with sqlite-vec loaded."""
    conn = connect_sqlite_db(path, user_id=user_id, busy_timeout_ms=busy_timeout_ms)
    try:
        try:
            conn.enable_load_extension(True)
        except Exception:
            pass
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        finally:
            try:
                conn.enable_load_extension(False)
            except Exception:
                pass
        return conn
    except BaseException:
        conn.close()
        raise


def initialize_sqlite_vec_db(path: str | os.PathLike[str], ddl: str, *, user_id: str | None = None) -> Any:
    """Open a sqlite-vec DB, apply schema DDL, commit, and return connection."""
    conn = connect_sqlite_vec(path, user_id=user_id)
    try:
        conn.executescript(ddl)
        conn.commit()
        ensure_vec0_cosine_metric(conn)
        return conn
    except BaseException:
        conn.close()
        raise


_VEC0_TABLE_RE = re.compile(
    r"CREATE\s+VIRTUAL\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(\w+)\s+USING\s+vec0\s*\((.*)\)\s*;?\s*$",
    re.S | re.I,
)


def ensure_vec0_cosine_metric(conn: sqlite3.Connection) -> list[str]:
    """Atomically rebuild legacy vector indexes without materializing them in RAM.

    Copy identifiers and embeddings in the same SQL projection; separate scans
    are not guaranteed to visit virtual-table rows in the same order. A savepoint
    retains the original index and caller transaction if any rebuild step fails.
    """
    migrated = []
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall()
    for name, ddl_sql in rows:
        match = _VEC0_TABLE_RE.search(ddl_sql or "")
        if not match or re.search(r"distance_metric\s*=\s*cosine", ddl_sql, re.I):
            continue
        name = _ident(name)
        cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
        if not any(c[1] == "embedding" for c in cols):
            continue
        key = "id" if any(c[1] == "id" for c in cols) else "rowid"
        definition = match.group(2).rstrip()
        if re.search(r"distance_metric\s*=", definition, re.I):
            definition = re.sub(r"distance_metric\s*=\s*\w+", "distance_metric=cosine", definition, flags=re.I)
        else:
            definition = re.sub(r"(\b(?:float|int8)\s*\[\s*\d+\s*\])",
                                r"\1 distance_metric=cosine", definition, count=1, flags=re.I)
        temporary = "vec_backup_" + uuid.uuid4().hex
        conn.execute("SAVEPOINT vec_metric_migration")
        try:
            conn.execute(f"CREATE TEMP TABLE {temporary} AS SELECT {key} AS vector_id, embedding FROM {name}")
            conn.execute(f"DROP TABLE {name}")
            conn.execute(f"CREATE VIRTUAL TABLE {name} USING vec0({definition})")
            conn.execute(f"INSERT INTO {name}({key}, embedding) SELECT vector_id, embedding FROM {temporary}")
            conn.execute(f"DROP TABLE {temporary}")
            conn.execute("RELEASE SAVEPOINT vec_metric_migration")
        except BaseException:
            conn.execute("ROLLBACK TO SAVEPOINT vec_metric_migration")
            conn.execute("RELEASE SAVEPOINT vec_metric_migration")
            raise
        migrated.append(name)
        log.info("vec0 %s: migrated to distance_metric=cosine", name)
    return migrated


_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "how", "what", "who", "when", "where", "why",
    "we", "you", "i", "he", "she", "it", "they", "this", "that", "these",
    "those", "some", "any", "all", "each", "can", "could", "will",
    "would", "should", "shall", "may", "might", "must", "to", "of", "in",
    "on", "at", "for", "with", "and", "or", "not", "no", "yes", "make",
    "made", "get", "got", "go", "going", "let", "lets", "want", "wants",
    "just", "so", "up", "down", "out", "about", "if", "then", "than",
})


def fts_or_query(text: str, *, max_terms: int = 16) -> str | None:
    """Build a conservative FTS5 OR query from literal terms.

    This intentionally stays lexical/deterministic; semantic ranking is handled
    separately by embeddings and fused later.
    """
    terms: list[str] = []
    for match in _WORD_RE.finditer(text or ""):
        term = match.group(0).strip().replace('"', "")
        if len(term) >= 2 and term.casefold() not in _STOPWORDS:
            terms.append(f'"{term}"')
    return " OR ".join(terms[:max_terms]) or None


def rrf_score(item_id: str, rank_knn: dict[str, int], rank_fts: dict[str, int], *, k: int) -> float:
    score = 0.0
    if item_id in rank_knn:
        score += 1.0 / (k + rank_knn[item_id])
    if item_id in rank_fts:
        score += 1.0 / (k + rank_fts[item_id])
    return score


def store_db_path(path_value: str | os.PathLike[str], *, user_id: str | None = None) -> Path:
    """Alias for resolving a store-owned DB path under the active user state."""
    return resolve_user_db_path(path_value, user_id=user_id)


def fetch_by_ids(conn: sqlite3.Connection, table: str, ids: set[str], *, id_column: str = "id") -> dict[str, sqlite3.Row]:
    """Fetch rows from a table by ids and return them keyed by id."""
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(f"SELECT * FROM {table} WHERE {id_column} IN ({placeholders})", list(ids)).fetchall()
    return {str(row[id_column]): row for row in rows}


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name: str) -> str:
    """Validate a SQLite identifier used in shared generated SQL."""
    if not _IDENT_RE.fullmatch(name or ""):
        raise ValueError(f"unsafe SQLite identifier: {name!r}")
    return name


def utc_now_iso() -> str:
    """Current UTC timestamp as an ISO string, shared by durable stores."""
    from system.bioclock import utc_now
    return utc_now().isoformat()


def initialize_store_db(
    path_value: str | os.PathLike[str],
    ddl: str,
    *,
    user_id: str | None = None,
    vector: bool = True,
) -> sqlite3.Connection:
    """Resolve a store DB path, initialize its schema, and return a connection.

    Store modules pass their own env-configured path and schema. Relative paths
    live under the active user's state directory; absolute paths are respected.

    When the DDL defines the personal ``memories`` table, Phase A columns are
    ensured (idempotent ALTER) and write/recall hooks are installed once.
    """
    uid = user_id or current_user_id()
    path = store_db_path(path_value, user_id=uid)
    init = initialize_sqlite_vec_db if vector else initialize_sqlite_db
    conn = init(path, ddl, user_id=uid)
    # Phase A: lightweight schema migrate for personal memory only.
    # (Write/recall hooks are now native methods on memorize._MemoryBackend —
    #  no runtime monkey-patching needed.)
    if "memories_vec" in (ddl or "") or "CREATE TABLE IF NOT EXISTS memories" in (ddl or ""):
        try:
            from cognition.memory.memorize import ensure_phase_a_schema
            ensure_phase_a_schema(conn)
        except Exception:
            # Never block boot on optional Phase A wiring.
            pass
    return conn


def sqlite_vec_blob(vector: object) -> bytes:
    """Serialize a Python/numpy vector for sqlite-vec insertion/search."""
    import sqlite_vec
    return sqlite_vec.serialize_float32(vector)


def insert_vector(conn: sqlite3.Connection, table: str, item_id: str, vector: object) -> None:
    """Insert a serialized vector into a vec0 table with common id/embedding columns."""
    table = _ident(table)
    conn.execute(f"INSERT INTO {table}(id, embedding) VALUES(?, ?)", (item_id, sqlite_vec_blob(vector)))


def user_scoped_vec_knn(
    conn: sqlite3.Connection,
    *,
    vec_table: str,
    owner_table: str,
    owner_alias: str,
    vector: object,
    user_id: str,
    limit: int,
    owner_id_column: str = "id",
    vec_id_column: str = "id",
    user_column: str = "user_id",
    threshold: float | None = None,
) -> list[sqlite3.Row]:
    """Run the common sqlite-vec KNN pattern scoped through an owner table.

    Uses the vec0 MATCH index (tables are created with distance_metric=cosine,
    so `distance` == 1 - cosine_similarity). k is oversampled because the
    owner-scope filter applies after the vec0 scan picks the k nearest rows.

    threshold, if given, is a minimum cosine similarity (0..1) — rows whose
    distance exceeds (1 - threshold) are excluded, so a lone unrelated
    candidate can't win purely by being the only thing in the table.
    """
    vec_table = _ident(vec_table)
    owner_table = _ident(owner_table)
    owner_alias = _ident(owner_alias)
    owner_id_column = _ident(owner_id_column)
    vec_id_column = _ident(vec_id_column)
    user_column = _ident(user_column)
    blob = sqlite_vec_blob(vector)
    k = max(int(limit) * KNN_MATCH_OVERSCAN, KNN_MATCH_K_MIN)

    if threshold is not None:
        dist_ceil = 1.0 - threshold
        return conn.execute(
            f"""
            SELECT v.{vec_id_column} AS id, v.distance AS dist
            FROM {vec_table} v
            JOIN {owner_table} {owner_alias} ON {owner_alias}.{owner_id_column} = v.{vec_id_column}
            WHERE v.embedding MATCH ?
              AND v.k = ?
              AND {owner_alias}.{user_column} = ?
              AND v.distance <= ?
            ORDER BY v.distance ASC
            LIMIT ?
            """,
            (blob, k, user_id, dist_ceil, limit),
        ).fetchall()

    return conn.execute(
        f"""
        SELECT v.{vec_id_column} AS id, v.distance AS dist
        FROM {vec_table} v
        JOIN {owner_table} {owner_alias} ON {owner_alias}.{owner_id_column} = v.{vec_id_column}
        WHERE v.embedding MATCH ?
          AND v.k = ?
          AND {owner_alias}.{user_column} = ?
        ORDER BY v.distance ASC
        LIMIT ?
        """,
        (blob, k, user_id, limit),
    ).fetchall()


def user_scoped_fts_search(
    conn: sqlite3.Connection,
    *,
    fts_table: str,
    owner_table: str,
    owner_alias: str,
    query: str,
    user_id: str,
    limit: int,
    owner_id_column: str = "id",
    fts_id_column: str = "id",
    user_column: str = "user_id",
) -> list[sqlite3.Row]:
    """Run the common FTS5 MATCH pattern scoped through an owner table."""
    fts = fts_or_query(query)
    if not fts:
        return []
    fts_table = _ident(fts_table)
    owner_table = _ident(owner_table)
    owner_alias = _ident(owner_alias)
    owner_id_column = _ident(owner_id_column)
    fts_id_column = _ident(fts_id_column)
    user_column = _ident(user_column)
    return conn.execute(
        f"""
        SELECT f.{fts_id_column} AS id
        FROM {fts_table} f
        JOIN {owner_table} {owner_alias} ON {owner_alias}.{owner_id_column} = f.{fts_id_column}
        WHERE {fts_table} MATCH ? AND {owner_alias}.{user_column} = ?
        ORDER BY rank
        LIMIT ?
        """,
        (fts, user_id, limit),
    ).fetchall()


def rank_by_id(rows: list[sqlite3.Row]) -> dict[str, int]:
    """Convert ordered id rows into an RRF rank mapping."""
    return {str(row["id"]): i + 1 for i, row in enumerate(rows)}


def delete_by_id(conn: sqlite3.Connection, table: str, item_id: str, *, id_column: str = "id") -> int:
    """Delete a single row by id from a table and return affected row count."""
    table = _ident(table)
    id_column = _ident(id_column)
    cur = conn.execute(f"DELETE FROM {table} WHERE {id_column}=?", (item_id,))
    return int(cur.rowcount or 0)


def delete_user_row(conn: sqlite3.Connection, table: str, item_id: str, user_id: str, *, id_column: str = "id", user_column: str = "user_id") -> int:
    """Delete one row by id + user_id and return affected row count."""
    table = _ident(table)
    id_column = _ident(id_column)
    user_column = _ident(user_column)
    cur = conn.execute(f"DELETE FROM {table} WHERE {user_column}=? AND {id_column}=?", (user_id, item_id))
    return int(cur.rowcount or 0)
