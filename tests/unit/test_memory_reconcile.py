"""Tests for sleep-time memory reconciliation.

Covers:
  1. ensure_reconcile_schema idempotency (+ one-time episodic backfill).
  2. Temporal supersede: old row closed with valid_to/superseded_by, the
     identity/persona currently-true view excludes it, and the row remains
     as history (episodic recall is NOT validity-filtered).
  3. run_reconcile with a fake LLM: merge action supersedes members and
     inserts the merged row; noop stamps reconciled_at without changes;
     dry_run writes nothing; the cluster budget is respected.

Uses a real sqlite-vec DB with a deterministic hash-based FakeEmbedder
(mirroring tests/unit/test_memorize.py), so KNN clustering behavior is
fully controllable with no model load.
"""
from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from cognition.consolidate import reconcile as reconcile_mod
from cognition.consolidate.reconcile import run_reconcile
from cognition.memory.memorize import AikoMemorize, MemoryBackend
from cognition.memory.schema import (
    EMBED_DIMS,
    MEM_TYPE_EPISODIC,
    MEM_TYPE_SEMANTIC,
    ensure_reconcile_schema,
    existing_columns,
)

USER_ID = "reconcile-test-user"


class FakeEmbedder:
    """Deterministic stand-in: same text -> same vector (cosine 1.0)."""

    def _vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (h * (EMBED_DIMS // len(h) + 1))[: EMBED_DIMS * 4]
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = arr[:EMBED_DIMS] / 255.0
        norm = np.linalg.norm(arr)
        return arr / norm if norm else arr

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        return self._vec(text)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)


@pytest.fixture
def backend(tmp_path):
    b = MemoryBackend(
        db_path=str(tmp_path / "test_reconcile.db"),
        llm_base_url="http://unused",
        model="unused",
        embedder=FakeEmbedder(),
    )
    yield b
    b._conn.close()


@pytest.fixture
def emb():
    return FakeEmbedder()


def _add(backend, text, *, kind="fact", mem_type=None, pinned=0, user_id=USER_ID):
    """Insert one row with a deterministic vector; returns the id."""
    mem_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    vec = FakeEmbedder()._vec(text).tolist()
    backend._insert_row(
        mem_id=mem_id, user_id=user_id, text=text, now=now,
        vector=vec, pinned=pinned, kind=kind, mem_type=mem_type,
    )
    backend._conn.commit()
    return mem_id


def _row(backend, mem_id):
    return backend._conn.execute(
        "SELECT * FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()


class _PersonaShim:
    """Minimal stand-in so the REAL AikoMemorize._build_persona_context runs
    against the test backend (avoids booting a full AikoMemorize)."""

    def __init__(self, backend, user_id):
        self._mem = backend
        self._user_id_override = user_id

    @property
    def _conn(self):
        return self._mem._conn

    def _resolve_user_id(self, uid=None):
        return uid or self._user_id_override


# ── 1. schema ──────────────────────────────────────────────────────────────

class TestReconcileSchema:
    def test_idempotent(self, backend):
        conn = backend._conn
        # Backend __init__ already ran the migration; re-running is a no-op.
        assert ensure_reconcile_schema(conn) == []
        for col in ("mem_type", "valid_from", "valid_to",
                    "superseded_by", "reconciled_at"):
            assert col in existing_columns(conn)
        idx = {
            r["name"] for r in
            conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "idx_memories_user_memtype" in idx
        assert "idx_memories_user_valid" in idx

    def test_episodic_backfill_is_one_time(self, backend):
        # _insert_row defaults scene kind -> episodic tier.
        sid = _add(backend, "A rainy scene.", kind="scene")
        assert _row(backend, sid)["mem_type"] == MEM_TYPE_EPISODIC
        # A deliberate re-type to semantic must survive re-running the
        # migration (the backfill only fires when the column is added).
        backend._conn.execute(
            "UPDATE memories SET mem_type = ? WHERE id = ?", (MEM_TYPE_SEMANTIC, sid)
        )
        backend._conn.commit()
        ensure_reconcile_schema(backend._conn)
        assert _row(backend, sid)["mem_type"] == MEM_TYPE_SEMANTIC

    def test_insert_defaults(self, backend):
        fid = _add(backend, "User likes tea.")
        r = _row(backend, fid)
        assert r["mem_type"] == MEM_TYPE_SEMANTIC
        assert r["valid_from"]  # stamped at write time
        assert r["valid_to"] is None
        assert r["reconciled_at"] is None


# ── 2. temporal supersede + currently-true view ────────────────────────────

class TestTemporalSupersede:
    def test_close_and_reopen(self, backend):
        old_id = _add(backend, "User's favorite color is blue.", kind="identity")
        new_id = backend.supersede_with_temporal(
            old_id, "User's favorite color is green.", USER_ID
        )
        assert new_id and new_id != old_id
        old = _row(backend, old_id)
        assert old["status"] == "superseded"
        assert old["valid_to"]
        assert old["superseded_by"] == new_id
        new = _row(backend, new_id)
        assert new["status"] == "active"
        assert new["valid_from"]
        assert new["mem_type"] == MEM_TYPE_SEMANTIC  # inherited
        assert new["supersedes_id"] == old_id

    def test_supersede_already_closed_returns_none(self, backend):
        old_id = _add(backend, "User drives a sedan.")
        assert backend.supersede_with_temporal(old_id, "User drives a coupe.", USER_ID)
        assert backend.supersede_with_temporal(old_id, "User drives a truck.", USER_ID) is None

    def test_persona_view_excludes_superseded(self, backend):
        keep_id = _add(backend, "User is a night owl.", kind="identity")
        old_id = _add(backend, "User's favorite color is blue.", kind="identity")
        backend.supersede_with_temporal(old_id, "User's favorite color is green.", USER_ID)

        shim = _PersonaShim(backend, USER_ID)
        blob = AikoMemorize._build_persona_context(shim)
        assert blob is not None
        assert "night owl" in blob
        assert "color is green" in blob
        assert "color is blue" not in blob  # currently-true view

    def test_history_preserved_for_episodic_recall(self, backend):
        # Episodes are history: no validity filter on the episodic path.
        old_id = _add(backend, "User visited Kyoto in spring.", kind="episode")
        backend.supersede_with_temporal(old_id, "User visited Kyoto in autumn.", USER_ID)
        rows = backend._conn.execute(
            "SELECT id FROM memories WHERE user_id = ? AND kind = 'episode'",
            (USER_ID,),
        ).fetchall()
        ids = {r["id"] for r in rows}
        assert old_id in ids  # superseded row still present as history


# ── 3. run_reconcile ─────────────────────────────────────────────────────

def _fake_chat_merge(monkeypatch, id_a, id_b, merged_text="merged fact"):
    def fake(system, user, max_tokens=800, temperature=0.1):
        return (
            '[{"action":"merge","text":%s,"source_ids":["%s","%s"]}]'
            % (_json_str(merged_text), id_a, id_b)
        )
    monkeypatch.setattr(reconcile_mod, "_chat", fake)


def _json_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class TestRunReconcile:
    def test_merge(self, backend, monkeypatch):
        a = _add(backend, "User likes green tea.")
        b = _add(backend, "User likes green tea.")  # identical -> sim 1.0
        _fake_chat_merge(monkeypatch, a, b, merged_text="User likes green tea (confirmed).")

        result = run_reconcile(backend, USER_ID)
        assert result["clusters"] == 1
        assert result["merged"] == 1
        assert result["llm_calls"] == 1

        ra, rb = _row(backend, a), _row(backend, b)
        assert ra["status"] == "superseded" and ra["valid_to"]
        assert rb["status"] == "superseded" and rb["valid_to"]
        assert ra["superseded_by"] == rb["superseded_by"]
        merged_id = ra["superseded_by"]
        merged = _row(backend, merged_id)
        assert merged["memory"] == "User likes green tea (confirmed)."
        assert merged["valid_from"] and merged["valid_to"] is None
        # Every examined row is stamped so the job makes progress.
        assert ra["reconciled_at"] and rb["reconciled_at"]
        assert merged["reconciled_at"] is None  # fresh row, not yet examined

    def test_noop_stamps_without_changes(self, backend, monkeypatch):
        a = _add(backend, "User plays chess on Sundays.")
        monkeypatch.setattr(
            reconcile_mod, "_chat",
            lambda system, user, max_tokens=800, temperature=0.1: '[{"action":"keep"}]',
        )
        result = run_reconcile(backend, USER_ID)
        assert result["clusters"] == 0  # single row can't cluster
        assert result["llm_calls"] == 0
        r = _row(backend, a)
        assert r["status"] == "active"
        assert r["reconciled_at"]  # stamped as examined

    def test_dry_run_writes_nothing(self, backend, monkeypatch):
        a = _add(backend, "User likes green tea.")
        b = _add(backend, "User likes green tea.")
        _fake_chat_merge(monkeypatch, a, b)

        result = run_reconcile(backend, USER_ID, dry_run=True)
        assert result["dry_run"] is True
        assert result["planned"], "dry_run must return planned actions"
        ra, rb = _row(backend, a), _row(backend, b)
        assert ra["status"] == "active" and rb["status"] == "active"
        assert ra["reconciled_at"] is None and rb["reconciled_at"] is None
        count = backend._conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE user_id = ?", (USER_ID,)
        ).fetchone()["c"]
        assert count == 2

    def test_supersede_action(self, backend, monkeypatch):
        old_id = _add(backend, "User lives in Osaka.")
        new_id = _add(backend, "User lives in Osaka.")  # identical -> cluster
        monkeypatch.setattr(
            reconcile_mod, "_chat",
            lambda system, user, max_tokens=800, temperature=0.1: (
                '[{"action":"supersede","old_id":"%s","new_text":"User lives in Kyoto."}]' % old_id
            ),
        )
        result = run_reconcile(backend, USER_ID)
        assert result["superseded"] == 1
        old = _row(backend, old_id)
        assert old["status"] == "superseded" and old["valid_to"]
        assert old["superseded_by"]
        new = _row(backend, old["superseded_by"])
        assert "Kyoto" in new["memory"] and new["valid_from"]

    def test_cluster_budget_respected(self, backend, monkeypatch):
        # Two independent clusters via a planted KNN map (deterministic).
        a1 = _add(backend, "Fact alpha one.")
        a2 = _add(backend, "Fact alpha one.")
        b1 = _add(backend, "Fact beta one.")
        b2 = _add(backend, "Fact beta one.")
        neighbors = {a1: [a2], a2: [a1], b1: [b2], b2: [b1]}

        # Deterministic KNN: match the query vector back to its row by bytes.
        vec_to_id = {}
        for mid in (a1, a2, b1, b2):
            v = backend._conn.execute(
                "SELECT embedding FROM memories_vec WHERE id = ?", (mid,)
            ).fetchone()["embedding"]
            vec_to_id[bytes(v)] = mid

        def fake_knn(conn, vector, user_id, limit, threshold=None, active_only=True):
            import sqlite_vec as _sv

            key = bytes(_sv.serialize_float32(vector))
            mid = vec_to_id.get(key)
            if not mid:
                return []
            return [{"id": mid, "dist": 0.0}] + [
                {"id": n, "dist": 0.0} for n in neighbors.get(mid, [])
            ]

        monkeypatch.setattr(reconcile_mod, "_sqlite_knn_search", fake_knn)
        monkeypatch.setattr(reconcile_mod, "MEMORY_RECONCILE_MAX_CLUSTERS", 1)
        monkeypatch.setattr(
            reconcile_mod, "_chat",
            lambda system, user, max_tokens=800, temperature=0.1: '[{"action":"keep"}]',
        )
        result = run_reconcile(backend, USER_ID)
        assert result["clusters"] == 1  # budget cut the second cluster
        assert result["llm_calls"] == 1

    # ── CodeRabbit follow-ups (PR #311) ────────────────────────────────

    def test_stamp_only_examined_on_budget_cut(self, backend, monkeypatch):
        # Two disjoint clusters; the cluster budget allows only one. The
        # unprocessed cluster's members must NOT be stamped reconciled_at —
        # otherwise the next pass would skip them for RESCAN_DAYS.
        a1 = _add(backend, "User likes green tea.")
        a2 = _add(backend, "User likes green tea.")
        b1 = _add(backend, "User plays chess on Sundays.")
        b2 = _add(backend, "User plays chess on Sundays.")

        neighbors = {a1: [a2], a2: [a1], b1: [b2], b2: [b1]}
        vec_to_id = {}
        for mid in (a1, a2, b1, b2):
            v = backend._conn.execute(
                "SELECT embedding FROM memories_vec WHERE id = ?", (mid,)
            ).fetchone()["embedding"]
            vec_to_id[bytes(v)] = mid

        def fake_knn(conn, vector, user_id, limit, threshold=None, active_only=True):
            import sqlite_vec as _sv
            key = bytes(_sv.serialize_float32(vector))
            mid = vec_to_id.get(key)
            if not mid:
                return []
            return [{"id": mid, "dist": 0.0}] + [
                {"id": n, "dist": 0.0} for n in neighbors.get(mid, [])
            ]

        monkeypatch.setattr(reconcile_mod, "_sqlite_knn_search", fake_knn)
        monkeypatch.setattr(reconcile_mod, "MEMORY_RECONCILE_MAX_CLUSTERS", 1)
        monkeypatch.setattr(
            reconcile_mod, "_chat",
            lambda system, user, max_tokens=800, temperature=0.1: '[{"action":"keep"}]',
        )
        result = run_reconcile(backend, USER_ID)
        assert result["clusters"] == 1
        assert result["examined"] == 2
        stamped = {
            r["id"] for r in backend._conn.execute(
                "SELECT id FROM memories WHERE reconciled_at IS NOT NULL"
            ).fetchall()
        }
        # Exactly one cluster's members stamped; the other pair untouched.
        assert stamped == {a1, a2} or stamped == {b1, b2}

    def test_cluster_apply_is_atomic(self, backend, monkeypatch):
        # A mid-cluster failure must roll back the whole cluster: no partial
        # merge row, no half-closed members, no result counters moved.
        a = _add(backend, "User likes green tea.")
        b = _add(backend, "User likes green tea.")
        members = [
            {"id": a, "memory": "User likes green tea.", "kind": "fact",
             "mem_type": "semantic", "created_at": "2026-01-01"},
            {"id": b, "memory": "User likes green tea.", "kind": "fact",
             "mem_type": "semantic", "created_at": "2026-01-01"},
        ]
        actions = [{"action": "merge", "text": "merged fact",
                    "source_ids": [a, b]}]

        orig_close = reconcile_mod._close_row_temporal
        calls: list = []

        def flaky_close(conn, old_id, new_id, user_id, now):
            calls.append(old_id)
            if len(calls) > 1:
                raise RuntimeError("boom")
            return orig_close(conn, old_id, new_id, user_id, now)

        monkeypatch.setattr(reconcile_mod, "_close_row_temporal", flaky_close)
        now = datetime.now(timezone.utc).isoformat()
        result = {"merged": 0, "superseded": 0, "kept": 0}
        with pytest.raises(RuntimeError):
            reconcile_mod._apply_cluster(backend, USER_ID, members, actions, now, result)

        rows = backend._conn.execute(
            "SELECT id, status, valid_to FROM memories WHERE user_id = ?",
            (USER_ID,),
        ).fetchall()
        assert len(rows) == 2  # no merged row leaked
        assert all(r["status"] == "active" for r in rows)
        assert all(r["valid_to"] is None for r in rows)
        assert result == {"merged": 0, "superseded": 0, "kept": 0}

    def test_reconcile_clears_caches_after_writes(self, backend, monkeypatch):
        a = _add(backend, "User likes green tea.")
        b = _add(backend, "User likes green tea.")
        _fake_chat_merge(monkeypatch, a, b)
        calls: list = []
        monkeypatch.setattr(backend, "_clear_search_cache",
                            lambda: calls.append(1))
        result = run_reconcile(backend, USER_ID)
        assert result["merged"] == 1
        assert len(calls) == 1

    def test_reconcile_no_cache_clear_without_writes(self, backend, monkeypatch):
        _add(backend, "User plays chess on Sundays.")
        monkeypatch.setattr(
            reconcile_mod, "_chat",
            lambda system, user, max_tokens=800, temperature=0.1: '[{"action":"keep"}]',
        )
        calls: list = []
        monkeypatch.setattr(backend, "_clear_search_cache",
                            lambda: calls.append(1))
        run_reconcile(backend, USER_ID)
        assert calls == []

    def test_wrapper_clear_search_cache_clears_both(self):
        # AikoMemorize._clear_search_cache must clear its own _search_cache
        # (the one _search_top reads), not just the backend's.
        import threading
        from cognition.memory.memorize import AikoMemorize
        m = AikoMemorize.__new__(AikoMemorize)
        m._search_cache = {"k": "v"}
        m._search_cache_lock = threading.RLock()

        backend_calls: list = []

        class FakeBackend:
            def _clear_search_cache(self):
                backend_calls.append(1)

        m._mem_backend = FakeBackend()
        m._mem = m._mem_backend
        m._persona_lock = threading.RLock()
        m._persona_cached = "blob"
        m._persona_cache_at = 123.0
        m._clear_search_cache()
        assert m._search_cache == {}
        assert backend_calls == [1]
        assert m._persona_cached is None
