"""Functional tests for learned-state wipe: knowledge + experience delete_all.

Uses temp DB files (absolute paths are respected by the stores) and two user
ids to prove per-user isolation. Triggers are exercised too: FTS/vec side
tables must not retain rows for deleted users.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_clear_mem_stores.py -q --override-ini="addopts="
"""
from __future__ import annotations

import pytest

from cognition.knowledge import schema as ks
from agentic.experience import schema as es


@pytest.fixture()
def knowledge_db(tmp_path, monkeypatch):
    db = tmp_path / "knowledge.db"
    monkeypatch.setattr(ks, "KNOWLEDGE_DB_PATH", str(db))
    conn = ks.connect("u1")
    try:
        conn.execute(
            "INSERT INTO learned_docs(id, user_id, title, created_at) VALUES(?,?,?,?)",
            ("d1", "u1", "t", "2026-01-01"),
        )
        conn.execute(
            "INSERT INTO learned_chunks(id, doc_id, user_id, chunk_index, text, created_at) VALUES(?,?,?,?,?,?)",
            ("c1", "d1", "u1", 0, "hello world", "2026-01-01"),
        )
        conn.execute(
            "INSERT INTO learned_chunks_archive(id, doc_id, user_id, chunk_index, text, created_at, archived_at) VALUES(?,?,?,?,?,?,?)",
            ("a1", "d1", "u1", 0, "old", "2026-01-01", "2026-02-01"),
        )
        conn.execute(
            "INSERT INTO knowledge_prune_meta(user_id, last_id, updated_at) VALUES(?,?,?)",
            ("u1", "c1", "2026-02-01"),
        )
        conn.execute(
            "INSERT INTO learned_docs(id, user_id, title, created_at) VALUES(?,?,?,?)",
            ("d2", "u2", "other", "2026-01-01"),
        )
        conn.commit()
    finally:
        conn.close()
    return db


@pytest.fixture()
def experience_db(tmp_path, monkeypatch):
    import agentic.experience as exp_pkg

    db = tmp_path / "experience.db"
    monkeypatch.setattr(exp_pkg, "EXPERIENCE_DB_PATH", str(db))
    conn = es.connect("u1")
    try:
        conn.execute(
            "INSERT INTO experiences(id, user_id, goal, record_text, steps_json, outcome, score, answer_excerpt, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            ("e1", "u1", "g", "did a thing", "[]", "ok", 1.0, "done", "2026-01-01"),
        )
        conn.execute(
            "INSERT INTO experiences(id, user_id, goal, record_text, steps_json, outcome, score, answer_excerpt, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            ("e2", "u2", "g2", "other thing", "[]", "ok", 1.0, "done", "2026-01-01"),
        )
        conn.execute(
            "INSERT INTO engram_relations(from_engram, to_engram, relation_type, confidence, created_at) VALUES(?,?,?,?,?)",
            ("e1", "e2", "refines", 1.0, "2026-01-01"),
        )
        conn.commit()
    finally:
        conn.close()
    return db


def _count(conn, table, uid_col="user_id", uid="u1"):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {uid_col}=?", (uid,)).fetchone()[0]


class TestKnowledgeDeleteAll:
    def test_wipes_user_only(self, knowledge_db):
        counts = ks.delete_all("u1")
        assert counts["learned_chunks"] == 1
        assert counts["learned_docs"] == 1
        conn = ks.connect("u1")
        try:
            assert _count(conn, "learned_chunks") == 0
            assert _count(conn, "learned_docs") == 0
            assert _count(conn, "learned_chunks_archive") == 0
            assert _count(conn, "knowledge_prune_meta") == 0
            assert _count(conn, "learned_docs", uid="u2") == 1  # other user intact
            assert conn.execute("SELECT COUNT(*) FROM learned_chunks_fts").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM learned_chunks_vec").fetchone()[0] == 0
        finally:
            conn.close()


class TestExperienceDeleteAll:
    def test_wipes_user_only(self, experience_db):
        counts = es.delete_all("u1")
        assert counts["experiences"] == 1
        conn = es.connect("u1")
        try:
            assert _count(conn, "experiences") == 0
            assert _count(conn, "experiences", uid="u2") == 1
            assert conn.execute("SELECT COUNT(*) FROM engram_relations").fetchone()[0] == 0
            # The ai trigger auto-indexed both rows; u1's is gone, u2's survives.
            assert conn.execute("SELECT COUNT(*) FROM experiences_fts").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM experiences_fts WHERE id='e2'").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM experiences_vec").fetchone()[0] == 0
        finally:
            conn.close()
