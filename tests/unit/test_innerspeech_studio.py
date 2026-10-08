"""Unit tests for the Inner Speech Studio (read-only thought journal).

Handlers are called directly (the fly-studio pattern): going through the
ASGI middleware would pull the auth chain, which needs server-only deps.
"""
import json
import sqlite3

import pytest


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "inner_speech.db"
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE thoughts (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'reflection',
            source TEXT NOT NULL DEFAULT 'turn',
            created_at TEXT NOT NULL,
            consolidated INTEGER NOT NULL DEFAULT 0,
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed_at TEXT NOT NULL DEFAULT 'never'
        );
        CREATE VIRTUAL TABLE thoughts_fts USING fts5(
            text, id UNINDEXED, content='thoughts', content_rowid='rowid');
        CREATE TRIGGER thoughts_ai AFTER INSERT ON thoughts BEGIN
            INSERT INTO thoughts_fts(rowid, text, id)
            VALUES (new.rowid, new.text, new.id);
        END;
        """
    )
    rows = [
        ("t1", "oppa", "Oppa seemed tired today, I should be gentler",
         "reflection", "turn", "2026-10-07T10:00:00", 0, 2),
        ("t2", "oppa", "That GPU question was interesting, learned a lot",
         "learning", "turn", "2026-10-07T11:00:00", 1, 0),
        ("t3", "oppa", "Reminder fired, he said thanks",
         "reflection", "post_job", "2026-10-07T12:00:00", 0, 0),
        ("t9", "someone_else", "not mine", "reflection", "turn",
         "2026-10-07T12:00:00", 0, 0),
    ]
    conn.executemany(
        "INSERT INTO thoughts(id, user_id, text, kind, source, created_at,"
        " consolidated, access_count) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()

    import cognition.memory.inner_speech_store as store

    monkeypatch.setattr(store, "_db_path", lambda user_id=None: str(p))
    import system.userspace as userspace

    monkeypatch.setattr(userspace, "current_user_id", lambda: "oppa")
    return p


@pytest.fixture()
def api(db_path):
    from interface.webui.studio.innerspeech.backend import api as api_mod

    return api_mod


def _body(resp):
    return json.loads(resp.body)


def test_health(api):
    assert api.health()["mode"] == "read-only"


def test_stats_user_scoped(api):
    s = _body(api.stats())
    assert s["exists"] is True
    assert s["total"] == 3  # someone_else's row excluded
    assert s["by_kind"] == {"reflection": 2, "learning": 1}
    assert s["by_source"] == {"turn": 2, "post_job": 1}
    assert s["consolidated"] == 1
    assert s["unconsolidated"] == 2


def test_thoughts_newest_first(api):
    ts = _body(api.thoughts(limit=50, offset=0))["thoughts"]
    assert [t["id"] for t in ts] == ["t3", "t2", "t1"]
    assert ts[0]["text"].startswith("Reminder fired")


def test_thoughts_kind_filter(api):
    ts = _body(api.thoughts(limit=50, offset=0, kind="learning"))["thoughts"]
    assert [t["id"] for t in ts] == ["t2"]


def test_thoughts_fts_search(api):
    ts = _body(api.thoughts(limit=50, offset=0, q="GPU"))["thoughts"]
    assert [t["id"] for t in ts] == ["t2"]


def test_thoughts_pagination(api):
    ts = _body(api.thoughts(limit=1, offset=1))["thoughts"]
    assert [t["id"] for t in ts] == ["t2"]


def test_read_only_surface(api):
    # The privacy boundary: the module exposes no write/delete/inject
    # route functions at all — inspection only.
    names = {r.name for r in api.app.routes}
    assert names >= {"health", "stats", "thoughts", "index"}
    assert not (names & {"write", "delete", "inject", "update", "create"})


def test_missing_db_reports_cleanly(tmp_path, monkeypatch):
    import cognition.memory.inner_speech_store as store
    import system.userspace as userspace

    monkeypatch.setattr(
        store, "_db_path", lambda user_id=None: str(tmp_path / "nope.db"))
    monkeypatch.setattr(userspace, "current_user_id", lambda: "oppa")
    from interface.webui.studio.innerspeech.backend import api as api_mod

    assert _body(api_mod.stats()) == {"total": 0, "exists": False}
    assert _body(api_mod.thoughts()) == {"thoughts": [], "exists": False}


def test_thoughts_malformed_fts_no_500(api):
    # FTS5 metacharacter input must degrade to [] — never an exception.
    for bad in ['"unbalanced', "foo-bar", "AND", "a:b", "*"]:
        ts = _body(api.thoughts(limit=50, offset=0, q=bad))["thoughts"]
        assert ts == [], bad
    # Empty query means "no filter", not "match nothing".
    ts = _body(api.thoughts(limit=50, offset=0, q=""))["thoughts"]
    assert len(ts) == 3


def test_thoughts_q_combined_with_filters(api):
    # kind/source must apply in the FTS branch too, not be silently ignored.
    ts = _body(api.thoughts(limit=50, offset=0, q="GPU", kind="reflection"))["thoughts"]
    assert ts == []  # t2 is kind=learning
    ts = _body(api.thoughts(limit=50, offset=0, q="GPU", kind="learning"))["thoughts"]
    assert [t["id"] for t in ts] == ["t2"]
    ts = _body(api.thoughts(limit=50, offset=0, q="tired", source="post_job"))["thoughts"]
    assert ts == []  # t1 is source=turn
