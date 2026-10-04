"""Practice loop PR3a: use counting + auto-promotion into playbook DAGs."""
from __future__ import annotations

import json
import os

import pytest

from agentic import experience as exp_pkg
from agentic.experience import (
    connect,
    ensure_experience_schema_migrated,
    record_experience_use,
    record_practice_experience,
)
from agentic import graph_engine as schema


@pytest.fixture()
def exp_db(tmp_path, monkeypatch):
    db = tmp_path / "experience.db"
    monkeypatch.setattr(exp_pkg, "EXPERIENCE_DB_PATH", str(db))
    # acquire.py binds current_user_id at import time — patch its namespace
    monkeypatch.setattr("agentic.experience.acquire.current_user_id", lambda: "test-user")
    monkeypatch.setattr("system.userspace.current_user_id", lambda: "test-user")
    conn = connect("test-user")
    ensure_experience_schema_migrated(conn)
    conn.close()
    return db


@pytest.fixture()
def playbook_file(tmp_path, monkeypatch):
    pb = tmp_path / "playbooks.json"
    pb.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(schema, "_playbook_file", lambda: pb)
    # graph_engine caches loaded playbooks; force a reload path
    monkeypatch.setattr(schema, "_playbooks_cache", None, raising=False)
    return pb


def _seed_ok_experience(steps=None):
    steps = steps or [
        {"tool": "kb_search", "ok": True, "args": {"query": "x"}},
        {"tool": "save_note", "ok": True, "args": {"title": "t"}},
    ]
    exp_id = record_practice_experience("practice task", steps, "done", verified_ok=True, score=1.0)
    assert exp_id
    return exp_id


def test_record_experience_use_increments(exp_db):
    exp_id = _seed_ok_experience()
    assert record_experience_use(exp_id) == 1
    assert record_experience_use(exp_id) == 2
    conn = connect("test-user")
    row = conn.execute("SELECT use_count, last_used_at FROM experiences WHERE id=?", (exp_id,)).fetchone()
    conn.close()
    assert row["use_count"] == 2
    assert row["last_used_at"]


def test_record_experience_use_unknown_id_returns_none(exp_db):
    assert record_experience_use("no-such-id") is None


def test_append_playbook_links_source_experience(exp_db, playbook_file):
    exp_id = _seed_ok_experience()
    path, plan_id = schema.append_playbook_from_experience(
        "practice task",
        [{"tool": "kb_search", "ok": True, "args": {}}],
        source_experience_id=exp_id,
    )
    assert str(path) == str(playbook_file)
    plans = json.loads(playbook_file.read_text(encoding="utf-8"))
    assert len(plans) == 1
    assert plans[0]["id"] == plan_id
    assert plans[0]["source_experience_id"] == exp_id


def test_autopromote_below_threshold_skipped(exp_db, playbook_file, monkeypatch):
    monkeypatch.setenv("AIKO_PRACTICE_AUTOPROMOTE_USES", "3")
    _seed_ok_experience()  # use_count = 0
    assert schema.maybe_autopromote_experiences() == []
    assert json.loads(playbook_file.read_text(encoding="utf-8")) == []


def test_autopromote_at_threshold_promotes_once(exp_db, playbook_file, monkeypatch):
    monkeypatch.setenv("AIKO_PRACTICE_AUTOPROMOTE_USES", "2")
    exp_id = _seed_ok_experience()
    record_experience_use(exp_id)
    record_experience_use(exp_id)
    promoted = schema.maybe_autopromote_experiences()
    assert len(promoted) == 1
    assert promoted[0]["experience_id"] == exp_id
    plans = json.loads(playbook_file.read_text(encoding="utf-8"))
    assert len(plans) == 1
    assert plans[0]["source_experience_id"] == exp_id
    # second sweep must not promote again
    assert schema.maybe_autopromote_experiences() == []
    assert len(json.loads(playbook_file.read_text(encoding="utf-8"))) == 1
    conn = connect("test-user")
    row = conn.execute(
        "SELECT promoted_playbook_id FROM experiences WHERE id=?", (exp_id,)
    ).fetchone()
    conn.close()
    assert row["promoted_playbook_id"] == promoted[0]["playbook_id"]


def test_autopromote_skips_failed_outcome(exp_db, playbook_file, monkeypatch):
    monkeypatch.setenv("AIKO_PRACTICE_AUTOPROMOTE_USES", "1")
    exp_id = record_practice_experience(
        "bad practice",
        [{"tool": "kb_search", "ok": False, "args": {}}],
        "failed", verified_ok=False, score=0.1,
    )
    record_experience_use(exp_id)
    assert schema.maybe_autopromote_experiences() == []
    assert json.loads(playbook_file.read_text(encoding="utf-8")) == []


def test_autopromote_kill_switch(exp_db, playbook_file, monkeypatch):
    monkeypatch.setenv("AIKO_PRACTICE_AUTOPROMOTE_ENABLED", "0")
    monkeypatch.setenv("AIKO_PRACTICE_AUTOPROMOTE_USES", "1")
    exp_id = _seed_ok_experience()
    record_experience_use(exp_id)
    assert schema.maybe_autopromote_experiences() == []
    assert json.loads(playbook_file.read_text(encoding="utf-8")) == []


def _row_count():
    conn = connect("test-user")
    n = conn.execute("SELECT COUNT(*) AS n FROM experiences WHERE user_id=?", ("test-user",)).fetchone()["n"]
    conn.close()
    return n


def _use_count(exp_id):
    conn = connect("test-user")
    row = conn.execute("SELECT use_count FROM experiences WHERE id=?", (exp_id,)).fetchone()
    conn.close()
    return row["use_count"]


def test_rehearse_ok_experience_bumps_use_count_no_new_row(exp_db):
    """The circular-promotion fix: rehearsing an ok experience reinforces it."""
    exp_id = _seed_ok_experience()
    assert _row_count() == 1
    steps = [{"tool": "kb_search", "ok": True, "args": {}}]
    returned = record_practice_experience(
        "rehearse", steps, "still works", verified_ok=True,
        source_experience_id=exp_id)
    assert returned == exp_id
    assert _use_count(exp_id) == 1
    assert _row_count() == 1  # no duplicate row


def test_rehearse_failed_run_records_fresh_row(exp_db):
    exp_id = _seed_ok_experience()
    steps = [{"tool": "kb_search", "ok": False, "args": {}}]
    new_id = record_practice_experience(
        "rehearse", steps, "broke", verified_ok=False,
        source_experience_id=exp_id)
    assert new_id != exp_id
    assert _use_count(exp_id) == 0  # failed reuse is not reinforcement
    assert _row_count() == 2


def test_retry_of_failed_experience_records_fresh_row(exp_db):
    """A successful retry of a failed workflow is a new (corrected) workflow."""
    failed_id = record_practice_experience(
        "broken task", [{"tool": "x", "ok": False, "args": {}}],
        "failed", verified_ok=False, score=0.2)
    assert failed_id
    new_id = record_practice_experience(
        "retry", [{"tool": "x", "ok": True, "args": {}}], "fixed",
        verified_ok=True, source_experience_id=failed_id)
    assert new_id != failed_id
    assert _use_count(failed_id) == 0
    assert _row_count() == 2


def test_rehearse_unknown_source_falls_through(exp_db):
    new_id = record_practice_experience(
        "rehearse", [{"tool": "x", "ok": True, "args": {}}], "ok",
        verified_ok=True, source_experience_id="no-such-id")
    assert new_id and new_id != "no-such-id"
    assert _row_count() == 1


def test_three_rehearses_trigger_autopromote(exp_db, playbook_file, monkeypatch):
    """End-to-end: rehearse x3 -> use_count=3 -> autopromote fires."""
    monkeypatch.setenv("AIKO_PRACTICE_AUTOPROMOTE_USES", "3")
    exp_id = _seed_ok_experience()
    steps = [{"tool": "kb_search", "ok": True, "args": {}}]
    for _ in range(3):
        assert record_practice_experience(
            "rehearse", steps, "ok", verified_ok=True,
            source_experience_id=exp_id) == exp_id
    assert _use_count(exp_id) == 3
    promoted = schema.maybe_autopromote_experiences(user_id="test-user")
    assert len(promoted) == 1
    assert promoted[0]["experience_id"] == exp_id

def test_plan_from_master_threads_source_experience_id(exp_db, playbook_file):
    """Regression: #225's merge dropped the _extras threading; plan must carry it."""
    exp_id = _seed_ok_experience()
    _, plan_id = schema.append_playbook_from_experience(
        "practice task",
        [{"tool": "kb_search", "ok": True, "args": {}}],
        source_experience_id=exp_id,
    )
    graph = schema.plan_from_master("practice task")
    assert graph is not None
    assert (graph._extras or {}).get("source_experience_id") == exp_id


def test_successful_playbook_run_bumps_source_use_count(exp_db, playbook_file, monkeypatch):
    """Regression: #225's merge dropped the run-success use bump."""
    from types import SimpleNamespace

    exp_id = _seed_ok_experience()
    schema.append_playbook_from_experience(
        "practice task",
        [{"tool": "kb_search", "ok": True, "args": {}}],
        source_experience_id=exp_id,
    )
    fake_result = SimpleNamespace(
        results=[SimpleNamespace(ok=True)],
        goal_score=None,
        final_answer="done",
    )
    monkeypatch.setattr(schema, "execute_graph", lambda *a, **k: fake_result)
    out = schema.run_schema_agent("practice task")
    assert out is fake_result
    assert _use_count(exp_id) == 1


def test_failed_playbook_run_does_not_bump_use_count(exp_db, playbook_file, monkeypatch):
    from types import SimpleNamespace

    exp_id = _seed_ok_experience()
    schema.append_playbook_from_experience(
        "practice task",
        [{"tool": "kb_search", "ok": True, "args": {}}],
        source_experience_id=exp_id,
    )
    fake_result = SimpleNamespace(
        results=[SimpleNamespace(ok=False)],
        goal_score=None,
        final_answer="failed",
    )
    monkeypatch.setattr(schema, "execute_graph", lambda *a, **k: fake_result)
    schema.run_schema_agent("practice task")
    assert _use_count(exp_id) == 0
