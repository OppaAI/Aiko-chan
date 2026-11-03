"""Practice loop PR3b: autonomous practice scheduler job."""
from __future__ import annotations

import json

import pytest

from agentic import experience as exp_pkg
from agentic.experience import connect, ensure_experience_schema_migrated, record_practice_experience
from agentic import practice as practice_mod
from system import practice as sys_practice


@pytest.fixture()
def exp_db(tmp_path, monkeypatch):
    db = tmp_path / "experience.db"
    monkeypatch.setattr(exp_pkg, "EXPERIENCE_DB_PATH", str(db))
    monkeypatch.setattr("agentic.experience.acquire.current_user_id", lambda: "test-user")
    monkeypatch.setattr("system.userspace.current_user_id", lambda: "test-user")
    conn = connect("test-user")
    ensure_experience_schema_migrated(conn)
    conn.close()
    return db


def test_pick_task_empty_db_returns_curriculum(exp_db):
    task = practice_mod.pick_practice_task()
    assert task["source"] == "curriculum"
    assert task["goal"]
    assert task["suggested_tools"]


def test_pick_task_prefers_recent_failure(exp_db):
    exp_id = record_practice_experience(
        "failed workflow",
        [{"tool": "kb_search", "ok": False, "args": {}}],
        "boom", verified_ok=False, score=0.1,
    )
    task = practice_mod.pick_practice_task()
    assert task["source"] == "retry"
    assert task["experience_id"] == exp_id
    assert task["suggested_tools"] == ["kb_search"]


def test_pick_task_rehearses_low_use_success(exp_db):
    exp_id = record_practice_experience(
        "good workflow",
        [{"tool": "kb_search", "ok": True, "args": {}}],
        "done", verified_ok=True, score=1.0,
    )
    task = practice_mod.pick_practice_task()
    assert task["source"] == "rehearse"
    assert task["experience_id"] == exp_id


def test_pick_task_never_raises(monkeypatch):
    monkeypatch.setattr(exp_pkg, "EXPERIENCE_DB_PATH", "/nonexistent/dir/x.db")
    task = practice_mod.pick_practice_task()
    assert task["source"] == "curriculum"


def test_ensure_practice_job_idempotent(monkeypatch):
    seen = []

    def fake_schedule_job_record(**kwargs):
        assert kwargs["title"] == sys_practice.PRACTICE_JOB_TITLE
        assert kwargs["frequency"] == "interval"
        assert kwargs["requires_idle"] is True
        assert kwargs["action"] == "agentic"
        rec = {"id": "job-1", **kwargs}
        seen.append(rec)
        return rec

    monkeypatch.setattr("system.schedule._read_all", lambda user_id=None: [])
    monkeypatch.setattr("system.schedule.schedule_job_record", fake_schedule_job_record)
    monkeypatch.setattr("system.schedule.notify_scheduler_new_job", lambda: None)

    rec = sys_practice.ensure_practice_job(user_id="u1")
    assert rec["id"] == "job-1"
    assert len(seen) == 1
    # second call finds the existing record -> no new seed
    monkeypatch.setattr(
        "system.schedule._read_all",
        lambda user_id=None: [{"title": sys_practice.PRACTICE_JOB_TITLE, "id": "job-1"}],
    )
    rec2 = sys_practice.ensure_practice_job(user_id="u1")
    assert rec2["id"] == "job-1"
    assert len(seen) == 1


def test_ensure_practice_job_respects_kill_switch(monkeypatch):
    monkeypatch.setenv("AIKO_PRACTICE_ENABLED", "0")
    assert sys_practice.ensure_practice_job(user_id="u1") is None


def test_worker_skill_mentions_safety_bounds():
    skill = sys_practice.worker_skill_text()
    assert "suggest_practice_task" in skill
    assert "record_practice_result" in skill
    assert "practice_sweep" in skill
    assert "irreversible" in skill
