"""Idle learner fixes: knowledge.db writes + curriculum fallback."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, "/home/hatch/workspace/repos/Aiko-chan")

from cognition.memory import learn as learn_mod


@pytest.fixture()
def user_dir(tmp_path, monkeypatch):
    d = tmp_path / "user_state"
    d.mkdir()
    monkeypatch.setattr(learn_mod, "user_state_path", lambda name, uid=None: d / name)
    return d


def test_curriculum_order_and_progress(user_dir):
    assert learn_mod._next_curriculum_topic("u1") == "landscape photography"
    learn_mod._mark_curriculum_done("landscape photography", "u1")
    assert learn_mod._next_curriculum_topic("u1") == "nature"
    # progress persisted
    data = json.loads((user_dir / "idle_learner_curriculum.json").read_text())
    assert data["completed"] == ["landscape photography"]
    for t in ["nature", "wildlife", "astro photography"]:
        learn_mod._mark_curriculum_done(t, "u1")
    assert learn_mod._next_curriculum_topic("u1") is None


def test_knowledge_dedup_check(user_dir):
    # fake KnowledgeSchema with one existing doc
    fake_conn = mock.MagicMock()
    fake_conn.execute.return_value.fetchone.return_value = {"1": 1}
    fake_schema = mock.MagicMock()
    fake_schema.connect.return_value = fake_conn
    with mock.patch("cognition.knowledge.schema.KnowledgeSchema", return_value=fake_schema):
        assert learn_mod._knowledge_has_doc("Self-study: nature", "u1") is True
    fake_conn.execute.return_value.fetchone.return_value = None
    with mock.patch("cognition.knowledge.schema.KnowledgeSchema", return_value=fake_schema):
        assert learn_mod._knowledge_has_doc("Self-study: nature", "u1") is False


def test_curriculum_topic_ingests_to_knowledge_db(user_dir):
    """End-to-end of the new path: no candidates -> curriculum -> ingest_text."""
    import cognition.memory.learn as L

    calls = {}

    def fake_ingest_text(title, text, *, source="", kind="ingested", embedder=None, user_id=None):
        calls.update(title=title, source=source, kind=kind, user_id=user_id,
                     text_len=len(text))
        return "doc-123"

    fake_owner = mock.MagicMock()
    fake_owner._history = []
    fake_owner._history_lock = mock.MagicMock()
    fake_owner._history_lock.__enter__ = mock.MagicMock(return_value=None)
    fake_owner._history_lock.__exit__ = mock.MagicMock(return_value=False)
    fake_owner._memorize.get_user_id.return_value = "u1"
    fake_owner._speak = None
    fake_owner._last_chat_time = 0

    # drive one iteration of the loop body: first sleep passes, second raises
    import time as time_mod

    sleep_calls = {"n": 0}

    def one_shot(_):
        sleep_calls["n"] += 1
        if sleep_calls["n"] > 1:
            raise StopIteration

    with mock.patch.object(L, "quick_studying", return_value="research result text"), \
         mock.patch("cognition.knowledge.ingest.ingest_text", side_effect=fake_ingest_text), \
         mock.patch.object(L.time, "sleep", side_effect=one_shot), \
         mock.patch.object(L, "_knowledge_has_doc", return_value=False):
        # idle long enough
        with mock.patch.object(L.time, "time", return_value=10**9):
            fake_owner._last_chat_time = 0
            try:
                L.idle_learner_loop(fake_owner, check_interval=0.01)
            except StopIteration:
                pass

    assert calls.get("title") == "Self-study: landscape photography"
    assert calls.get("source") == "idle_learner"
    assert calls.get("kind") == "self_learned"
    assert calls.get("user_id") == "u1"
    # curriculum progress recorded
    data = json.loads((user_dir / "idle_learner_curriculum.json").read_text())
    assert "landscape photography" in data["completed"]
