"""Unit tests for cognition/inner_speech.py and its store.

The inner-speech LLM is always mocked — no network, no model. The store
uses an in-memory sqlite db with a fake embedder.
"""

import os
import sqlite3
import time

import pytest

from cognition.inner_speech import InnerSpeech
from cognition.memory.inner_speech_store import InnerSpeechStore


# ── fakes ──────────────────────────────────────────────────────────────


class FakeEmbedder:
    """Deterministic toy embeddings: keyword overlap as a vector."""

    def embed(self, texts):
        import numpy as np

        vecs = []
        for t in texts:
            low = t.lower()
            v = np.zeros(8, dtype=np.float32)
            for i, kw in enumerate(["walk", "photo", "code", "tired", "laya", "weekend", "train", "model"]):
                if kw in low:
                    v[i] = 1.0
            if v.sum() == 0:
                v[0] = 0.5
            vecs.append(v)
        return vecs


@pytest.fixture()
def store(tmp_path, monkeypatch):
    st = InnerSpeechStore.__new__(InnerSpeechStore)
    # Bypass __init__ file IO: in-memory db, fake embedder.
    import threading

    st._lock = threading.RLock()
    st._user_id = "oppa"
    st._path = ":memory:"
    st._embedder = FakeEmbedder()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    from cognition.memory.inner_speech_store import _ddl

    with conn:
        conn.executescript(_ddl(dims=8))
    st._conn = conn
    yield st
    st.close()


@pytest.fixture()
def speech(store, monkeypatch):
    monkeypatch.setenv("INNER_SPEECH_SALIENCE", "0.5")
    return InnerSpeech(store=store)


# ── salience gate ──────────────────────────────────────────────────────


def test_gate_silent_on_bland_turn(speech):
    s = speech.salience("ok", "sure thing")
    assert s < 0.5


def test_gate_fires_on_urgency_and_question(speech):
    s = speech.salience("urgent: help me now?", "", cues={"urgency": 1.0, "question": 1.0})
    assert s >= 0.5


def test_gate_fires_on_failure(speech):
    s = speech.salience("it broke", "", cues={"outcome_fail": 1.0})
    assert s >= 0.5


def test_gate_fires_on_task_event(speech):
    s = speech.salience("", "", task_event=True)
    assert s >= 0.5


def test_gate_fires_on_affect_swing(speech):
    speech._last_intensity = 0.1
    s = speech.salience("whatever", "", intensity=0.9)
    assert s >= 0.5


# ── observe: silence vs thought ────────────────────────────────────────


def test_observe_silent_turn_thinks_nothing(speech, monkeypatch):
    calls = []
    monkeypatch.setattr(speech, "think", lambda *a, **k: calls.append(1) or ["x"])
    speech.observe("ok", "sure")
    assert calls == []
    assert speech.prompt_block() == ""  # no fake thinking, no filler


def test_observe_salient_turn_thinks_and_persists(speech, monkeypatch, store):
    monkeypatch.setattr(
        speech, "think",
        lambda *a, **k: ["Maybe we should take a walk this weekend."],
    )
    speech.observe("look at these red leaves photos", "wow, beautiful",
                   cues={"question": 1.0}, intensity=0.8)
    assert "walk this weekend" in speech.latest()
    block = speech.prompt_block()
    assert "<inner_speech>" in block
    assert "</inner_speech>" in block
    assert "<inner_voice>" not in block
    # persisted for future recall
    rows = store.unconsolidated_since("2000-01-01T00:00:00")
    assert any("walk this weekend" in r["text"] for r in rows)


def test_observe_empty_thinker_result_stays_silent(speech, monkeypatch):
    monkeypatch.setattr(speech, "think", lambda *a, **k: [])
    speech.observe("urgent question?", "", cues={"urgency": 1.0, "question": 1.0})
    assert speech.prompt_block() == ""


# ── recall ─────────────────────────────────────────────────────────────


def test_recall_surfaces_similar_past_thought(store):
    store.write("Maybe we should take more photos this weekend.", kind="reflection")
    store.write("The laya training loop is stuck again.", kind="reflection")
    hits = store.recall("should we go take photos this weekend?", k=2)
    assert hits
    assert "photos" in hits[0]["text"]


def test_recall_prefers_recent_over_old(store):
    store.write("Old thought about photos from long ago.", kind="reflection")
    # Age the first row artificially.
    with store._lock, store._conn:
        store._conn.execute(
            "UPDATE thoughts SET created_at = '2020-01-01T00:00:00+00:00'"
        )
    store.write("Fresh thought about photos this week.", kind="reflection")
    hits = store.recall("photos", k=2)
    assert hits
    assert "Fresh thought" in hits[0]["text"]


def test_recall_never_leaves_the_inner_layer(speech, monkeypatch, store):
    # Recalled thoughts appear ONLY in the inner prompt block.
    store.write("A private musing about the weekend.", kind="reflection")
    monkeypatch.setattr(speech, "think", lambda *a, **k: ["Noted internally."])
    speech.observe("weekend plans?", "", cues={"question": 1.0, "urgency": 1.0})
    block = speech.prompt_block()
    assert "You've thought before:" in block
    assert "private musing" in block


# ── asides / snapshot / fly ────────────────────────────────────────────


def test_aside_queue_and_consume(speech):
    speech.queue_aside("A pop-up thought")
    assert speech.maybe_aside() == "A pop-up thought"
    assert speech.maybe_aside() is None


def test_snapshot_restore_roundtrip(speech):
    speech._thread.append("line one")
    data = speech.snapshot()
    v2 = InnerSpeech(store=speech._store)
    v2.restore(data)
    assert v2.latest() == "line one"


def test_modulatory_snapshot_shape(speech):
    snap = speech.modulatory_snapshot()
    # Empty before any observe — fly must tolerate it.
    assert snap == {} or set(snap) >= {"urge_speak", "urge_act", "valence", "energy"}
    speech._update_snapshot(0.8, emotion="warm", intensity=0.7,
                            impulse="ask_followup_question", focus={"laya"}, thought="t")
    snap = speech.modulatory_snapshot()
    assert snap["urge_speak"] > 0.4
    assert snap["valence"] > 0
    assert snap["focus"] == ["laya"]
    assert snap["has_thought"] is True


def test_reflect_on_task_writes_improvement(speech, monkeypatch):
    class FakeResp:
        class Choice:
            class Msg:
                content = "Next time I should batch the searches first."
            message = Msg()
        choices = [Choice()]

    class FakeClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    return FakeResp()
            completions = Completions()
        chat = Chat()

    monkeypatch.setattr(speech, "_get_client", lambda: FakeClient())
    speech.reflect_on_task("nightly web research job", outcome="completed")
    assert "batch the searches" in speech.latest()


def test_capture_thinking_ignores_trivia(speech, store):
    speech.capture_thinking("short")
    assert store.unconsolidated_since("2000-01-01T00:00:00") == []
    speech.capture_thinking(
        "Oppa seems stuck on the training loop; maybe I should look up "
        "efficient fine-tuning tips before he burns out on it."
    )
    rows = store.unconsolidated_since("2000-01-01T00:00:00")
    assert len(rows) == 1
    assert rows[0]["source"] == "thinking-box"
