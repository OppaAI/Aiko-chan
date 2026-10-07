"""Unit tests for cognition/policy.py — learned behavior policies."""

import json
import sqlite3
import threading
import time

import pytest

from cognition.policy import (
    PolicyEngine,
    TriggerBus,
    _policy_ddl,
    evaluate_condition,
    extract_teaching,
    looks_like_teaching,
    trigger_bus,
)


class FakeEmbedder:
    def embed(self, texts):
        import numpy as np

        vecs = []
        for t in texts:
            low = t.lower()
            v = np.zeros(8, dtype=np.float32)
            for i, kw in enumerate(["gpu", "idle", "temp", "rest", "nudge", "cpu", "hot", "quiet"]):
                if kw in low:
                    v[i] = 1.0
            if v.sum() == 0:
                v[0] = 0.5
            vecs.append(v)
        return vecs


@pytest.fixture()
def engine():
    e = PolicyEngine.__new__(PolicyEngine)
    e._lock = threading.RLock()
    e._user_id = "oppa"
    e._embedder = FakeEmbedder()
    e._action_handlers = {"log": lambda ctx: None, "notify": lambda ctx: None}
    e._proposals = {}
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    with conn:
        conn.executescript(_policy_ddl(dims=8))
    e._conn = conn
    yield e
    e.close()


# ── condition evaluation ─────────────────────────────────────────────


def test_condition_ops():
    assert evaluate_condition({"metric": "t", "op": ">", "value": 85}, {"t": 90})
    assert not evaluate_condition({"metric": "t", "op": ">", "value": 85}, {"t": 80})
    assert evaluate_condition({"metric": "t", "op": "<=", "value": 75}, {"t": 75})
    assert not evaluate_condition({"metric": "t", "op": ">", "value": 1}, {})
    assert evaluate_condition({}, {"t": 1})  # empty = always


def test_condition_boolean_combos():
    c = {"all": [
        {"metric": "t", "op": ">", "value": 80},
        {"metric": "load", "op": "<", "value": 0.9},
    ]}
    assert evaluate_condition(c, {"t": 85, "load": 0.5})
    assert not evaluate_condition(c, {"t": 85, "load": 0.95})
    assert evaluate_condition({"any": [c, {"metric": "t", "op": ">", "value": 99}]},
                              {"t": 100, "load": 1.0})


# ── teach / version / supersede ──────────────────────────────────────


def test_teach_versions_and_supersedes(engine):
    v1 = engine.teach(name="gpu_rest", situation="GPU hot during jobs",
                      trigger_kind="sensor",
                      condition={"metric": "gpu_temp_c", "op": ">", "value": 85},
                      action="rest_heavy_jobs", approval="notify")
    assert v1["version"] == 1
    assert v1["status"] == "active"
    # Oppa: "make it 75" — new version, old superseded.
    v2 = engine.teach(name="gpu_rest", situation="GPU hot during jobs",
                      trigger_kind="sensor",
                      condition={"metric": "gpu_temp_c", "op": ">", "value": 75},
                      action="rest_heavy_jobs", approval="notify")
    assert v2["version"] == 2
    assert v2["condition"]["value"] == 75
    old = engine.get(v1["id"])
    assert old["status"] == "superseded"
    assert engine.active_policies()[0]["id"] == v2["id"]


# ── lookup ───────────────────────────────────────────────────────────


def test_lookup_matches_condition(engine):
    engine.teach(name="gpu_rest", situation="GPU temperature too high",
                 trigger_kind="sensor",
                 condition={"metric": "gpu_temp_c", "op": ">", "value": 85},
                 action="rest_heavy_jobs")
    hit = engine.lookup("sensor", {"gpu_temp_c": 90}, "gpu overheating")
    assert hit is not None
    assert hit["name"] == "gpu_rest"
    miss = engine.lookup("sensor", {"gpu_temp_c": 60}, "gpu fine")
    assert miss is None


def test_lookup_wrong_trigger_kind(engine):
    engine.teach(name="idle_nudge", situation="Oppa idle a long while",
                 trigger_kind="event",
                 condition={"metric": "idle_seconds", "op": ">", "value": 1200},
                 action="notify")
    assert engine.lookup("sensor", {"idle_seconds": 9999}, "idle") is None
    assert engine.lookup("event", {"idle_seconds": 9999}, "idle") is not None


# ── proposals ────────────────────────────────────────────────────────


def test_propose_confirm_flow(engine):
    draft = {"name": "idle_nudge", "situation": "Oppa idle", "trigger_kind": "event",
             "condition": {"metric": "idle_seconds", "op": ">", "value": 1200},
             "action": "notify", "approval": "notify", "teacher": "oppa"}
    prop_id, text = engine.propose(draft)
    assert "idle_nudge" in text and "yes" in text.lower()
    assert len(engine.pending_proposals()) == 1
    policy = engine.confirm_proposal(prop_id)
    assert policy["version"] == 1
    assert engine.pending_proposals() == []


def test_proposals_expire(engine):
    draft = {"name": "x", "situation": "y", "trigger_kind": "event",
             "condition": {}, "action": "log", "approval": "notify", "teacher": "oppa"}
    prop_id, _ = engine.propose(draft)
    # Age it past TTL.
    ts, d = engine._proposals[prop_id]
    engine._proposals[prop_id] = (ts - 7200, d)
    assert engine.pending_proposals() == []
    assert engine.confirm_proposal(prop_id) is None


# ── DAG compilation ──────────────────────────────────────────────────


def test_compile_dag_structure(engine):
    p = engine.teach(name="gpu_rest", situation="GPU hot",
                     trigger_kind="sensor",
                     condition={"metric": "gpu_temp_c", "op": ">", "value": 85},
                     action="rest_heavy_jobs",
                     action_params={"wait_s": 60}, approval="notify")
    dag = engine.compile_dag(p)
    ids = [n.id for n in dag.nodes]
    assert ids == ["sense", "check", "act", "verify", "notify"]
    act = [n for n in dag.nodes if n.id == "act"][0]
    assert act.tool == "rest_heavy_jobs"
    assert act.run_if == {"policy_check.ok": True}
    assert act.needs_approval is False
    ask_p = dict(p, approval="ask")
    dag2 = engine.compile_dag(ask_p)
    assert [n for n in dag2.nodes if n.id == "act"][0].needs_approval is True


# ── trigger bus ──────────────────────────────────────────────────────


def test_trigger_bus_dispatches():
    bus = TriggerBus()
    seen = []
    bus.register("sensor", lambda readings, ctx: seen.append((readings, ctx)))
    bus.emit("sensor", {"t": 90}, "hot")
    assert seen == [({"t": 90}, "hot")]
    bus.emit("event", {"x": 1})  # no handlers, no crash
    assert len(seen) == 1


def test_global_bus_singleton():
    assert trigger_bus() is trigger_bus()


# ── teaching extraction ─────────────────────────────────────────────


def test_looks_like_teaching():
    assert looks_like_teaching("if the GPU goes over 85, rest the heavy jobs")
    assert looks_like_teaching("when I'm idle 20 minutes, nudge me")
    assert looks_like_teaching("change the threshold to 75")
    assert not looks_like_teaching("what's the weather like?")
    assert not looks_like_teaching("tell me a joke")


def test_extract_teaching_mocked():
    payload = json.dumps({
        "is_teaching": True, "name": "gpu_rest",
        "situation": "GPU temperature too high during heavy jobs",
        "trigger_kind": "sensor",
        "condition": {"metric": "gpu_temp_c", "op": ">", "value": 85},
        "action": "rest_heavy_jobs", "approval": "notify",
    })

    class FakeResp:
        class Choice:
            class Msg:
                content = payload
            message = Msg()
        choices = [Choice()]

    class FakeClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kw):
                    assert kw["temperature"] == 0.0
                    return FakeResp()
            completions = Completions()
        chat = Chat()

    draft = extract_teaching("if GPU goes over 85 then rest the heavy jobs",
                             llm_client=FakeClient())
    assert draft is not None
    assert draft["name"] == "gpu_rest"
    assert draft["condition"] == {"metric": "gpu_temp_c", "op": ">", "value": 85.0}
    assert draft["approval"] == "notify"


def test_extract_teaching_rejects_non_teaching():
    assert extract_teaching("what's the weather like?") is None


# ── outcomes ─────────────────────────────────────────────────────────


def test_record_outcome(engine):
    p = engine.teach(name="x", situation="y", trigger_kind="event",
                     condition={}, action="log")
    engine.record_outcome(p["id"], True)
    engine.record_outcome(p["id"], False)
    row = engine.get(p["id"])
    assert row["use_count"] == 2
    assert row["success_count"] == 1


# ── CodeRabbit follow-ups ────────────────────────────────────────────


def test_supersede_deletes_vec_row(engine):
    v1 = engine.teach(name="gpu_rest", situation="GPU hot", trigger_kind="sensor",
                      condition={"metric": "t", "op": ">", "value": 85},
                      action="log")
    v2 = engine.teach(name="gpu_rest", situation="GPU hot", trigger_kind="sensor",
                      condition={"metric": "t", "op": ">", "value": 75},
                      action="log")
    rows = engine._conn.execute("SELECT id FROM learned_policies_vec").fetchall()
    ids = {r["id"] for r in rows}
    assert v2["id"] in ids
    assert v1["id"] not in ids  # dead vector must not fill KNN slots


def test_approval_normalized_at_teach(engine):
    p = engine.teach(name="x", situation="y", trigger_kind="event",
                     condition={}, action="log", approval="Ask")
    assert p["approval"] == "ask"
    p2 = engine.teach(name="z", situation="y", trigger_kind="event",
                      condition={}, action="log", approval="whenever")
    assert p2["approval"] == "ask"


def test_dispatch_fail_closed_on_bad_approval(engine):
    asked = []
    engine._handle_ask = lambda policy, readings: asked.append(policy["name"])
    ran = []
    engine.register_action("do_thing", lambda ctx: ran.append(1))
    # Simulate a legacy/corrupt row with an unrecognized approval value.
    p = engine.teach(name="x", situation="y", trigger_kind="event",
                     condition={}, action="do_thing", approval="notify")
    p["approval"] = "Maybe"  # bypass teach() normalization
    engine.dispatch(p, {})
    assert asked == ["x"]      # asked, not executed
    assert ran == []


def test_dispatch_unregistered_action_records_failure(engine):
    p = engine.teach(name="x", situation="y", trigger_kind="event",
                     condition={}, action="no_such_handler", approval="notify")
    engine.dispatch(p, {})
    row = engine.get(p["id"])
    assert row["use_count"] == 1
    assert row["success_count"] == 0  # nothing ran: must not count as success


def test_bus_user_scoping_and_unregister():
    bus = TriggerBus()
    seen = []
    fn_a = lambda r, c: seen.append(("a", r))
    fn_b = lambda r, c: seen.append(("b", r))
    bus.register("sensor", fn_a, user_id="alice")
    bus.register("sensor", fn_b)  # broadcast
    bus.emit("sensor", {"t": 1}, user_id="alice")
    assert ("a", {"t": 1}) in seen and ("b", {"t": 1}) in seen
    seen.clear()
    bus.emit("sensor", {"t": 2}, user_id="bob")
    assert seen == [("b", {"t": 2})]  # alice's handler not fired
    bus.unregister("sensor", fn_b)
    seen.clear()
    bus.emit("sensor", {"t": 3}, user_id="alice")
    assert seen == [("a", {"t": 3})]


def test_extract_teaching_normalizes_approval():
    import json as _json
    payload = _json.dumps({
        "is_teaching": True, "name": "p", "situation": "s",
        "trigger_kind": "event",
        "condition": {"metric": "m", "op": ">", "value": 1},
        "action": "log", "approval": "Sometimes",
    })

    class FakeResp:
        class Choice:
            class Msg:
                content = payload
            message = Msg()
        choices = [Choice()]

    class FakeClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kw):
                    return FakeResp()
            completions = Completions()
        chat = Chat()

    draft = extract_teaching("when x then do y from now on", llm_client=FakeClient())
    assert draft["approval"] == "ask"  # unrecognized -> fail closed
