"""Learned policies: behavior as data, not code.

The fast-action learning loop:

    sense (trigger bus) --> policy lookup --> known? --> DAG executes
                                                  --> unknown? --> LLM asks Oppa
        --> Oppa teaches via chat --> stored (versioned, supersede chain)
        --> compiled to DAG --> next time the trigger fires autonomously

Policies live in experience.db (``learned_policies`` table), versioned with
supersede chains — "make it 75" creates v2, v1 stays for history and rollback.
Thresholds are data, never code. Approval levels ride with the policy:
autonomous (reversible, notify after), notify (act + loud notify), ask
(always confirm first).

This is the framework. System sensors (CPU/GPU) plug into the trigger bus
later; conversation-derived triggers (idle, events) work now.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

log = logging.getLogger("aiko.policy")

EMBED_DIMS = int(os.getenv("EMBED_DIMS", "640"))

_POLICY_DDL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS learned_policies (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    name             TEXT NOT NULL,
    situation        TEXT NOT NULL,
    trigger_kind     TEXT NOT NULL DEFAULT 'event',
    condition_json   TEXT NOT NULL DEFAULT '{{}}',
    action           TEXT NOT NULL,
    action_params_json TEXT NOT NULL DEFAULT '{{}}',
    approval         TEXT NOT NULL DEFAULT 'notify',
    teacher          TEXT NOT NULL DEFAULT 'oppa',
    version          INTEGER NOT NULL DEFAULT 1,
    supersedes_id    TEXT,
    status           TEXT NOT NULL DEFAULT 'active',
    use_count        INTEGER NOT NULL DEFAULT 0,
    success_count    INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policies_user ON learned_policies(user_id);
CREATE INDEX IF NOT EXISTS idx_policies_name ON learned_policies(name);
CREATE INDEX IF NOT EXISTS idx_policies_status ON learned_policies(status);

CREATE VIRTUAL TABLE IF NOT EXISTS learned_policies_fts USING fts5(
    name, situation,
    id UNINDEXED,
    content='learned_policies',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS learned_policies_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dims}] distance_metric=cosine
);

CREATE TRIGGER IF NOT EXISTS learned_policies_ai AFTER INSERT ON learned_policies BEGIN
    INSERT INTO learned_policies_fts(rowid, name, situation, id)
    VALUES (new.rowid, new.name, new.situation, new.id);
END;

CREATE TRIGGER IF NOT EXISTS learned_policies_ad AFTER DELETE ON learned_policies BEGIN
    INSERT INTO learned_policies_fts(learned_policies_fts, rowid, name, situation, id)
    VALUES ('delete', old.rowid, old.name, old.situation, old.id);
END;
"""


def _policy_ddl(dims: int = EMBED_DIMS) -> str:
    return _POLICY_DDL_TEMPLATE.format(dims=dims)


# ── condition evaluation (no LLM; pure data) ───────────────────────────

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def evaluate_condition(condition: dict, readings: dict) -> bool:
    """Evaluate a policy condition against sensor/readings.

    Condition: {"metric": "gpu_temp_c", "op": ">", "value": 85}
    or {"all": [cond, cond]} / {"any": [cond, cond]}.
    """
    if not condition:
        return True
    if "all" in condition:
        return all(evaluate_condition(c, readings) for c in condition["all"])
    if "any" in condition:
        return any(evaluate_condition(c, readings) for c in condition["any"])
    try:
        metric = condition["metric"]
        op = _OPS[condition["op"]]
        value = float(condition["value"])
        reading = readings.get(metric)
        if reading is None:
            return False
        return bool(op(float(reading), value))
    except (KeyError, TypeError, ValueError):
        return False


# ── trigger bus ────────────────────────────────────────────────────────

class TriggerBus:
    """In-process pub/sub for sensor and event inputs.

    System sensors (CPU/GPU) plug in later via emit("sensor", {...}).
    Conversation-derived triggers (idle, events) emit now.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handlers: dict[str, list[Callable]] = {}

    def register(self, kind: str, fn: Callable[[dict, str], None]) -> None:
        with self._lock:
            self._handlers.setdefault(kind, []).append(fn)

    def emit(self, kind: str, readings: dict, context: str = "") -> None:
        with self._lock:
            fns = list(self._handlers.get(kind, []))
        for fn in fns:
            try:
                fn(dict(readings), context)
            except Exception as e:
                log.debug("policy: trigger handler failed: %s", e)


_BUS = TriggerBus()


def trigger_bus() -> TriggerBus:
    return _BUS


# ── policy engine ──────────────────────────────────────────────────────

class PolicyEngine:
    """Teach / lookup / supersede learned behavior policies."""

    def __init__(self, user_id: str | None = None, embedder=None) -> None:
        self._lock = threading.RLock()
        self._user_id = user_id or "oppa"
        self._embedder = embedder
        self._conn: sqlite3.Connection | None = None
        self._action_handlers: dict[str, Callable[[dict], None]] = {
            "log": self._handle_log,
            "notify": self._handle_notify,
        }
        # Pending proposals awaiting Oppa's confirmation (id -> draft).
        self._proposals: dict[str, dict] = {}

    # ── storage ──

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        from agentic.experience.schema import _db_path
        from cognition.memory.vecstore import initialize_store_db

        conn = initialize_store_db(_db_path(), _policy_ddl(), user_id=self._user_id, vector=True)
        self._conn = conn
        return conn

    def _embed(self, text: str) -> list[float] | None:
        try:
            if self._embedder is None:
                from cognition.memory.vecstore import HarrierEmbedder

                self._embedder = HarrierEmbedder()
            vec = self._embedder.embed([text])
            import numpy as _np

            arr = _np.asarray(vec[0] if hasattr(vec, "__len__") else vec)
            return [float(x) for x in arr.ravel().tolist()]
        except Exception as e:
            log.debug("policy: embed failed: %s", e)
            return None

    # ── teach ──

    def teach(
        self,
        *,
        name: str,
        situation: str,
        trigger_kind: str = "event",
        condition: dict | None = None,
        action: str,
        action_params: dict | None = None,
        approval: str = "notify",
        teacher: str = "oppa",
    ) -> dict:
        """Store a new policy version; supersede the previous active one."""
        now = datetime.now(timezone.utc).isoformat()
        pid = uuid.uuid4().hex[:16]
        with self._lock:
            conn = self._connect()
            prev = conn.execute(
                "SELECT id, version FROM learned_policies"
                " WHERE user_id = ? AND name = ? AND status = 'active'"
                " ORDER BY version DESC LIMIT 1",
                (self._user_id, name),
            ).fetchone()
            version = 1
            supersedes = None
            if prev:
                version = int(prev["version"]) + 1
                supersedes = prev["id"]
                conn.execute(
                    "UPDATE learned_policies SET status = 'superseded', updated_at = ?"
                    " WHERE id = ?",
                    (now, prev["id"]),
                )
            row = {
                "id": pid, "user_id": self._user_id, "name": name,
                "situation": situation, "trigger_kind": trigger_kind,
                "condition_json": json.dumps(condition or {}),
                "action": action,
                "action_params_json": json.dumps(action_params or {}),
                "approval": approval, "teacher": teacher,
                "version": version, "supersedes_id": supersedes,
                "status": "active", "created_at": now, "updated_at": now,
            }
            cols = ", ".join(row.keys())
            conn.execute(
                f"INSERT INTO learned_policies({cols}) VALUES ({', '.join('?' * len(row))})",
                tuple(row.values()),
            )
            vec = self._embed(f"{name}\n{situation}")
            if vec:
                import sqlite_vec  # type: ignore

                conn.execute(
                    "INSERT INTO learned_policies_vec(id, embedding) VALUES (?, ?)",
                    (pid, sqlite_vec.serialize_float32(vec)),
                )
            conn.commit()
        log.info("policy: taught %s v%d (approval=%s)", name, version, approval)
        return self.get(pid)

    def get(self, pid: str) -> dict | None:
        with self._lock:
            row = self._connect().execute(
                "SELECT * FROM learned_policies WHERE id = ?", (pid,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def active_policies(self, trigger_kind: str | None = None) -> list[dict]:
        with self._lock:
            q = ("SELECT * FROM learned_policies WHERE user_id = ? AND status = 'active'")
            args: tuple = (self._user_id,)
            if trigger_kind:
                q += " AND trigger_kind = ?"
                args += (trigger_kind,)
            rows = self._connect().execute(q + " ORDER BY updated_at DESC", args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("condition_json", "action_params_json"):
            try:
                d[k.replace("_json", "")] = json.loads(d.pop(k) or "{}")
            except (TypeError, ValueError):
                d[k.replace("_json", "")] = {}
        return d

    # ── lookup ──

    def lookup(self, trigger_kind: str, readings: dict, situation_hint: str = "") -> dict | None:
        """Find the best active policy for this trigger.

        Vector similarity on (name + situation), then the policy's condition
        is evaluated against the actual readings. Most-used wins ties.
        """
        candidates = self.active_policies(trigger_kind)
        if not candidates:
            return None
        query = situation_hint or json.dumps(readings, sort_keys=True)[:500]
        vec = self._embed(query)
        ranked = candidates
        if vec:
            try:
                import sqlite_vec  # type: ignore

                blob = sqlite_vec.serialize_float32(vec)
                overscan = max(len(candidates) * 2, 8)
                with self._lock:
                    rows = self._connect().execute(
                        """SELECT t.id, v.distance AS dist FROM learned_policies_vec v
                           JOIN learned_policies t ON t.id = v.id
                           WHERE v.embedding MATCH ? AND v.k = ?
                             AND t.user_id = ? AND t.status = 'active'
                           ORDER BY v.distance ASC LIMIT ?""",
                        (blob, overscan, self._user_id, overscan),
                    ).fetchall()
                order = {r["id"]: float(r["dist"]) for r in rows}
                ranked = sorted(candidates, key=lambda p: order.get(p["id"], 2.0))
            except Exception as e:
                log.debug("policy: vector rank failed: %s", e)
        for p in ranked:
            if evaluate_condition(p.get("condition") or {}, readings):
                return p
        return None

    # ── outcome ──

    def record_outcome(self, pid: str, success: bool) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "UPDATE learned_policies SET use_count = use_count + 1,"
                    " success_count = success_count + ?, updated_at = ? WHERE id = ?",
                    (1 if success else 0, datetime.now(timezone.utc).isoformat(), pid),
                )
        except Exception as e:
            log.debug("policy: record_outcome failed: %s", e)

    # ── proposals (two-turn teaching confirmation) ──

    # Proposals expire after an hour without confirmation.
    PROPOSAL_TTL_S = 3600.0

    def propose(self, draft: dict) -> tuple[str, str]:
        """Stage a taught policy; returns (proposal_id, confirmation_text)."""
        prop_id = uuid.uuid4().hex[:8]
        with self._lock:
            self._proposals[prop_id] = (time.time(), draft)
        prev = None
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT version FROM learned_policies WHERE user_id = ? AND name = ?"
                " AND status = 'active' ORDER BY version DESC LIMIT 1",
                (self._user_id, draft["name"]),
            ).fetchone()
            if row:
                prev = int(row["version"])
        cond = draft.get("condition") or {}
        cond_txt = (
            f"{cond.get('metric')} {cond.get('op')} {cond.get('value')}"
            if cond.get("metric") else "when it triggers"
        )
        if prev:
            text = (f"Got it — updating '{draft['name']}' to v{prev + 1}: "
                    f"{draft['action']} when {cond_txt}. Say yes to confirm.")
        else:
            text = (f"Got it — new policy '{draft['name']}': {draft['action']} "
                    f"when {cond_txt} ({draft.get('approval', 'notify')}). Say yes to confirm.")
        return prop_id, text

    def confirm_proposal(self, prop_id: str) -> dict | None:
        with self._lock:
            item = self._proposals.pop(prop_id, None)
        if not item:
            return None
        _, draft = item
        return self.teach(**draft)

    def pending_proposals(self) -> list[tuple[str, dict]]:
        now = time.time()
        with self._lock:
            live = {
                pid: (ts, d) for pid, (ts, d) in self._proposals.items()
                if now - ts < self.PROPOSAL_TTL_S
            }
            self._proposals = live
            return [(pid, d) for pid, (_, d) in live.items()]

    # ── DAG compilation ──

    def compile_dag(self, policy: dict):
        """Compile a policy into an executable sense→check→act→verify→notify DAG."""
        from agentic.graph_engine import PlanGraph, PlanNode

        cond = policy.get("condition") or {}
        params = policy.get("action_params") or {}
        name = policy["name"]
        nodes = [
            PlanNode(
                id="sense",
                tool="policy_readings",
                args={"trigger_kind": policy.get("trigger_kind", "event"),
                      "metrics": [cond.get("metric")] if cond.get("metric") else []},
            ),
            PlanNode(
                id="check",
                tool="policy_check",
                args={"policy_id": policy["id"], "condition": cond},
                depends_on=("sense",),
            ),
            PlanNode(
                id="act",
                tool=policy["action"],
                args=params,
                depends_on=("check",),
                run_if={"policy_check.ok": True},
                needs_approval=(policy.get("approval") == "ask"),
            ),
            PlanNode(
                id="verify",
                tool="policy_verify",
                args={"policy_id": policy["id"], "wait_s": params.get("wait_s", 60)},
                depends_on=("act",),
            ),
            PlanNode(
                id="notify",
                tool="policy_notify",
                args={"policy_id": policy["id"], "policy_name": name,
                      "approval": policy.get("approval", "notify")},
                depends_on=("verify",),
            ),
        ]
        return PlanGraph(
            id=f"policy-{policy['id']}",
            name=f"policy:{name}",
            goal=f"Execute learned policy '{name}': {policy.get('situation', '')}",
            nodes=tuple(nodes),
            source="policy",
        )

    # ── action dispatch ──

    def register_action(self, name: str, fn: Callable[[dict], None]) -> None:
        self._action_handlers[name] = fn

    def dispatch(self, policy: dict, readings: dict) -> None:
        """Execute a matched policy: DAG when possible, handler fallback."""
        approval = policy.get("approval", "notify")
        if approval == "ask":
            log.info("policy: '%s' needs approval — queuing ask", policy["name"])
            self._handle_ask(policy, readings)
            return
        handler = self._action_handlers.get(policy["action"])
        if handler:
            try:
                handler({"policy": policy, "readings": readings})
                self.record_outcome(policy["id"], True)
            except Exception as e:
                log.warning("policy: action '%s' failed: %s", policy["action"], e)
                self.record_outcome(policy["id"], False)
        else:
            log.info("policy: no handler for action '%s' — logged only", policy["action"])
            self.record_outcome(policy["id"], True)

    def _handle_log(self, ctx: dict) -> None:
        p = ctx["policy"]
        log.info("policy: '%s' fired on %s", p["name"], ctx["readings"])

    def _handle_notify(self, ctx: dict) -> None:
        p = ctx["policy"]
        log.info("policy: NOTIFY '%s' — %s | readings=%s",
                 p["name"], p.get("situation"), ctx["readings"])

    def _handle_ask(self, policy: dict, readings: dict) -> None:
        log.info("policy: ASK Oppa — '%s' wants to fire on %s", policy["name"], readings)

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass


# ── chat teaching extraction ───────────────────────────────────────────

_TEACH_RE = re.compile(
    r"\b(if|when|whenever)\b.{0,80}\b(then|should|rest|stop|shutdown|notify|nudge|remind)\b"
    r"|\bthreshold\b|\bfrom now on\b|\bmake it\b|\bchange (the|it|this)\b",
    re.IGNORECASE,
)

_TEACH_SYSTEM = (
    "You extract behavior policies Oppa is teaching Aiko. Reply with strict JSON only, "
    "no prose. Fields: is_teaching (bool), name (snake_case short), "
    "situation (one line), trigger_kind (sensor|chat|time|event), "
    "condition {metric, op (one of >,<,>=,<=,==,!=), value (number)}, "
    "action (snake_case verb phrase), approval (autonomous|notify|ask). "
    'If not teaching, reply {"is_teaching": false}.'
)


def looks_like_teaching(text: str) -> bool:
    """Cheap heuristic pre-filter before the LLM extraction."""
    return bool(_TEACH_RE.search(text or ""))


def extract_teaching(user_text: str, llm_client=None) -> dict | None:
    """LLM extraction of a taught policy. Returns draft dict or None."""
    if not looks_like_teaching(user_text):
        return None
    try:
        if llm_client is None:
            from openai import OpenAI

            llm_client = OpenAI(
                base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
                api_key=os.getenv("LLM_API_KEY", "") or "not-needed",
            )
        model = os.getenv("INNER_SPEECH_MODEL", "").strip() or os.getenv("LLM_MODEL", "ministral")
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _TEACH_SYSTEM},
                {"role": "user", "content": (user_text or "")[:800]},
            ],
            max_tokens=256,
            temperature=0.0,
            timeout=30,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        if not data.get("is_teaching"):
            return None
        cond = data.get("condition") or {}
        return {
            "name": str(data.get("name", "unnamed_policy"))[:64],
            "situation": str(data.get("situation", user_text[:200]))[:500],
            "trigger_kind": data.get("trigger_kind", "event"),
            "condition": {
                "metric": cond.get("metric", ""),
                "op": cond.get("op", ">"),
                "value": float(cond.get("value", 0)),
            },
            "action": str(data.get("action", "notify"))[:64],
            "approval": data.get("approval", "notify"),
            "teacher": "oppa",
        }
    except Exception as e:
        log.debug("policy: extract_teaching failed: %s", e)
        return None
