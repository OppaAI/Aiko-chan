"""The conscience ledger — every verdict, written down.

Three jobs, in order of importance:

1. **Audit.** A moral gate you cannot inspect after the fact is a moral gate
   you cannot trust. Every row records the decision, the axis scores, the norms
   cited, the guardrail rules hit, and the canon version in force — so a
   decision can be re-litigated months later against the constitution that
   actually judged it.

2. **HITL state.** Escalations live here as `pending` rows until a human
   approves, denies, or lets them time out. The timeout resolves to `refuse`
   (see schema.HITL_DEFAULT) — default-closed, never default-open.

3. **Training harvest.** The rows where the fast judge and the slow judge
   disagreed, and the rows a human personally decided, are the only training
   data that teaches the SLM *your* judgement rather than a generic one.
   `harvest()` pulls exactly those.

Storage is a separate per-user sqlite file (memory/conscience.db) rather than a
table inside memory.db: a conscience record must survive a memory wipe, and
`/clear` should never be able to erase the evidence of what was decided.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

from cognition.memory.vecstore import initialize_store_db
from system.log import get_logger
from system.userspace import current_user_id

from .schema import (
    ESCALATE,
    HITL_DEFAULT,
    HITL_TIMEOUT_SECONDS,
    LEDGER_DB_PATH,
    LEDGER_DDL,
    LEDGER_ENABLED,
    LEDGER_RETAIN_DAYS,
    LEDGER_STORE_CONTENT,
    Verdict,
)

log = get_logger(__name__)

HITL_PENDING = "pending"
HITL_APPROVED = "approved"
HITL_DENIED = "denied"
HITL_TIMEOUT = "timeout"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_digest(content: str) -> str:
    """Stable digest used for dedup and for content-free ledger mode."""
    return hashlib.sha256((content or "").encode("utf-8", "replace")).hexdigest()[:32]


class ConscienceLedger:
    """Per-user ledger. One connection, guarded by a re-entrant lock.

    Same ownership model as _MemoryBackend: the connection is shared across
    the turn thread and any background approval handler, so every statement
    runs under the lock.
    """

    def __init__(self, user_id: str | None = None) -> None:
        self._user_id = user_id or current_user_id()
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # ── connection ────────────────────────────────────────────────────────

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = initialize_store_db(
                LEDGER_DB_PATH, LEDGER_DDL, user_id=self._user_id, vector=False,
            )
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    log.debug("[ccc] ledger close failed")
                self._conn = None

    # ── writing ───────────────────────────────────────────────────────────

    def record(
        self,
        verdict: Verdict,
        *,
        content: str = "",
        surface: str = "",
    ) -> str:
        """Persist one verdict. Returns the row id (also the escalation id).

        Best-effort: a ledger failure logs and returns a generated id rather
        than raising, because a broken audit table must not break the turn it
        was auditing.
        """
        row_id = verdict.escalation_id or uuid.uuid4().hex[:12]
        if not LEDGER_ENABLED:
            return row_id

        row = verdict.to_row()
        hitl_state = HITL_PENDING if verdict.decision == ESCALATE else ""
        try:
            with self._lock:
                self._db().execute(
                    """
                    INSERT INTO conscience_ledger (
                        id, user_id, created_at, act, surface, content_sha, content,
                        decision, gate, vertical, horizontal, confidence,
                        reasons, norms, rule_ids, parties, constraint_txt,
                        canon_version, layers_run, latency_ms, fail_mode, hitl_state
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row_id, self._user_id, _utc_now(), row["act"], surface,
                        content_digest(content),
                        (content[:4000] if LEDGER_STORE_CONTENT else None),
                        row["decision"], row["gate"], row["vertical"],
                        row["horizontal"], row["confidence"],
                        json.dumps(row["reasons"], ensure_ascii=False),
                        json.dumps(row["norms"], ensure_ascii=False),
                        json.dumps(row["rule_ids"], ensure_ascii=False),
                        json.dumps(row["parties"], ensure_ascii=False),
                        row["constraint"], row["canon_version"],
                        json.dumps(row["layers_run"], ensure_ascii=False),
                        row["latency_ms"], row["fail_mode"], hitl_state,
                    ),
                )
                self._db().commit()
        except Exception as exc:
            log.warning("[ccc] ledger write failed: %s", exc)
        return row_id

    # ── human-in-the-loop ─────────────────────────────────────────────────

    def resolve(self, escalation_id: str, state: str, note: str = "") -> bool:
        """Record a human decision on a pending escalation."""
        if state not in (HITL_APPROVED, HITL_DENIED, HITL_TIMEOUT):
            raise ValueError(f"unknown hitl state: {state!r}")
        try:
            with self._lock:
                cur = self._db().execute(
                    """
                    UPDATE conscience_ledger
                    SET hitl_state = ?, hitl_at = ?, hitl_note = ?, reviewed = 1
                    WHERE id = ? AND user_id = ? AND hitl_state = ?
                    """,
                    (state, _utc_now(), note[:500], escalation_id, self._user_id, HITL_PENDING),
                )
                self._db().commit()
                return bool(cur.rowcount)
        except Exception as exc:
            log.warning("[ccc] ledger resolve failed: %s", exc)
            return False

    def pending(self, limit: int = 10) -> list[dict]:
        """Open escalations, oldest first — what the human still owes an answer to."""
        try:
            with self._lock:
                rows = self._db().execute(
                    """
                    SELECT id, created_at, act, surface, decision, reasons, norms,
                           rule_ids, content, vertical, horizontal, confidence
                    FROM conscience_ledger
                    WHERE user_id = ? AND hitl_state = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (self._user_id, HITL_PENDING, int(limit)),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.warning("[ccc] ledger pending query failed: %s", exc)
            return []

    def expire_stale(self) -> int:
        """Time out escalations older than HITL_TIMEOUT_SECONDS.

        This is the default-closed half of the doubt rule: an escalation
        nobody answered becomes HITL_DEFAULT (refuse), never an allow. Called
        opportunistically from core.evaluate() so it needs no daemon thread.
        """
        if HITL_TIMEOUT_SECONDS <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=HITL_TIMEOUT_SECONDS)).isoformat()
        try:
            with self._lock:
                cur = self._db().execute(
                    """
                    UPDATE conscience_ledger
                    SET hitl_state = ?, hitl_at = ?, hitl_note = ?
                    WHERE user_id = ? AND hitl_state = ? AND created_at < ?
                    """,
                    (HITL_TIMEOUT, _utc_now(),
                     f"unanswered for {int(HITL_TIMEOUT_SECONDS)}s; resolved to {HITL_DEFAULT}",
                     self._user_id, HITL_PENDING, cutoff),
                )
                self._db().commit()
                n = int(cur.rowcount or 0)
        except Exception as exc:
            log.warning("[ccc] ledger expiry failed: %s", exc)
            return 0
        if n:
            log.info("[ccc] %d escalation(s) timed out → %s", n, HITL_DEFAULT)
        return n

    # ── diagnostics ───────────────────────────────────────────────────────

    def stats(self, days: int = 30) -> dict:
        """Decision mix over a window. Watch `escalate_rate`.

        Above roughly 2% you have scrupulosity rather than safety, and the fix
        is a narrower canon or a higher REFUSE_AT — not a higher tolerance for
        interruptions.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, days))).isoformat()
        out = {"total": 0, "allow": 0, "caution": 0, "escalate": 0, "refuse": 0,
               "pending": 0, "avg_latency_ms": 0.0}
        try:
            with self._lock:
                rows = self._db().execute(
                    """
                    SELECT decision, COUNT(*) AS n, AVG(latency_ms) AS lat
                    FROM conscience_ledger
                    WHERE user_id = ? AND created_at >= ?
                    GROUP BY decision
                    """,
                    (self._user_id, cutoff),
                ).fetchall()
                pend = self._db().execute(
                    "SELECT COUNT(*) FROM conscience_ledger WHERE user_id = ? AND hitl_state = ?",
                    (self._user_id, HITL_PENDING),
                ).fetchone()
        except Exception as exc:
            log.warning("[ccc] ledger stats failed: %s", exc)
            return out

        weighted = 0.0
        for row in rows:
            decision, n, lat = row["decision"], int(row["n"] or 0), float(row["lat"] or 0.0)
            out[decision] = out.get(decision, 0) + n
            out["total"] += n
            weighted += lat * n
        out["pending"] = int(pend[0]) if pend else 0
        if out["total"]:
            out["avg_latency_ms"] = round(weighted / out["total"], 1)
            out["escalate_rate"] = round(out["escalate"] / out["total"], 4)
            out["refuse_rate"] = round(out["refuse"] / out["total"], 4)
        return out

    # ── training harvest ──────────────────────────────────────────────────

    def harvest(self, limit: int = 2000, *, only_gold: bool = False) -> list[dict]:
        """Rows worth training on, highest signal first.

        Priority order, and the reasoning behind it:
          1. human-resolved escalations — your actual judgement, the gold
          2. rows with hand-entered gold_vertical/gold_horizontal
          3. rows where deliberation (L3) had to overrule the fast judge (L2)
        Ordinary `allow` rows are excluded: training on cases the model already
        gets right teaches it nothing and biases it toward permissiveness.
        """
        clauses = ["user_id = ?"]
        params: list = [self._user_id]
        if only_gold:
            clauses.append("(hitl_state IN ('approved','denied') OR gold_vertical IS NOT NULL)")
        else:
            clauses.append(
                "(hitl_state IN ('approved','denied') "
                " OR gold_vertical IS NOT NULL "
                " OR layers_run LIKE '%deliberate%' "
                " OR decision IN ('refuse','caution','escalate'))"
            )
        sql = f"""
            SELECT id, created_at, act, surface, content, content_sha, decision, gate,
                   vertical, horizontal, confidence, reasons, norms, rule_ids, parties,
                   canon_version, layers_run, hitl_state, hitl_note,
                   gold_vertical, gold_horizontal
            FROM conscience_ledger
            WHERE {' AND '.join(clauses)}
            ORDER BY
                CASE WHEN hitl_state IN ('approved','denied') THEN 0
                     WHEN gold_vertical IS NOT NULL THEN 1
                     WHEN layers_run LIKE '%deliberate%' THEN 2
                     ELSE 3 END,
                created_at DESC
            LIMIT ?
        """
        params.append(int(limit))
        try:
            with self._lock:
                rows = self._db().execute(sql, params).fetchall()
        except Exception as exc:
            log.warning("[ccc] ledger harvest failed: %s", exc)
            return []

        out: list[dict] = []
        for row in rows:
            record = dict(row)
            for key in ("reasons", "norms", "rule_ids", "parties", "layers_run"):
                try:
                    record[key] = json.loads(record.get(key) or "[]")
                except (json.JSONDecodeError, TypeError):
                    record[key] = []
            # A human decision overrides the machine label entirely — that is
            # the whole point of harvesting it.
            if record.get("hitl_state") == HITL_DENIED:
                record["label_vertical"] = min(-0.7, float(record["vertical"]))
                record["label_horizontal"] = min(-0.7, float(record["horizontal"]))
                record["label_confidence"] = 0.95
            elif record.get("hitl_state") == HITL_APPROVED:
                record["label_vertical"] = max(0.0, float(record["vertical"]))
                record["label_horizontal"] = max(0.0, float(record["horizontal"]))
                record["label_confidence"] = 0.9
            else:
                record["label_vertical"] = float(
                    record["gold_vertical"] if record["gold_vertical"] is not None else record["vertical"]
                )
                record["label_horizontal"] = float(
                    record["gold_horizontal"] if record["gold_horizontal"] is not None else record["horizontal"]
                )
                record["label_confidence"] = float(record["confidence"])
            out.append(record)
        return out

    def set_gold(self, row_id: str, vertical: float, horizontal: float, note: str = "") -> bool:
        """Hand-label a row while reviewing. Studio / CLI entry point."""
        try:
            with self._lock:
                cur = self._db().execute(
                    """
                    UPDATE conscience_ledger
                    SET gold_vertical = ?, gold_horizontal = ?, reviewed = 1,
                        hitl_note = CASE WHEN ? = '' THEN hitl_note ELSE ? END
                    WHERE id = ? AND user_id = ?
                    """,
                    (max(-1.0, min(1.0, vertical)), max(-1.0, min(1.0, horizontal)),
                     note, note[:500], row_id, self._user_id),
                )
                self._db().commit()
                return bool(cur.rowcount)
        except Exception as exc:
            log.warning("[ccc] ledger set_gold failed: %s", exc)
            return False

    def prune(self) -> int:
        """Drop rows past LEDGER_RETAIN_DAYS, keeping reviewed ones forever.

        Reviewed rows are training data you paid attention to produce; they
        outlive the retention window on purpose.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=LEDGER_RETAIN_DAYS)).isoformat()
        try:
            with self._lock:
                cur = self._db().execute(
                    "DELETE FROM conscience_ledger "
                    "WHERE user_id = ? AND created_at < ? AND reviewed = 0",
                    (self._user_id, cutoff),
                )
                self._db().commit()
                return int(cur.rowcount or 0)
        except Exception as exc:
            log.warning("[ccc] ledger prune failed: %s", exc)
            return 0


# ── per-user registry ─────────────────────────────────────────────────────────

_ledgers: dict[str, ConscienceLedger] = {}
_ledgers_lock = threading.Lock()


def ledger_for(user_id: str | None = None) -> ConscienceLedger:
    uid = user_id or current_user_id()
    with _ledgers_lock:
        led = _ledgers.get(uid)
        if led is None:
            led = ConscienceLedger(uid)
            _ledgers[uid] = led
            while len(_ledgers) > 8:  # LRU cap, same as the episodic store
                oldest = next(iter(_ledgers))
                if oldest == uid:
                    break
                try:
                    _ledgers.pop(oldest).close()
                except Exception:
                    _ledgers.pop(oldest, None)
        return led


def close_all() -> None:
    """Called from AikoMemorize.switch_user, alongside episodic.close_all()."""
    with _ledgers_lock:
        for led in list(_ledgers.values()):
            led.close()
        _ledgers.clear()


__all__ = [
    "ConscienceLedger", "HITL_APPROVED", "HITL_DENIED", "HITL_PENDING",
    "HITL_TIMEOUT", "close_all", "content_digest", "ledger_for",
]
