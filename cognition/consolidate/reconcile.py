"""Sleep-time memory reconciliation.

Letta-style consolidation + Mem0-style reconcile-before-write, adapted for
Aiko's sqlite-vec memory. Runs inside the nightly dream() window (see
attach_dream_hook): selects active fact/identity memories the job hasn't
examined recently, clusters near-duplicates with the existing vec0 KNN,
asks the consolidation LLM for keep/merge/supersede actions per cluster,
and applies them with temporal-validity bookkeeping (valid_from / valid_to
/ superseded_by) so history is never deleted — the currently-true view is a
query over open validity windows.

Pure and testable: run_reconcile() takes a memorize backend and returns a
result dict. It never raises out of the apply phase in a way that breaks
callers — per-cluster failures are logged and skipped.
"""
from __future__ import annotations

import textwrap
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

from system.config import env_bool, env_float, env_int
from system.log import get_logger
from system.userspace import current_display_name

from cognition.memory.schema import (
    KIND_FACT,
    MEM_TYPE_SEMANTIC,
    SOURCE_DREAM,
    _close_row_temporal,
    _sqlite_get_vector,
    _sqlite_knn_search,
    existing_columns,
    parse_json_array,
)

from .schema import _chat

log = get_logger(__name__)

__all__ = [
    "MEMORY_RECONCILE_BATCH",
    "MEMORY_RECONCILE_ENABLED",
    "MEMORY_RECONCILE_MAX_CLUSTERS",
    "MEMORY_RECONCILE_MAX_LLM_CALLS",
    "MEMORY_RECONCILE_RESCAN_DAYS",
    "MEMORY_RECONCILE_SIM",
    "attach_dream_hook",
    "run_reconcile",
]

# ── env tunables (system.config, like the rest of the codebase) ───────────────

MEMORY_RECONCILE_ENABLED = env_bool("MEMORY_RECONCILE_ENABLED", "1")
MEMORY_RECONCILE_RESCAN_DAYS = max(0, env_int("MEMORY_RECONCILE_RESCAN_DAYS", 30))
MEMORY_RECONCILE_BATCH = max(1, env_int("MEMORY_RECONCILE_BATCH", 200))
MEMORY_RECONCILE_MAX_CLUSTERS = max(1, env_int("MEMORY_RECONCILE_MAX_CLUSTERS", 20))
MEMORY_RECONCILE_MAX_LLM_CALLS = max(1, env_int("MEMORY_RECONCILE_MAX_LLM_CALLS", 20))
MEMORY_RECONCILE_SIM = min(0.99, max(0.0, env_float("MEMORY_RECONCILE_SIM", 0.75)))

_KNN_NEIGHBORS = 10  # neighbors probed per candidate for clustering


def _backend_of(memorize):
    """Normalize AikoMemorize / _MemoryBackend to the backend object."""
    return getattr(memorize, "_mem", memorize)


def _majority(values) -> str:
    vals = [v for v in values if v]
    if not vals:
        return MEM_TYPE_SEMANTIC
    return Counter(vals).most_common(1)[0][0]


def _candidate_rows(conn, user_id: str) -> list:
    """Active, unpinned fact/identity memories due for re-examination."""
    cols = existing_columns(conn)
    mem_type_sel = "m.mem_type" if "mem_type" in cols else "NULL"
    sql = (
        "SELECT m.id AS id, m.memory AS memory, m.kind AS kind, "
        f"{mem_type_sel} AS mem_type, m.created_at AS created_at "
        "FROM memories m "
        "WHERE m.user_id = ? AND (m.status = 'active' OR m.status IS NULL) "
        "AND m.kind IN ('fact', 'identity') "
        "AND (m.pinned IS NULL OR m.pinned = 0)"
    )
    params: list = [user_id]
    if "reconciled_at" in cols:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=MEMORY_RECONCILE_RESCAN_DAYS)
        ).isoformat()
        sql += " AND (m.reconciled_at IS NULL OR m.reconciled_at < ?)"
        params.append(cutoff)
    sql += " ORDER BY m.created_at ASC LIMIT ?"
    params.append(MEMORY_RECONCILE_BATCH)
    rows = conn.execute(sql, params).fetchall()
    return rows


def _cluster_candidates(conn, user_id: str, rows: list) -> tuple[list[list[str]], list[str]]:
    """Union-find clusters (size >= 2) of near-duplicate candidates.

    Uses stored vectors via _sqlite_get_vector — no re-embedding. Returns
    (clusters, vectorless_ids): candidates with no stored vector can't
    cluster but are still stamped reconciled_at by the caller.
    """
    vecs: dict[str, list[float]] = {}
    vectorless: list[str] = []
    for r in rows:
        mid = str(r["id"])
        v = _sqlite_get_vector(conn, mid)
        if v:
            vecs[mid] = v
        else:
            vectorless.append(mid)

    parent = {mid: mid for mid in vecs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for mid, v in vecs.items():
        try:
            hits = _sqlite_knn_search(
                conn, v, user_id,
                limit=_KNN_NEIGHBORS,
                threshold=MEMORY_RECONCILE_SIM,
                active_only=True,
            )
        except Exception as e:
            log.debug("reconcile: knn failed for %s: %s", mid, e)
            continue
        for h in hits:
            hid = str(h["id"])
            if hid in vecs and hid != mid:
                ra, rb = find(mid), find(hid)
                if ra != rb:
                    parent[rb] = ra

    groups: dict[str, list[str]] = {}
    for mid in vecs:
        groups.setdefault(find(mid), []).append(mid)
    clusters = [sorted(m) for m in groups.values() if len(m) >= 2]
    return clusters, vectorless


_RECONCILE_SYSTEM = textwrap.dedent("""
    You are reconciling a small cluster of near-duplicate memory facts about {USER_ID}.

    Rules:
    - If the facts agree and describe the same thing, merge them into ONE combined fact (short, self-contained, third person, about {USER_ID}).
    - If one fact contradicts or replaces an older one, keep the NEWER fact (use the created timestamp) and supersede the older one. Never drop the newer fact.
    - If the facts are actually distinct (different topics that only looked similar), keep all of them unchanged.
    - Preserve exact dates, names, and numbers. Never invent details not present in the source facts.

    Return ONLY a JSON array of actions covering the cluster:
      [{{"action":"keep"}}]
      [{{"action":"merge","text":"...","source_ids":["id1","id2"]}}]
      [{{"action":"supersede","old_id":"...","new_text":"..."}}]
    Use "keep" when every fact is distinct. A merge references 2+ source_ids;
    a supersede closes exactly one old fact. Every input id must be covered by
    exactly one action. No markdown, no explanation.
    (Note: doubled braces above are str.format escaping for the literal JSON
    example — without them .format() raises KeyError.)
""").strip()


def _decide_cluster_actions(members: list) -> list[dict]:
    """Ask the consolidation LLM for keep/merge/supersede actions (facts.py style)."""
    lines = []
    for m in members:
        mid = str(m["id"])
        created = str(m["created_at"] or "")
        text = (str(m["memory"]) or "").strip().replace("\n", " ")
        lines.append(f"id={mid} created={created} :: {text[:400]}")
    user_prompt = "Cluster facts:\n" + "\n".join(lines)
    raw = _chat(
        _RECONCILE_SYSTEM.format(USER_ID=current_display_name()),
        user_prompt,
        max_tokens=800,
        temperature=0.1,
    )
    arr = parse_json_array(raw)
    if not arr:
        return [{"action": "keep"}]
    clean: list[dict] = []
    for a in arr:
        if not isinstance(a, dict):
            continue
        act = str(a.get("action", "")).strip().lower()
        if act in ("keep", "merge", "supersede"):
            clean.append({"action": act, **{k: v for k, v in a.items() if k != "action"}})
    return clean or [{"action": "keep"}]


def _apply_cluster(backend, user_id: str, members: list, actions: list[dict],
                   now: str, result: dict) -> None:
    """Apply one cluster's actions. Never raises — caller logs per cluster."""
    conn = backend._conn
    by_id = {str(m["id"]): m for m in members}
    member_ids = set(by_id)
    covered: set[str] = set()

    for a in actions:
        act = a.get("action")
        if act == "merge":
            src_ids = [
                str(s) for s in (a.get("source_ids") or [])
                if str(s) in member_ids and str(s) not in covered
            ]
            text = str(a.get("text") or "").strip()
            if len(src_ids) < 2 or not text:
                continue
            mem_type = _majority(str(by_id[s]["mem_type"] or "") for s in src_ids)
            kind = _majority(str(by_id[s]["kind"] or "") for s in src_ids) or KIND_FACT
            try:
                vector = backend._embed(text)
            except Exception as e:
                log.debug("reconcile: merge embed failed: %s", e)
                vector = None
            new_id = str(uuid.uuid4())
            backend._insert_row(
                mem_id=new_id, user_id=user_id, text=text, now=now,
                vector=vector, pinned=0, source=SOURCE_DREAM,
                kind=kind, mem_type=mem_type, valid_from=now,
            )
            for s in src_ids:
                _close_row_temporal(conn, s, new_id, user_id, now)
                covered.add(s)
            result["merged"] += 1
            result["superseded"] += len(src_ids)
        elif act == "supersede":
            old_id = str(a.get("old_id") or "")
            new_text = str(a.get("new_text") or "").strip()
            if old_id not in member_ids or old_id in covered or not new_text:
                continue
            new_id = backend.supersede_with_temporal(old_id, new_text, user_id)
            if new_id:
                covered.add(old_id)
                result["superseded"] += 1
        # "keep" and unknown actions: nothing to do.

    # Stamp every examined member so the job makes progress even on keep/noop.
    for mid in member_ids:
        try:
            conn.execute(
                "UPDATE memories SET reconciled_at = ? WHERE id = ?", (now, mid)
            )
        except Exception as e:
            log.debug("reconcile: reconciled_at stamp failed for %s: %s", mid, e)
    try:
        conn.commit()
    except Exception:
        pass
    result["kept"] += len(member_ids - covered)


def _stamp_reconciled(conn, mem_ids: list[str], now: str) -> None:
    for mid in mem_ids:
        try:
            conn.execute(
                "UPDATE memories SET reconciled_at = ? WHERE id = ?", (now, mid)
            )
        except Exception as e:
            log.debug("reconcile: reconciled_at stamp failed for %s: %s", mid, e)
    try:
        conn.commit()
    except Exception:
        pass


def run_reconcile(memorize, user_id: str, now: str | None = None,
                  dry_run: bool = False) -> dict:
    """Run one sleep-time reconcile pass. Pure and testable.

    Returns a result dict; dry_run returns planned actions without writing.
    Never raises out of the apply phase in a way that breaks callers.
    """
    now = now or datetime.now(timezone.utc).isoformat()
    result: dict = {
        "candidates": 0,
        "clusters": 0,
        "merged": 0,
        "superseded": 0,
        "kept": 0,
        "llm_calls": 0,
        "dry_run": bool(dry_run),
    }
    try:
        uid = memorize._resolve_user_id(user_id)
    except Exception:
        uid = user_id
    backend = _backend_of(memorize)
    conn = backend._conn
    lock = getattr(backend, "_db_lock", None)

    def _locked(fn, *a, **k):
        if lock is not None:
            with lock:
                return fn(*a, **k)
        return fn(*a, **k)

    try:
        rows = _locked(_candidate_rows, conn, uid)
    except Exception as e:
        log.warning("reconcile: candidate selection failed: %s", e)
        result["error"] = str(e)
        return result
    result["candidates"] = len(rows)
    if not rows:
        return result

    try:
        clusters, vectorless = _locked(_cluster_candidates, conn, uid, rows)
    except Exception as e:
        log.warning("reconcile: clustering failed: %s", e)
        result["error"] = str(e)
        return result

    clusters = clusters[:MEMORY_RECONCILE_MAX_CLUSTERS]
    row_by_id = {str(r["id"]): r for r in rows}

    for cluster in clusters:
        members = [row_by_id[mid] for mid in cluster if mid in row_by_id]
        if len(members) < 2:
            continue
        if result["llm_calls"] >= MEMORY_RECONCILE_MAX_LLM_CALLS:
            log.info("reconcile: LLM call budget exhausted")
            break
        result["clusters"] += 1
        result["llm_calls"] += 1
        try:
            actions = _decide_cluster_actions(members)
        except Exception as e:
            log.warning("reconcile: cluster decision failed: %s", e)
            continue
        if dry_run:
            result.setdefault("planned", []).append(
                {"cluster": cluster, "actions": actions}
            )
            continue
        try:
            _locked(_apply_cluster, backend, uid, members, actions, now, result)
        except Exception:
            log.exception("reconcile: apply failed for cluster %s", cluster)
            continue

    # Stamp every examined candidate (clustered, vectorless, or lone) so the
    # job makes progress and doesn't re-select the same rows every night.
    # _apply_cluster already stamped clustered members; re-stamping is a
    # harmless no-op. dry_run writes nothing.
    if not dry_run:
        try:
            _locked(_stamp_reconciled, conn,
                    [str(r["id"]) for r in rows], now)
        except Exception as e:
            log.debug("reconcile: final stamp failed: %s", e)
    result["examined"] = len(rows)
    log.info(
        "reconcile: candidates=%d clusters=%d merged=%d superseded=%d "
        "kept=%d llm_calls=%d dry_run=%s",
        result["candidates"], result["clusters"], result["merged"],
        result["superseded"], result["kept"], result["llm_calls"], dry_run,
    )
    return result


def attach_dream_hook() -> None:
    """Wrap AikoMemorize.dream to run sleep-time reconcile after the nightly pass.

    EMC-4 precedent (cognition.memory.episode.attach_dream_hook): idempotent
    monkey-patch, fail-soft, dry_run passes through. The
    MEMORY_RECONCILE_ENABLED gate is read at run time so it can be toggled
    without a reboot.
    """
    from cognition.memory.memorize import AikoMemorize, DREAM_MERGE_THRESHOLD

    if getattr(AikoMemorize, "_reconcile_dream_patched", False):
        return

    _orig = AikoMemorize.dream

    def dream(self, user_id=None, dry_run=False, threshold=None, **kwargs):
        if threshold is None:
            threshold = DREAM_MERGE_THRESHOLD
        result = _orig(self, user_id=user_id, dry_run=dry_run,
                       threshold=threshold, **kwargs)
        try:
            if env_bool("MEMORY_RECONCILE_ENABLED", "1"):
                rec = run_reconcile(self, user_id=user_id, dry_run=dry_run)
                if isinstance(result, dict):
                    result["reconcile"] = rec
        except Exception as e:
            log.warning("memory reconcile after dream failed: %s", e)
            if isinstance(result, dict):
                result["reconcile_error"] = str(e)
        return result

    AikoMemorize.dream = dream  # type: ignore[method-assign]
    AikoMemorize._reconcile_dream_patched = True  # type: ignore[attr-defined]
    log.debug("reconcile dream hook attached")
