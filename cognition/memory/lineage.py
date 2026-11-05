"""Phase 19 — supersession lineage walk for Studio / GET /api/memory/{id}/lineage."""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Protocol


def _max_depth() -> int:
    try:
        return max(1, int(os.getenv("MEMORY_LINEAGE_MAX_DEPTH", "32")))
    except ValueError:
        return 32


class _Store(Protocol):
    def get_by_id(self, mem_id: str, user_id: str | None = None) -> dict | None: ...
    # Optional: implement with raw SQL if get_by_id does not exist yet.


def _row_public(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "memory", "created_at", "status", "supersedes_id",
        "pinned", "kind", "source", "valence_score", "arousal_score",
        "entities", "access_count", "superseded_by", "valid_from", "valid_to",
    )
    return {k: row.get(k) for k in keys if k in row or row.get(k) is not None}


def walk_supersession_lineage(
    store: Any,
    mem_id: str,
    *,
    user_id: str | None = None,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Return ordered supersession chain around ``mem_id``.

    - ``chain``: oldest → newest (root predecessor … current … successors)
    - ``center_id``: requested id
    - Walks ``supersedes_id`` backward (this → older) then forward (who points here)

    Requires the store to support loading a row by id for ``user_id``.
    If ``get_by_id`` is missing, implement via:

        SELECT * FROM memories WHERE id=? AND user_id=?
    """
    depth = max(1, min(256, int(max_depth if max_depth is not None else _max_depth())))
    uid = user_id
    conn = getattr(store, '_conn', None)
    lock = getattr(store, '_db_lock', None)
    cache = {}

    def load(mid):
        if mid in cache:
            return cache[mid]
        if hasattr(store, 'get_by_id'):
            row = store.get_by_id(mid, user_id=uid)
        elif callable(getattr(store, '_get_memory_row', None)):
            row = store._get_memory_row(mid, user_id=uid)
        elif conn is not None:
            with lock or nullcontext():
                raw = conn.execute('SELECT * FROM memories WHERE id=? AND (? IS NULL OR user_id=?)',
                                   (mid, uid, uid)).fetchone()
            row = dict(raw) if raw else None
        else:
            raise AttributeError('store must support memory row lookup')
        cache[mid] = row
        return row

    center = load(mem_id)
    if not center:
        return {'center_id': mem_id, 'chain': [], 'error': 'not_found'}
    uid = uid or center.get('user_id')
    temporal = False
    if conn is not None:
        with lock or nullcontext():
            temporal = any(row[1] == 'superseded_by' for row in conn.execute('PRAGMA table_info(memories)'))

    seen = {str(center.get('id') or mem_id)}
    backward = []
    frontier = [center]
    while frontier and len(seen) < depth:
        following = []
        for row in frontier:
            ids = [row['supersedes_id']] if row.get('supersedes_id') else []
            if temporal:
                with lock or nullcontext():
                    ids.extend(r[0] for r in conn.execute(
                        'SELECT id FROM memories WHERE superseded_by=? AND (? IS NULL OR user_id=?) '
                        'ORDER BY created_at,id LIMIT ?', (row['id'], uid, uid, depth)))
            for mid in ids:
                if len(seen) >= depth:
                    break
                if mid in seen:
                    continue
                previous = load(mid)
                if previous:
                    seen.add(mid)
                    backward.append(previous)
                    following.append(previous)
        frontier = following
    backward.reverse()

    forward = []
    frontier = [center]
    while frontier and len(seen) < depth:
        following = []
        for row in frontier:
            children = _find_superseding(store, row['id'], user_id=uid)
            if row.get('superseded_by'):
                child = load(str(row['superseded_by']))
                if child:
                    children.append(child)
            for child in children:
                mid = str(child.get('id') or '')
                if len(seen) >= depth:
                    break
                if mid and mid not in seen:
                    seen.add(mid)
                    # Load full temporal fields for the next hop.
                    child = load(mid) or child
                    forward.append(child)
                    following.append(child)
        frontier = following
    chain = [_row_public(row) for row in backward + [center] + forward]
    return {'center_id': str(center['id']), 'chain': chain, 'depth': len(chain)}


def _find_superseding(store: Any, mem_id: str, user_id: str | None) -> list[dict]:
    """Return rows whose supersedes_id == mem_id."""
    if hasattr(store, "find_by_supersedes"):
        return list(store.find_by_supersedes(mem_id, user_id=user_id) or [])
    # Raw SQL fallback on backend connection
    conn = getattr(store, "_conn", None)
    lock = getattr(store, "_db_lock", None)
    if conn is None:
        return []
    sql = """
        SELECT id, memory, created_at, status, supersedes_id, pinned,
               kind, source, valence_score, arousal_score, entities, access_count
        FROM memories
        WHERE supersedes_id = ? AND (? IS NULL OR user_id = ?)
        ORDER BY created_at ASC
        LIMIT 16
    """
    args = (mem_id, user_id, user_id)
    if lock is not None:
        with lock:
            rows = conn.execute(sql, args).fetchall()
    else:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]
