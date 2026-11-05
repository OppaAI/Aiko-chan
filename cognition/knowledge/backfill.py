"""Repair missing knowledge vectors in bounded batches after embedder outages."""
import numpy as np

from cognition import reason
from cognition.memory.vecstore import insert_vector
from system.log import get_logger
from system.userspace import current_user_id
from . import schema
from .search import maybe_clear_knowledge_cache

log = get_logger(__name__)


def backfill_missing_vectors(*, embedder, user_id=None, limit=32):
    if embedder is None or limit <= 0:
        return 0
    uid = user_id or current_user_id()
    conn = schema.connect(uid)
    try:
        rows = conn.execute(
            "SELECT c.id,c.text FROM learned_chunks c LEFT JOIN learned_chunks_vec v ON v.id=c.id "
            "WHERE c.user_id=? AND v.id IS NULL AND (c.status='active' OR c.status IS NULL) "
            "ORDER BY c.created_at,c.id LIMIT ?", (uid, min(128, int(limit)))).fetchall()
    finally:
        conn.close()
    if not rows:
        return 0
    # No connection or writer transaction is held during remote model work.
    batch = reason.embed_batch_or_none(embedder, [row['text'] for row in rows])
    if batch is None or len(batch) != len(rows):
        return 0
    valid = []
    for row, value in zip(rows, batch):
        try:
            vector = np.asarray(value, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if (vector.shape != (schema.EMBED_DIMS,) or not np.isfinite(vector).all()
                    or not np.isfinite(norm) or norm <= 1e-12):
                continue
            valid.append((row, vector))
        except (TypeError, ValueError, OverflowError):
            continue
    if not valid:
        return 0
    conn = schema.connect(uid)
    written = 0
    try:
        conn.execute('BEGIN IMMEDIATE')
        for row, vector in valid:
            current = conn.execute(
                "SELECT c.text FROM learned_chunks c LEFT JOIN learned_chunks_vec v ON v.id=c.id "
                "WHERE c.id=? AND c.user_id=? AND v.id IS NULL "
                "AND (c.status='active' OR c.status IS NULL)", (row['id'], uid)).fetchone()
            if current is not None and current['text'] == row['text']:
                insert_vector(conn, 'learned_chunks_vec', row['id'], vector)
                written += 1
        conn.commit()
    except Exception:
        conn.rollback()
        log.exception('Knowledge vector backfill failed; text recall remains available')
        return 0
    finally:
        conn.close()
    if written:
        maybe_clear_knowledge_cache()
    return written
