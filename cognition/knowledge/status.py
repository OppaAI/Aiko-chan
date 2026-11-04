"""Bounded, per-user library import outcomes in the knowledge database."""
from system.userspace import current_user_id
from . import schema


def record_ingestion_statuses(records, *, user_id=None):
    if not records:
        return
    uid = user_id or current_user_id()
    conn = schema.connect(uid)
    try:
        stamp = schema.now()
        for source, state in records:
            conn.execute(
                "INSERT INTO knowledge_ingestion_status(user_id,source,state,doc_id,last_attempt) "
                "VALUES(?,?,?,(SELECT doc_id FROM knowledge_file_state WHERE user_id=? AND source=?),?) "
                "ON CONFLICT(user_id,source) DO UPDATE SET state=excluded.state,doc_id=excluded.doc_id,"
                "last_attempt=excluded.last_attempt WHERE state IS NOT excluded.state "
                "OR doc_id IS NOT excluded.doc_id OR excluded.state='failed'",
                (uid, source, state, uid, source, stamp))
        conn.execute(
            "DELETE FROM knowledge_ingestion_status WHERE user_id=? AND source NOT IN "
            "(SELECT source FROM knowledge_ingestion_status WHERE user_id=? "
            "ORDER BY last_attempt DESC,source LIMIT 2000)", (uid, uid))
        conn.commit()
    finally:
        conn.close()


def ingestion_status(*, user_id=None, limit=100):
    """Return latest ready/failed/needs_caption library outcomes after restart.

    A failed refresh can still have a doc_id: its prior accepted version remains
    searchable. This reports scan outcomes, not current filesystem availability.
    """
    uid = user_id or current_user_id()
    bound = max(0, min(2000, int(limit)))
    conn = schema.connect(uid)
    try:
        return [dict(row) for row in conn.execute(
            "SELECT source,state,doc_id,last_attempt FROM knowledge_ingestion_status "
            "WHERE user_id=? ORDER BY last_attempt DESC,source LIMIT ?",
            (uid, bound))]
    finally:
        conn.close()
