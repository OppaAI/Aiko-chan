"""SQLite-backed checkpointing for graph runs (agentic/graph_engine.py).

A run_id identifies one invocation of execute_graph. If interrupted,
calling execute_graph again with the same run_id resumes from the last
completed node instead of restarting the whole graph.
"""
import json
import os
import tempfile
import sqlite3
import threading
from pathlib import Path

from system.log import get_logger
from system.userspace import current_user_id, user_state_dir

log = get_logger(__name__)

_lock = threading.Lock()


def _db_path() -> Path:
    return user_state_dir(current_user_id()) / "agentic" / "checkpoints.db"


def _open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


def _get_conn() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_migrated()
    conn = _open_db(path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS node_checkpoints (
                run_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                tool TEXT NOT NULL,
                ok INTEGER NOT NULL,
                content TEXT NOT NULL,
                args TEXT NOT NULL,
                error_type TEXT,
                seq INTEGER NOT NULL,
                state_json TEXT DEFAULT '{}',
                PRIMARY KEY (run_id, node_id)
            )
        """)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(node_checkpoints)")}
        if "state_json" not in cols:
            conn.execute("ALTER TABLE node_checkpoints ADD COLUMN state_json TEXT DEFAULT '{}'")
            conn.commit()
    except BaseException:
        conn.close()
        raise
    return conn


def _ensure_migrated() -> None:
    """Ensure checkpoint DB is in user-space location, migrating if needed."""
    old_path = Path(__file__).parent / "checkpoint.db"
    new_path = _db_path()

    if not old_path.exists():
        return

    if new_path.exists():
        return

    from system.userspace import resolve_owner_user_id
    if current_user_id() != resolve_owner_user_id():
        return  # a guest must never receive the legacy owner's checkpoints
    new_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-migrate-", dir=new_path.parent)
    os.close(fd)
    try:
        source = sqlite3.connect(old_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            target = sqlite3.connect(temporary)
            try:
                source.backup(target)  # includes committed data still in WAL
            finally:
                target.close()
        finally:
            source.close()
        try:
            os.link(temporary, new_path)  # publish only if nobody created it
        except FileExistsError:
            pass
        log.info("[checkpoint] copied legacy checkpoints; original retained at %s", old_path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_node_result(run_id: str, seq: int, result, state_json: str = "{}") -> None:
    """Persist one NodeResult and an optional state snapshot."""
    safe_args = {
        k: v for k, v in result.args.items()
        if k not in {"embedder", "client", "model"}
    }
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO node_checkpoints "
                "(run_id, node_id, tool, ok, content, args, error_type, seq, state_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, result.node_id, result.tool, int(result.ok),
                 result.content, json.dumps(safe_args, default=str), result.error_type, seq, state_json),
            )
            conn.commit()
        finally:
            conn.close()


def load_checkpoint(run_id: str, node_result_cls) -> list:
    """Returns completed NodeResults for run_id, in original order. Empty if none."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT node_id, tool, ok, content, args, error_type, state_json "
                "FROM node_checkpoints WHERE run_id = ? AND node_id != '__graph_state__' ORDER BY seq", (run_id,)
            ).fetchall()
        finally:
            conn.close()
    out = []
    for r in rows:
        node_id, tool, ok, content, args_json, error_type, state_json = r
        nr = node_result_cls(node_id=node_id, tool=tool, ok=bool(ok), content=content,
                              args=json.loads(args_json), error_type=error_type, checkpoint_state=state_json)
        out.append(nr)
    return out


def save_graph_state(run_id: str, state: dict) -> None:
    """Persist a standalone state snapshot keyed by run_id.

    Non-serializable values (threading.Lock, live handles) are coerced to
    their repr so a tool that stashes one in state can never kill the graph
    run with a ``TypeError: ... is not JSON serializable`` at snapshot time.
    """
    state = state or {}
    with _lock:
        conn = _get_conn()
        try:
            try:
                state_json = json.dumps(state)
            except TypeError:
                state_json = json.dumps(state, default=str)
            conn.execute(
                "INSERT OR REPLACE INTO node_checkpoints "
                "(run_id, node_id, tool, ok, content, args, error_type, seq, state_json) "
                "VALUES (?, '__graph_state__', '', 1, '', '{}', NULL, 0, ?)",
                (run_id, state_json),
            )
            conn.commit()
        finally:
            conn.close()


def load_graph_state(run_id: str) -> dict:
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT state_json FROM node_checkpoints "
                "WHERE run_id = ? AND node_id = '__graph_state__'", (run_id,)
            ).fetchone()
        finally:
            conn.close()
    if row:
        return json.loads(row[0])
    return {}


def delete_node_checkpoint(run_id: str, node_id: str) -> None:
    """Remove one stale checkpoint entry during loop-back reset."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "DELETE FROM node_checkpoints WHERE run_id = ? AND node_id = ?",
                (run_id, node_id),
            )
            conn.commit()
        finally:
            conn.close()


def clear_checkpoint(run_id: str) -> None:
    """Call after a run completes successfully."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM node_checkpoints WHERE run_id = ?", (run_id,))
            conn.commit()
        finally:
            conn.close()
