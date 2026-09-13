"""
Shared experience store for the game backends (shogi / go / koikoi).

This is how Aiko improves without training (impossible on the Nano):
every finished game appends one row; every K games a background
reflection asks the LLM to distill lessons; lessons feed back in two ways:

1. Prompt injection — text lessons are appended to banter/teaching prompts.
2. Numeric biases — small clamped offsets tune per-game AI knobs
   (e.g. blunder rates, koi-koi courage) toward the observed win-rate band.

Storage follows the other game DBs (see lingo_store.py):
    <USER_SPACE_ROOT>/<uid>/agentic/<game>.db
tables: matches (one row per finished game), lessons (single row).

Per-game thin wrappers live next to each backend (shogi/records.py,
go/records.py, koikoi/records.py) and define each game's bias keys,
prompt builder, and record summarizer.

Env knobs (checked with per-game prefix first, then bare):
  <GAME>_LEARNING / KOIKOI_LEARNING ... 1/0 master switch (default 1)
  <GAME>_REFLECT_EVERY / ...             reflect every K games (default 5)
  KOIKOI_DATA_DIR                        override the state root (tests)

All functions are defensive: I/O failures degrade to neutral defaults,
never break a game. Thread-safe enough for one background reflector
(module lock + _reflecting set keyed by uid+game).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

_MAX_TEXT_LESSONS = 5

_lock = threading.Lock()
_reflecting: set[str] = set()
_lessons_cache: dict[str, tuple[float, dict]] = {}


def _env(game: str, key: str, default: str) -> str:
    return (os.getenv(f"{game.upper()}_{key}", os.getenv(key, default)) or default)


def learning_enabled(game: str) -> bool:
    return _env(game, "LEARNING", "1").strip().lower() in {"1", "true", "yes", "on"}


def reflect_every(game: str) -> int:
    try:
        return max(1, int(_env(game, "REFLECT_EVERY", "5")))
    except (TypeError, ValueError):
        return 5


def _state_root() -> Path:
    raw = (os.getenv("KOIKOI_DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    try:
        from system.userspace import _user_state_root_value

        return Path(_user_state_root_value()).expanduser()
    except Exception:
        pass
    try:
        from system.userspace import user_state_dir

        return user_state_dir(None)
    except Exception:
        return Path.home() / ".aiko"


def user_game_db_path(uid: str, game: str) -> Path:
    try:
        from system.userspace import user_state_path

        p = user_state_path(f"agentic/{game}.db", uid)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        p = _state_root() / (uid or "guest") / f"agentic/{game}.db"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _connect(uid: str, game: str) -> sqlite3.Connection:
    path = user_game_db_path(uid, game)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("""CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        difficulty TEXT NOT NULL DEFAULT '',
        span INTEGER NOT NULL DEFAULT 0,
        winner TEXT NOT NULL DEFAULT '',
        you_pts INTEGER NOT NULL DEFAULT 0,
        aiko_pts INTEGER NOT NULL DEFAULT 0,
        moves_made INTEGER NOT NULL DEFAULT 0,
        extra_json TEXT NOT NULL DEFAULT '{}')""")
    con.execute("""CREATE TABLE IF NOT EXISTS lessons (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        updated INTEGER NOT NULL DEFAULT 0,
        match_count INTEGER NOT NULL DEFAULT 0,
        text_json TEXT NOT NULL DEFAULT '[]',
        biases_json TEXT NOT NULL DEFAULT '{}')""")
    return con


def append_match(uid: str, game: str, record: dict[str, Any]) -> None:
    """Append one finished-game record (best effort, never raises)."""
    if not learning_enabled(game):
        return
    try:
        extra = record.get("extra")
        with _connect(uid, game) as con:
            con.execute(
                """INSERT INTO matches
                   (ts, difficulty, span, winner, you_pts, aiko_pts,
                    moves_made, extra_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (int(record.get("ts", time.time())),
                 str(record.get("difficulty") or ""),
                 int(record.get("span") or 0),
                 str(record.get("winner") or ""),
                 int(record.get("you_pts", 0)), int(record.get("aiko_pts", 0)),
                 int(record.get("moves_made", 0)),
                 json.dumps(extra if isinstance(extra, dict) else {},
                            ensure_ascii=False)))
            con.commit()
    except Exception:
        log.debug("learn: append failed for %s", game, exc_info=True)


def total_matches(uid: str, game: str) -> int:
    try:
        with _connect(uid, game) as con:
            row = con.execute("SELECT COUNT(*) FROM matches").fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def load_recent(uid: str, game: str, limit: int = 20) -> list[dict]:
    try:
        with _connect(uid, game) as con:
            rows = con.execute(
                """SELECT ts, difficulty, span, winner, you_pts, aiko_pts,
                          moves_made, extra_json FROM matches
                   ORDER BY id DESC LIMIT ?""",
                (max(1, limit),)).fetchall()
    except Exception:
        return []
    out = []
    for r in rows:
        try:
            extra = json.loads(r[7] or "{}")
        except ValueError:
            extra = {}
        out.append({
            "ts": r[0], "difficulty": r[1], "span": r[2], "winner": r[3],
            "you_pts": r[4], "aiko_pts": r[5], "moves_made": r[6],
            "extra": extra if isinstance(extra, dict) else {},
        })
    return out


def stats(uid: str, game: str, recent: Optional[list[dict]] = None) -> dict[str, Any]:
    """Win-rate summary, overall + per difficulty (unknown winners tracked)."""
    rows = recent if recent is not None else load_recent(uid, game, 200)
    out: dict[str, Any] = {"matches": len(rows), "aiko_wins": 0, "you_wins": 0,
                           "draws": 0, "unknown": 0, "by_difficulty": {}}
    for r in rows:
        w = r.get("winner")
        if w == "aiko":
            out["aiko_wins"] += 1
        elif w == "you":
            out["you_wins"] += 1
        elif w in ("draw",):
            out["draws"] += 1
        else:
            out["unknown"] += 1
        d = r.get("difficulty") or "?"
        bucket = out["by_difficulty"].setdefault(d, {"matches": 0, "aiko_wins": 0})
        bucket["matches"] += 1
        if w == "aiko":
            bucket["aiko_wins"] += 1
    return out


def load_lessons(uid: str, game: str) -> dict:
    """Cached lessons row (text + numeric biases), neutral when absent."""
    try:
        mtime = user_game_db_path(uid, game).stat().st_mtime
    except OSError:
        cached = _lessons_cache.get(f"{uid}:{game}")
        return dict(cached[1]) if cached else {}
    key = f"{uid}:{game}"
    cached = _lessons_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return dict(cached[1])
    try:
        with _connect(uid, game) as con:
            row = con.execute(
                "SELECT updated, match_count, text_json, biases_json"
                " FROM lessons WHERE id = 1").fetchone()
    except Exception:
        return dict(cached[1]) if cached else {}
    data: dict = {}
    if row:
        try:
            texts = json.loads(row[2] or "[]")
        except ValueError:
            texts = []
        try:
            biases = json.loads(row[3] or "{}")
        except ValueError:
            biases = {}
        data = {"updated": row[0], "match_count": row[1],
                "text": [t for t in texts if isinstance(t, str)],
                "biases": biases if isinstance(biases, dict) else {}}
    _lessons_cache[key] = (mtime, data)
    return dict(data)


def save_lessons(uid: str, game: str, data: dict,
                 bias_spec: dict[str, float]) -> None:
    try:
        texts = [t for t in (data.get("text") or []) if isinstance(t, str)][: _MAX_TEXT_LESSONS]
        biases = sanitize_biases(data.get("biases"), bias_spec)
        with _connect(uid, game) as con:
            con.execute(
                """INSERT INTO lessons
                   (id, updated, match_count, text_json, biases_json)
                   VALUES (1, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET updated=excluded.updated,
                     match_count=excluded.match_count, text_json=excluded.text_json,
                     biases_json=excluded.biases_json""",
                (int(data.get("updated", time.time())),
                 int(data.get("match_count", total_matches(uid, game))),
                 json.dumps(texts, ensure_ascii=False),
                 json.dumps(biases, ensure_ascii=False)))
            con.commit()
        _lessons_cache.pop(f"{uid}:{game}", None)
    except Exception:
        log.debug("learn: save lessons failed for %s", game, exc_info=True)


def lesson_texts(uid: str, game: str) -> list[str]:
    data = load_lessons(uid, game)
    texts = data.get("text") or []
    return [t for t in texts if isinstance(t, str) and t.strip()][: _MAX_TEXT_LESSONS]


def get_biases(uid: Optional[str], game: str,
               bias_spec: dict[str, float]) -> dict[str, float]:
    """Numeric AI biases with safe clamps (missing row -> neutral zeros)."""
    data = load_lessons(uid or "", game).get("biases") if uid else {}
    return sanitize_biases(data, bias_spec)


def sanitize_biases(raw: Any, bias_spec: dict[str, float]) -> dict[str, float]:
    """Clamp proposed biases into the safe band."""
    out = {k: 0.0 for k in bias_spec}
    if not isinstance(raw, dict):
        return out
    for key, clamp in bias_spec.items():
        try:
            out[key] = max(-clamp, min(clamp, float(raw.get(key, out[key]))))
        except (TypeError, ValueError):
            pass
    return out


def _extract_json(text: str) -> Optional[dict]:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[start:end])
        return data if isinstance(data, dict) else None
    except (ValueError, IndexError):
        return None


def reflect_now(uid: str, game: str, think,
                prompt_fn: Callable[[list[dict], dict], str],
                bias_spec: dict[str, float]) -> bool:
    """Run one reflection with an LLM think client. Returns True if saved."""
    recent = load_recent(uid, game, 20)
    if not recent:
        return False
    st = stats(uid, game, recent)
    try:
        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {"role": "system",
                 "content": "You coach game AI. Reply with exactly one JSON object, no other text."},
                {"role": "user", "content": prompt_fn(recent, st)},
            ],
            max_tokens=300,
            timeout=60.0,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception:
        log.debug("learn: LLM call failed for %s", game, exc_info=True)
        return False
    data = _extract_json(text)
    if not data:
        log.debug("learn: no JSON in reply for %s", game)
        return False
    lessons = [t.strip()[:160] for t in (data.get("lessons") or [])
               if isinstance(t, str) and t.strip()][: _MAX_TEXT_LESSONS]
    if not lessons:
        return False
    prev = load_lessons(uid, game)
    save_lessons(uid, game, {
        "updated": int(time.time()),
        "match_count": total_matches(uid, game),
        "text": lessons or prev.get("text", []),
        "biases": sanitize_biases(data, bias_spec),
    }, bias_spec)
    log.info("learn: saved %d lessons for %s/%s over %d games",
             len(lessons), uid, game, total_matches(uid, game))
    return True


def maybe_reflect(uid: str, game: str, think,
                  prompt_fn: Callable[[list[dict], dict], str],
                  bias_spec: dict[str, float]) -> None:
    """Fire-and-forget reflection every K finished games (guarded)."""
    if not learning_enabled(game) or think is None or not uid:
        return
    key = f"{uid}:{game}"
    with _lock:
        if key in _reflecting:
            return
        if total_matches(uid, game) % max(1, reflect_every(game)) != 0:
            return
        _reflecting.add(key)

    def _run() -> None:
        try:
            reflect_now(uid, game, think, prompt_fn, bias_spec)
        finally:
            with _lock:
                _reflecting.discard(key)

    thread = threading.Thread(target=_run, name=f"{game}-reflect", daemon=True)
    thread.start()


def get_think():
    """LLM think client when the server has one, else None (never raises)."""
    try:
        from interface.webui import auth

        inst = auth.aiko_web_instance
        return inst._think if inst and inst._think else None
    except Exception:
        return None
