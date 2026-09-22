"""Durable prefer/avoid facts (Stage 3 / A–H G).

JSON list under the user state dir. Used by teach_api and recall ranking
so a taught topic is more than a one-shot MB reinforce.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger("aiko.memory.preference_store")

_MAX = 64


def _path(user_id: str | None) -> Path | None:
    try:
        from system.userspace import user_state_dir
        root = Path(user_state_dir(user_id or ""))
        root.mkdir(parents=True, exist_ok=True)
        return root / "fly_preferences.json"
    except Exception:
        return None


def load_preferences(user_id: str | None = None) -> list[dict]:
    p = _path(user_id)
    if p is None or not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return list(data) if isinstance(data, list) else []
    except Exception as exc:
        log.debug("load_preferences failed: %s", exc)
        return []


def record_preference(topic: str, direction: str, *, user_id: str | None = None) -> dict:
    topic = (topic or "").strip()
    direction = "prefer" if str(direction).lower() in ("prefer", "approach", "like", "want") else "avoid"
    out = {"topic": topic, "direction": direction, "ts": time.time()}
    if not topic:
        return out
    rows = [r for r in load_preferences(user_id) if str(r.get("topic") or "").lower() != topic.lower()]
    rows.append(out)
    rows = rows[-_MAX:]
    p = _path(user_id)
    if p is not None:
        try:
            p.write_text(json.dumps(rows, indent=0), encoding="utf-8")
        except Exception as exc:
            log.debug("record_preference write failed: %s", exc)
    return out


def preference_delta(text: str, *, user_id: str | None = None) -> float:
    """Score nudge: + for prefer-topic overlap, − for avoid-topic overlap."""
    low = (text or "").lower()
    if not low:
        return 0.0
    delta = 0.0
    for row in load_preferences(user_id):
        topic = str(row.get("topic") or "").lower()
        if len(topic) < 2 or topic not in low:
            continue
        delta += 0.04 if row.get("direction") == "prefer" else -0.08
    return max(-0.25, min(0.15, delta))
