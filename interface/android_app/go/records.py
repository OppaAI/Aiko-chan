"""
Go experience store: thin wrapper over the shared core.

Storage: <USER_SPACE_ROOT>/<uid>/agentic/go.db (see learn.py).
Learned knob: blunder_delta shifts the casual-mistake rate up/down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .. import learn as _learn

GAME = "go"
BIAS_SPEC = {"blunder_delta": 0.15}


def learning_enabled() -> bool:
    return _learn.learning_enabled(GAME)


def reflect_every() -> int:
    return _learn.reflect_every(GAME)


def user_go_db_path(uid: str) -> Path:
    return _learn.user_game_db_path(uid, GAME)


def append_match(uid: str, record: dict[str, Any]) -> None:
    _learn.append_match(uid, GAME, {
        "difficulty": record.get("difficulty"),
        "span": record.get("size"),
        "winner": record.get("winner"),
        "you_pts": 0,
        "aiko_pts": 0,
        "moves_made": record.get("moves_made", 0),
        "extra": {
            "cap_you": record.get("cap_you", 0),
            "cap_aiko": record.get("cap_aiko", 0),
            "engine": record.get("engine", ""),
        },
    })


def total_matches(uid: str) -> int:
    return _learn.total_matches(uid, GAME)


def load_recent(uid: str, limit: int = 20) -> list[dict]:
    out = []
    for r in _learn.load_recent(uid, GAME, limit):
        extra = r.get("extra") or {}
        out.append({
            "ts": r["ts"], "difficulty": r["difficulty"], "size": r["span"],
            "winner": r["winner"], "moves_made": r["moves_made"],
            "cap_you": extra.get("cap_you", 0),
            "cap_aiko": extra.get("cap_aiko", 0),
            "engine": extra.get("engine", ""),
        })
    return out


def stats(uid: str, recent: Optional[list[dict]] = None) -> dict[str, Any]:
    if recent is None:
        return _learn.stats(uid, GAME)
    rows = [{"winner": r.get("winner"), "difficulty": r.get("difficulty")}
            for r in recent]
    return _learn.stats(uid, GAME, rows)


def load_lessons(uid: str) -> dict:
    return _learn.load_lessons(uid, GAME)


def save_lessons(uid: str, data: dict) -> None:
    _learn.save_lessons(uid, GAME, data, BIAS_SPEC)


def lesson_texts(uid: str) -> list[str]:
    return _learn.lesson_texts(uid, GAME)


def biases(uid: Optional[str] = None) -> dict[str, float]:
    return _learn.get_biases(uid, GAME, BIAS_SPEC)


def sanitize_biases(raw: Any) -> dict[str, float]:
    return _learn.sanitize_biases(raw, BIAS_SPEC)


def build_reflection_prompt(recent: list[dict], st: dict) -> str:
    lines = []
    for r in recent[-12:]:
        lines.append(
            f"- {r.get('difficulty', '?')} {r.get('size', '?')}x{r.get('size', '?')}: "
            f"winner={r.get('winner')} moves={r.get('moves_made', 0)} "
            f"captures you={r.get('cap_you', 0)} aiko={r.get('cap_aiko', 0)}"
        )
    history = "\n".join(lines) if lines else "(no games yet)"
    aiko_rate = (st["aiko_wins"] / st["matches"] * 100) if st["matches"] else 0.0
    return (
        "You coach Aiko, a cat-girl AI that plays Go via KataGo (or casual "
        "legal moves when KataGo is offline). She learns from experience "
        "WITHOUT retraining: your output tunes her casual-mistake knob and "
        "her table talk.\n\n"
        f"Record: {st['matches']} games, Aiko won {aiko_rate:.0f}% "
        f"({st['aiko_wins']}-{st['you_wins']}-{st['draws']}).\n"
        f"Recent games:\n{history}\n\n"
        "Reply with EXACTLY one JSON object, no other text:\n"
        '{"lessons": ["<3 short lessons, each under 18 words, about how Aiko '
        'should play or teach differently>"], '
        '"blunder_delta": <number -0.15..0.15; positive to play weaker '
        "(more casual mistakes, friendlier games), negative to play sharper>}\n"
        "Guidance: if Aiko wins >65%, suggest positive blunder_delta. "
        "If she loses >65%, suggest negative blunder_delta. Otherwise keep "
        "it near zero and focus lessons on teaching moments."
    )


def reflect_now(uid: str, think) -> bool:
    return _learn.reflect_now(uid, GAME, think, build_reflection_prompt, BIAS_SPEC)


def maybe_reflect(uid: str, think) -> None:
    _learn.maybe_reflect(uid, GAME, think, build_reflection_prompt, BIAS_SPEC)
