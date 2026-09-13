"""
Koi-Koi experience store: thin wrapper over the shared core.

Storage: <USER_SPACE_ROOT>/<uid>/agentic/koikoi.db (see learn.py and
lingo_store.py for the convention). Game-specific bits here are the
bias keys and the reflection prompt; everything else is shared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .. import learn as _learn

GAME = "koikoi"
BIAS_SPEC = {"koi_courage": 0.30, "blunder_delta": 0.10}


def learning_enabled() -> bool:
    return _learn.learning_enabled(GAME)


def reflect_every() -> int:
    return _learn.reflect_every(GAME)


def data_dir() -> Path:
    return _learn._state_root()


def user_koikoi_db_path(uid: str) -> Path:
    return _learn.user_game_db_path(uid, GAME)


def append_match(uid: str, record: dict[str, Any]) -> None:
    totals = record.get("totals") or {}
    _learn.append_match(uid, GAME, {
        "difficulty": record.get("difficulty"),
        "span": record.get("months"),
        "winner": record.get("winner"),
        "you_pts": (totals.get("you", 0)),
        "aiko_pts": (totals.get("aiko", 0)),
        "moves_made": record.get("moves_made", 0),
        "extra": {"koi_calls": record.get("koi_calls", 0),
                  "leads": record.get("leads", 0),
                  "vetoes": record.get("vetoes", 0)},
    })


def total_matches(uid: str) -> int:
    return _learn.total_matches(uid, GAME)


def load_recent(uid: str, limit: int = 20) -> list[dict]:
    out = []
    for r in _learn.load_recent(uid, GAME, limit):
        extra = r.get("extra") or {}
        out.append({
            "ts": r["ts"], "difficulty": r["difficulty"], "months": r["span"],
            "winner": r["winner"],
            "totals": {"you": r["you_pts"], "aiko": r["aiko_pts"]},
            "koi_calls": extra.get("koi_calls", 0),
            "moves_made": r["moves_made"],
            "leads": extra.get("leads", 0),
            "vetoes": extra.get("vetoes", 0),
        })
    return out


def stats(uid: str, recent: Optional[list[dict]] = None) -> dict[str, Any]:
    if recent is None:
        return _learn.stats(uid, GAME)
    rows = [{
        "winner": r.get("winner"), "difficulty": r.get("difficulty"),
    } for r in recent]
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
    leads = vetoes = 0
    for r in recent[-12:]:
        lines.append(
            f"- {r.get('difficulty', '?')} {r.get('months', '?')}mo: "
            f"winner={r.get('winner')} totals={r.get('totals')} "
            f"koi_calls={r.get('koi_calls', 0)} moves={r.get('moves_made', 0)}"
        )
        leads += r.get("leads", 0) or 0
        vetoes += r.get("vetoes", 0) or 0
    history = "\n".join(lines) if lines else "(no matches yet)"
    aiko_rate = (st["aiko_wins"] / st["matches"] * 100) if st["matches"] else 0.0
    lead_note = ""
    if leads:
        lead_note = (f"\nAiko-leads mode: she chose {leads} plays herself, "
                     f"engine vetoed {vetoes}.\n")
    return (
        "You coach Aiko, a cat-girl AI that plays Koi-Koi (hanafuda) via a "
        "heuristic engine. She learns from experience WITHOUT retraining: "
        "your output tunes her courage/blunder knobs and her table talk.\n\n"
        f"Record: {st['matches']} matches, Aiko won {aiko_rate:.0f}% "
        f"({st['aiko_wins']}-{st['you_wins']}-{st['draws']}).\n"
        f"Recent matches:\n{history}\n{lead_note}\n"
        "Reply with EXACTLY one JSON object, no other text:\n"
        '{"lessons": ["<3 short lessons, each under 18 words, about how Aiko '
        'should play or teach differently>"], '
        '"koi_courage": <number -0.3..0.3; positive if Aiko should call koi-koi '
        "more boldly, negative if she throws away leads>, "
        '"blunder_delta": <number -0.1..0.1; positive to play weaker '
        "(more casual mistakes), negative to play sharper>}\n"
        "Guidance: if Aiko wins >65%, suggest positive blunder_delta "
        "(keep games fun, not crushing). If she loses >65%, suggest negative "
        "blunder_delta and bolder koi-koi. Otherwise keep biases near zero "
        "and focus lessons on teaching moments."
    )


def _extract_json(text: str) -> Optional[dict]:
    return _learn._extract_json(text)


def reflect_now(uid: str, think) -> bool:
    return _learn.reflect_now(uid, GAME, think, build_reflection_prompt, BIAS_SPEC)


def maybe_reflect(uid: str, think) -> None:
    _learn.maybe_reflect(uid, GAME, think, build_reflection_prompt, BIAS_SPEC)
