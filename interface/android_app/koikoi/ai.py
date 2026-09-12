"""
Built-in Koi-Koi AI ("Aiko" heuristic).

There is no lightweight external Koi-Koi engine comparable to YaneuraOu
(Shogi/USI) or KataGo (Go/GTP), so Aiko plays from these heuristics
directly — always available, no binary, no warmup, Nano-friendly.
Difficulty only changes blunder rate and koi-koi courage:

  easy   — 35% random plays, timid/erratic koi-koi calls
  medium — 10% random plays, sensible stops (default)
  hard   — best heuristic play, sharp koi-koi judgement
"""

from __future__ import annotations

import random

from . import cards as C

_DIFFICULTY = ("easy", "medium", "hard")
_DEFAULT_DIFFICULTY = "medium"
_BLUNDER = {"easy": 0.35, "medium": 0.10, "hard": 0.0}


def difficulty_name(value: str | None) -> str:
    raw = (value or "").strip().lower()
    return raw if raw in _DIFFICULTY else _DEFAULT_DIFFICULTY


def available() -> bool:
    return True


def _option_value(hand: int, take: list[int], captured: list[int],
                   round_month: int | None = None) -> float:
    """Value of playing `hand` and taking `take` (field ids)."""
    before = C.yaku_points(C.detect_yaku(captured, round_month))
    after_cards = captured + [hand] + list(take)
    after = C.yaku_points(C.detect_yaku(after_cards, round_month))
    gain = (after - before) * 3.0
    loot = sum(C.card_value(c) for c in [hand] + list(take))
    progress = (C.near_yaku_score(after_cards) - C.near_yaku_score(captured)) * 1.5
    sweep_bonus = 1.0 if len(take) >= 3 else 0.0
    dry_penalty = -0.5 if not take else 0.0  # no capture feeds the field
    return gain + loot + progress + sweep_bonus + dry_penalty


def enumerate_plays(hand_cards: list[int], field: list[int]) -> list[tuple[int, list[int]]]:
    """Every legal (hand, take) pair; take=[] means no capture."""
    options: list[tuple[int, list[int]]] = []
    for h in hand_cards:
        takes = C.match_options(h, field)
        if not takes:
            options.append((h, []))
        else:
            for t in takes:
                options.append((h, list(t)))
    return options


def choose_play(
    hand_cards: list[int],
    field: list[int],
    captured: list[int],
    difficulty: str | None = None,
    rng: random.Random | None = None,
    round_month: int | None = None,
) -> tuple[int, list[int]]:
    rng = rng or random.Random()
    options = enumerate_plays(hand_cards, field)
    if not options:
        raise ValueError("no cards to play")
    if rng.random() < _BLUNDER[difficulty_name(difficulty)]:
        return rng.choice(options)
    return max(options, key=lambda o: (_option_value(o[0], o[1], captured, round_month), rng.random()))


def choose_flip(
    flip: int,
    options: list[list[int]],
    captured: list[int],
    difficulty: str | None = None,
    rng: random.Random | None = None,
    round_month: int | None = None,
) -> list[int]:
    """Pick among deck-flip takes (each option already includes the flip)."""
    rng = rng or random.Random()
    if not options:
        return []
    if len(options) == 1:
        return list(options[0])
    if rng.random() < _BLUNDER[difficulty_name(difficulty)]:
        return list(rng.choice(options))
    scored = [(_option_value(flip, o, captured, round_month), o) for o in options]
    return list(max(scored, key=lambda s: (s[0], rng.random()))[1])


def choose_decision(
    captured: list[int],
    opp_captured: list[int],
    koi_count: int,
    cards_left: int,
    difficulty: str | None = None,
    rng: random.Random | None = None,
    round_month: int | None = None,
) -> str:
    """'stop' (bank points) or 'koi' (continue, doubling the stakes)."""
    rng = rng or random.Random()
    level = difficulty_name(difficulty)
    current = C.yaku_points(C.detect_yaku(captured, round_month))
    potential = C.near_yaku_score(captured)
    threat = C.near_yaku_score(opp_captured) + len(opp_captured) / 48.0
    late = cards_left <= 10

    if rng.random() < _BLUNDER[level]:
        # Casual mistake: sometimes bank tiny scores, sometimes chase recklessly.
        return rng.choice(["stop", "koi"])

    # Bank solid scores, especially late or when the opponent threatens.
    stop_line = {"easy": 8.0, "medium": 6.0, "hard": 5.0}[level]
    if current >= stop_line:
        return "stop"
    if late and current >= 3:
        return "stop"
    if threat >= 1.4 and current >= 4:
        return "stop"
    # Chase when there is real potential and little banked.
    chase_line = {"easy": 1.2, "medium": 0.7, "hard": 0.45}[level]
    if potential >= chase_line and not late:
        return "koi"
    # Small score with nothing brewing: bank it rather than risk the double.
    return "stop"
