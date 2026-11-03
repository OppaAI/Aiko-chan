"""
Hanafuda cards + Koi-Koi round rules (pure stdlib — no engine binary needed).

Card ids 0..47: month = id // 4 (0=Jan pine … 11=Dec paulownia),
slot = id % 4. Composition follows standard hanafuda:

- Hikari (bright, 5): crane Jan, curtain Mar, moon Aug, rain-man Nov,
  phoenix Dec.
- Tane (animal, 9): nightingale Feb, cuckoo Apr, bridge May, butterfly Jun,
  boar Jul, geese Aug, sake-cup Sep, deer Oct, swallow Nov.
- Tanzaku (ribbon, 10): poetry Jan/Feb/Mar, plain-red Apr/May/Jul/Nov,
  blue Jun/Sep/Oct.
- Kasu (chaff, 24): everything else.

Yaku (standard scores, documented in-app on the rules page):
  Goko 10 · Shiko 8 · Ame-shiko 7 · Sanko 5 · Tsukimi/Hanami-zake 5 ·
  Ino-shika-cho 5 · Akatan/Aotan 5 · Tane 1+extra · Tanzaku 1+extra ·
  Kasu 1+extra (10 needed).
Dealt yaku (checked on the initial 8-card hands, auto-win, no koi-koi):
  Teshi (four of a month) 6 · Kuttsuki (four pairs) 6.
Month yaku: Tsuki-fuda (all four of the round's calendar month) 4.
"""

from __future__ import annotations

import random

# month index -> (kanji, flower/thing label for UI)
MONTHS: list[tuple[str, str]] = [
    ("松", "pine"), ("梅", "plum"), ("桜", "cherry"), ("藤", "wisteria"),
    ("菖蒲", "iris"), ("牡丹", "peony"), ("萩", "clover"), ("芒", "pampas"),
    ("菊", "chrysanthemum"), ("紅葉", "maple"), ("柳", "willow"), ("桐", "paulownia"),
]

# month index -> list of 4 (kind, special-name); kind in
# hikari | tane | tan-poetry | tan-red | tan-blue | kasu
DECK: list[tuple[str, str]] = [
    [("hikari", "crane"), ("tan-poetry", "poetry"), ("kasu", ""), ("kasu", "")],
    [("tane", "nightingale"), ("tan-poetry", "poetry"), ("kasu", ""), ("kasu", "")],
    [("hikari", "curtain"), ("tan-poetry", "poetry"), ("kasu", ""), ("kasu", "")],
    [("tane", "cuckoo"), ("tan-red", "ribbon"), ("kasu", ""), ("kasu", "")],
    [("tane", "bridge"), ("tan-red", "ribbon"), ("kasu", ""), ("kasu", "")],
    [("tane", "butterfly"), ("tan-blue", "ribbon"), ("kasu", ""), ("kasu", "")],
    [("tane", "boar"), ("tan-red", "ribbon"), ("kasu", ""), ("kasu", "")],
    [("hikari", "moon"), ("tane", "geese"), ("kasu", ""), ("kasu", "")],
    [("tane", "sake-cup"), ("tan-blue", "ribbon"), ("kasu", ""), ("kasu", "")],
    [("tane", "deer"), ("tan-blue", "ribbon"), ("kasu", ""), ("kasu", "")],
    [("hikari", "rain-man"), ("tane", "swallow"), ("tan-red", "ribbon"), ("kasu", "lightning")],
    [("hikari", "phoenix"), ("kasu", ""), ("kasu", ""), ("kasu", "")],
]

N_CARDS = 48


def month_of(card: int) -> int:
    return card // 4


def kind_of(card: int) -> str:
    return DECK[month_of(card)][card % 4][0]


def special_of(card: int) -> str:
    return DECK[month_of(card)][card % 4][1]


def full_deck(rng: random.Random) -> list[int]:
    deck = list(range(N_CARDS))
    rng.shuffle(deck)
    return deck


def has_four_of_month(cards: list[int]) -> bool:
    by_month: dict[int, int] = {}
    for c in cards:
        m = month_of(c)
        by_month[m] = by_month.get(m, 0) + 1
        if by_month[m] >= 4:
            return True
    return False


def deal(rng: random.Random) -> tuple[list[int], list[int], list[int], list[int]]:
    """Deal (you, aiko, field, stock), redealing 4-of-a-month layouts.

    Standard mishaps (kuttsuki/teshi) are simplified to a redeal so every
    round starts fair and playable.
    """
    for _ in range(100):
        deck = full_deck(rng)
        you, aiko, field = deck[:8], deck[8:16], deck[16:24]
        stock = deck[24:]
        if has_four_of_month(field) or has_four_of_month(you) or has_four_of_month(aiko):
            continue
        return you, aiko, field, stock
    # Fallback (practically unreachable): accept the last deal.
    return you, aiko, field, stock


def match_options(played: int, field: list[int]) -> list[list[int]]:
    """Field-card subsets the played card can take (each a list of field ids).

    - 0 same-month on field → [] (no capture; played card joins the field,
      deck flip still happens).
    - 1 → [[that card]].
    - 2 → [[a], [b]] (chooser picks).
    - 3 → [[all three]] (sweep all four).
    """
    same = [c for c in field if month_of(c) == month_of(played)]
    if not same:
        return []
    if len(same) == 1:
        return [same]
    if len(same) == 2:
        return [[same[0]], [same[1]]]
    return [list(same)]


# Special card ids (for yaku + UI).
CRANE = 0
CURTAIN = 2 * 4
MOON = 7 * 4
RAIN_MAN = 10 * 4
PHOENIX = 11 * 4
BOAR = 6 * 4
BUTTERFLY = 5 * 4
DEER = 9 * 4
SAKE_CUP = 8 * 4

TESHI_POINTS = 6
KUTTSUKI_POINTS = 6
TSUKI_POINTS = 4


def detect_hand_yaku(hand: list[int]) -> list[dict]:
    """Dealt yaku on an initial 8-card hand (teshi before kuttsuki).

    A hand can hold at most one of them (teshi uses 4 of one month,
    kuttsuki needs 4 distinct pairs).
    """
    counts: dict[int, int] = {}
    for c in hand:
        m = month_of(c)
        counts[m] = counts.get(m, 0) + 1
    if any(n >= 4 for n in counts.values()):
        return [{"id": "teshi", "name": "Four of a Month", "jp": "手四",
                 "points": TESHI_POINTS}]
    if len(hand) >= 8 and len(counts) == 4 and all(n == 2 for n in counts.values()):
        return [{"id": "kuttsuki", "name": "Four Pairs", "jp": "くっつき",
                 "points": KUTTSUKI_POINTS}]
    return []


def detect_yaku(captured: list[int], round_month: int | None = None) -> list[dict]:
    """All completed yaku in a captured collection, each {id, name, jp, points}.

    round_month is the 1-based calendar month of the current round
    (round N plays month N): holding all four of its cards completes
    Tsuki-fuda (月札, 4 pts).
    """
    have = set(captured)
    kinds = [kind_of(c) for c in captured]
    yaku: list[dict] = []

    brights = [c for c in captured if kind_of(c) == "hikari"]
    n_bright = len(brights)
    has_rain = RAIN_MAN in have
    if n_bright == 5:
        yaku.append({"id": "goko", "name": "Five Brights", "jp": "五光", "points": 10})
    elif n_bright == 4 and not has_rain:
        yaku.append({"id": "shiko", "name": "Four Brights", "jp": "四光", "points": 8})
    elif n_bright == 4 and has_rain:
        yaku.append({"id": "ame-shiko", "name": "Rainy Four Brights", "jp": "雨四光", "points": 7})
    elif n_bright == 3 and not has_rain:
        yaku.append({"id": "sanko", "name": "Three Brights", "jp": "三光", "points": 5})

    if MOON in have and SAKE_CUP in have:
        yaku.append({"id": "tsukimi", "name": "Moon Viewing", "jp": "月見酒", "points": 5})
    if CURTAIN in have and SAKE_CUP in have:
        yaku.append({"id": "hanami", "name": "Flower Viewing", "jp": "花見酒", "points": 5})
    if BOAR in have and DEER in have and BUTTERFLY in have:
        yaku.append({"id": "ino-shika-cho", "name": "Boar-Deer-Butterfly", "jp": "猪鹿蝶", "points": 5})

    poetry = [c for c in captured if kind_of(c) == "tan-poetry"]
    blue = [c for c in captured if kind_of(c) == "tan-blue"]
    if len(poetry) == 3:
        yaku.append({"id": "akatan", "name": "Poetry Ribbons", "jp": "赤短", "points": 5})
    if len(blue) == 3:
        yaku.append({"id": "aotan", "name": "Blue Ribbons", "jp": "青短", "points": 5})

    n_tane = sum(1 for k in kinds if k == "tane")
    if n_tane >= 5:
        yaku.append({
            "id": "tane", "name": f"Animals ×{n_tane}", "jp": "タネ",
            "points": 1 + (n_tane - 5),
        })
    n_tan = sum(1 for k in kinds if k.startswith("tan-"))
    if n_tan >= 5:
        yaku.append({
            "id": "tanzaku", "name": f"Ribbons ×{n_tan}", "jp": "短冊",
            "points": 1 + (n_tan - 5),
        })
    n_kasu = sum(1 for k in kinds if k == "kasu")
    if n_kasu >= 10:
        yaku.append({
            "id": "kasu", "name": f"Chaff ×{n_kasu}", "jp": "カス",
            "points": 1 + (n_kasu - 10),
        })
    if round_month is not None:
        need = {(round_month - 1) * 4 + s for s in range(4)}
        if need <= have:
            yaku.append({
                "id": "tsuki-fuda", "name": "Month Cards", "jp": "月札",
                "points": TSUKI_POINTS,
            })
    return yaku


def yaku_points(yaku: list[dict]) -> int:
    return sum(y["points"] for y in yaku)


def near_yaku_score(captured: list[int]) -> float:
    """Progress toward uncompleted yaku (0..~1 each) for AI greed."""
    have = set(captured)
    kinds = [kind_of(c) for c in captured]
    score = 0.0
    n_bright = sum(1 for k in kinds if k == "hikari")
    score += min(n_bright, 3) * 0.25
    if MOON in have or CURTAIN in have:
        score += 0.2 if SAKE_CUP in have else 0.1
    if SAKE_CUP in have and not (MOON in have or CURTAIN in have):
        score += 0.1
    trio = sum(1 for c in (BOAR, DEER, BUTTERFLY) if c in have)
    score += trio * 0.2
    poetry = sum(1 for k in kinds if k == "tan-poetry")
    blue = sum(1 for k in kinds if k == "tan-blue")
    score += min(poetry, 2) * 0.2 + min(blue, 2) * 0.2
    n_tane = sum(1 for k in kinds if k == "tane")
    n_tan = sum(1 for k in kinds if k.startswith("tan-"))
    n_kasu = sum(1 for k in kinds if k == "kasu")
    if n_tane >= 3:
        score += 0.15 * (n_tane - 2)
    if n_tan >= 3:
        score += 0.15 * (n_tan - 2)
    if n_kasu >= 7:
        score += 0.1 * (n_kasu - 6)
    return score


_CARD_VALUE = {
    "hikari": 6.0,
    "tane": 3.0,
    "tan-poetry": 3.0,
    "tan-blue": 3.0,
    "tan-red": 2.0,
    "kasu": 0.5,
}


def card_value(card: int) -> float:
    return _CARD_VALUE[kind_of(card)]
