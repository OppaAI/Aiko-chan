"""
Onmyoji: deterministic journey state (hard state stays in code).

The LLM narrates around this state but never overrides it: date,
location, inventory, standings, bond, entities, and ritual outcomes live
here, computed deterministically. Decided design inputs baked in:
Sengoku era, autonomous Aiko (consultation protocol), per-spirit dread
dial, entity fade with farewell, ritual grammar over combat.
"""

from __future__ import annotations

from datetime import date as _date
from datetime import timedelta
from pydantic import BaseModel, Field

# In-game weeks before a passing encounter fades (with a farewell beat).
FADE_AFTER_DAYS = 21

# Sengoku road graph: where you can travel directly.
ROADS: dict[str, list[str]] = {
    "Kyoto": ["Azuchi", "Osaka", "Kiyosu"],
    "Azuchi": ["Kyoto", "Kiyosu"],
    "Osaka": ["Kyoto", "Sakai"],
    "Sakai": ["Osaka", "Kyoto"],
    "Kiyosu": ["Kyoto", "Azuchi", "Odawara"],
    "Odawara": ["Kiyosu"],
}

# What a careful search can turn up, per location.
SEARCH_LOOT: dict[str, list[str]] = {
    "Kyoto": ["coin", "salt", "ofuda"],
    "Azuchi": ["coin", "sake"],
    "Osaka": ["coin", "coin", "salt"],
    "Sakai": ["coin", "sake", "sake"],
    "Kiyosu": ["salt", "cord"],
    "Odawara": ["salt", "herbs"],
}


class Entity(BaseModel):
    """A generated human or spirit. Persisted on first contact."""

    id: str = Field(description="Stable id, e.g. 'sakai-teahouse-keeper-1'")
    kind: str = Field(description="human | spirit")
    name: str = ""
    role: str = ""  # e.g. "tea merchant", "battlefield ghost"
    disposition: str = ""  # e.g. "wary but curious"
    details: list[str] = Field(default_factory=list)
    dread: int = Field(default=0, description="0 whimsical – 3 folk-horror (spirits)")
    tier: str = Field(default="passing", description="anchor | bonded | passing")
    last_seen: str = Field(default="", description="In-game ISO date of last encounter")
    location: str = ""


# Ritual grammar: ward → bind → purify → banish. Costs are inventory
# item names consumed on cast; power gates which dread it can hold.
RITUALS: dict[str, dict] = {
    "ward": {"cost": ["salt"], "power": 1,
             "use": "Seal a place or person against lesser spirits."},
    "bind": {"cost": ["cord", "ofuda"], "power": 2,
             "use": "Hold a named spirit for questioning or bargaining."},
    "purify": {"cost": ["salt", "sake"], "power": 2,
               "use": "Cleanse a curse, place, or lingering grief."},
    "banish": {"cost": ["ofuda", "sake", "coin"], "power": 3,
               "use": "Send a bound spirit on. Fails loudly past its power."},
}


class JourneyState(BaseModel):
    """Save-game snapshot. Everything the narrator may read, nothing it may write."""

    date: str = Field(default="1582-06-01", description="In-game date, ISO (default: eve of Honno-ji)")
    location: str = Field(default="Kyoto", description="Real place name")
    inventory: list[str] = Field(default_factory=list)
    # Standing per faction/entity id, e.g. {"oda": 0, "hongwanji": -1}.
    standing: dict[str, int] = Field(default_factory=dict)
    bond: int = Field(default=0, description="Aiko pact-bond level")
    standing_orders: list[str] = Field(default_factory=list)
    journey_summary: str = Field(default="", description="Rolling narrator summary")
    entities: list[Entity] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list, description="One-shot markers, e.g. searched:Kyoto:1582-06-01")


def new_journey() -> JourneyState:
    return JourneyState(
        inventory=["salt", "salt", "ofuda", "ofuda", "sake", "cord", "coin", "coin"],
    )


def _days_between(then_iso: str, now_iso: str) -> int:
    try:
        return (_date.fromisoformat(now_iso) - _date.fromisoformat(then_iso)).days
    except ValueError:
        return 0


def advance_day(state: JourneyState, n: int = 1) -> None:
    """Move the calendar forward; clears per-day flags."""
    try:
        day = _date.fromisoformat(state.date)
    except ValueError:
        return
    state.date = (day + timedelta(days=max(0, n))).isoformat()
    state.flags = [f for f in state.flags if not f.startswith("searched:")]


def do_travel(state: JourneyState, dest: str) -> tuple[bool, str]:
    """Travel the road graph. Returns (ok, note)."""
    dest = (dest or "").strip()
    if dest not in ROADS:
        return False, f"no such place on the map: {dest or '?'} (try {', '.join(sorted(ROADS))})"
    if dest == state.location:
        return False, f"you are already in {dest}"
    if dest not in ROADS.get(state.location, []):
        return False, f"no direct road from {state.location} to {dest} — try {', '.join(ROADS.get(state.location, [])) or 'nowhere'}"
    state.location = dest
    advance_day(state, 1)
    faded = prune_entities(state)
    note = f"🧭 Travel to {dest} (1 day, now {state.date}). Aiko scouts ahead and reports the road clear."
    if faded:
        names = ", ".join(e.name or e.id for e in faded)
        verb = "goes" if len(faded) == 1 else "go"
        note += f" Farewells: {names} {verb} their separate way."
    return True, note


def do_rest(state: JourneyState) -> tuple[bool, str]:
    """Rest a day. Aiko keeps watch."""
    advance_day(state, 1)
    return True, f"😴 You rest in {state.location} ({state.date}). Aiko keeps watch through the night."


def do_search(state: JourneyState, rng=None) -> tuple[bool, str]:
    """Search the area once per place per day."""
    import random as _random
    key = f"searched:{state.location}:{state.date}"
    if key in state.flags:
        return False, "you have already searched here today — travel or rest first"
    state.flags.append(key)
    pool = SEARCH_LOOT.get(state.location, ["salt"])
    found = (rng or _random).choice(pool)
    state.inventory.append(found)
    return True, f"🔍 Searching {state.location} turns up: {found}."


def do_ritual(state: JourneyState, ritual: str, target_id: str) -> tuple[bool, str]:
    """Cast via resolve_ritual; successes that help earn bond."""
    ok, note = resolve_ritual(state, ritual, target_id)
    if ok and (ritual or "").strip().lower() in ("purify", "banish"):
        state.bond += 1
        note += f" Bond with Aiko deepens ({state.bond})."
    return ok, note


def prune_entities(state: JourneyState) -> list[Entity]:
    """Fade passing encounters older than FADE_AFTER_DAYS.

    Returns the faded entities so the narrator can give each a farewell
    beat. Anchors and bonded entities never fade.
    """
    kept, faded = [], []
    for e in state.entities:
        if e.tier in ("anchor", "bonded"):
            kept.append(e)
            continue
        if not e.last_seen or _days_between(e.last_seen, state.date) <= FADE_AFTER_DAYS:
            kept.append(e)
        else:
            faded.append(e)
    state.entities = kept
    return faded


def resolve_ritual(state: JourneyState, ritual: str, target_id: str) -> tuple[bool, str]:
    """Cast a ritual deterministically. Returns (success, narrator-ready note).

    - Unknown ritual / missing target / unpaid cost: fail quietly with reason.
    - Success needs power >= target dread; overmatched casts fail LOUDLY
      (note says so; narrator plays the backlash) and still consume cost.
    - Banishing a bonded spirit is refused (pact logic, not power logic).
    """
    spec = RITUALS.get((ritual or "").strip().lower())
    if spec is None:
        return False, f"unknown ritual '{ritual}' (ward, bind, purify, banish)"
    target = next((e for e in state.entities if e.id == target_id), None)
    if target is None:
        return False, f"no known entity '{target_id}'"
    if ritual == "banish" and target.tier == "bonded":
        return False, f"{target.name or target_id} is pact-bound and cannot be banished"
    missing = [c for c in spec["cost"] if c not in state.inventory]
    if missing:
        return False, f"missing ritual goods: {', '.join(missing)}"
    for c in spec["cost"]:
        state.inventory.remove(c)
    if target.kind == "spirit" and target.dread > spec["power"]:
        return False, (
            f"{spec['use']} overmatched: {target.name or target_id} "
            f"(dread {target.dread}) shrugs off a power-{spec['power']} {ritual} — backlash"
        )
    return True, f"{ritual} holds on {target.name or target_id} ({spec['use']})"
