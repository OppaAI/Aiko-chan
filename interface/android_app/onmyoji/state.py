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

# Castle/merchant towns with odd jobs to be had.
WORK_TOWNS = ("Kyoto", "Azuchi", "Osaka", "Sakai")

# Candidate encounters per town. ensure_encounters() introduces the first
# unused candidate of each kind on arrival, so the phone UI always has
# someone to greet (Talk) and rites always have targets (Ritual).
# Without this the journey starts with zero entities and those buttons
# stay disabled/hidden forever.
ENCOUNTERS: dict[str, dict[str, list[dict]]] = {
    "Kyoto": {
        "human": [
            {"suffix": "teahouse-keeper", "name": "O-Kiku", "role": "teahouse keeper",
             "disposition": "wary but curious", "details": ["remembers every rumor on the street"]},
        ],
        "spirit": [
            {"suffix": "alley-wisp", "name": "Chōchin-obake", "role": "paper-lantern ghost",
             "disposition": "mischievous, harmless", "details": ["startles drunks, loves gossip"], "dread": 1},
        ],
    },
    "Azuchi": {
        "human": [
            {"suffix": "ashigaru", "name": "Jinsuke", "role": "ashigaru footman",
             "disposition": "boastful, homesick", "details": ["served at the castle works"]},
        ],
        "spirit": [
            {"suffix": "burned-banner", "name": "The Banner", "role": "battlefield ghost",
             "disposition": "restless, sorrowful", "details": ["drifts where a standard once fell"], "dread": 2},
        ],
    },
    "Osaka": {
        "human": [
            {"suffix": "dock-foreman", "name": "Gonza", "role": "dock foreman",
             "disposition": "loud, shrewd", "details": ["knows every cargo and its curse"]},
        ],
        "spirit": [
            {"suffix": "drowned-coin", "name": "Zeni-baba", "role": "drowned miser spirit",
             "disposition": "grasping, pitiable", "details": ["counts coins that are not there"], "dread": 1},
        ],
    },
    "Sakai": {
        "human": [
            {"suffix": "tea-merchant", "name": "Sōan", "role": "tea merchant",
             "disposition": "polished, watchful", "details": ["trades with Jesuits and warlords alike"]},
        ],
        "spirit": [
            {"suffix": "harbor-kappa", "name": "Kawatarō", "role": "harbor kappa",
             "disposition": "mischievous, bribable with cucumber", "details": ["steals sandals, returns them for a price"], "dread": 0},
        ],
    },
    "Kiyosu": {
        "human": [
            {"suffix": "road-monk", "name": "Enkai", "role": "itinerant monk",
             "disposition": "gentle, tired", "details": ["walks the Nakasendō with an empty bowl"]},
        ],
        "spirit": [
            {"suffix": "crossroad-jizo", "name": "The Weeping Jizō", "role": "sorrowful roadside spirit",
             "disposition": "quiet, grieving", "details": ["weeping heard at crossroads after rain"], "dread": 2},
        ],
    },
    "Odawara": {
        "human": [
            {"suffix": "salt-peddler", "name": "O-Tsuru", "role": "salt peddler",
             "disposition": "cheerful, sharp-eyed", "details": ["has walked every siege road"]},
        ],
        "spirit": [
            {"suffix": "siege-dead", "name": "The Sleepless", "role": "siege dead",
             "disposition": "cold, angry", "details": ["still mans a wall no one can see"], "dread": 3},
        ],
    },
}

# Hard cap on hangers-on so entity data cannot grow unbounded.
MAX_ENTITIES = 12

# Greetings cycle deterministically so repeats stay fresh without an LLM.
FIRST_MEETING_NOTES = {
    "human": "You introduce yourself as a traveling onmyoji. {name} the {role} studies you, then bows — {disp}.",
    "spirit": "The air chills. {name} regards you without blinking — {disp}. Aiko steps half before you.",
}
RETURN_NOTES = [
    "{name} nods at your return. {disp_cap}, as ever.",
    "You exchange road news with {name}.",
    "{name} seems glad — or at least less wary — to see you again.",
]

# --- Vitals, skills, morality ---
MAX_HP = 10
MAX_MP = 10
RITUAL_MP: dict[str, int] = {"ward": 1, "bind": 2, "purify": 2, "banish": 3}
SKILL_FOR_RITUAL: dict[str, str] = {
    "ward": "wards", "bind": "binding", "purify": "purification", "banish": "banishing",
}
ALL_SKILLS = ("divination", "wards", "binding", "purification", "banishing")
XP_PER_LEVEL = 3  # successful casts to raise a skill by 1 (up to its limit)
SKILL_POWER_AT = 3  # skill level granting +1 effective ritual power
LIMIT_BREAK_AT = 6  # broken skills grant a further +1 (total +2)
BASE_SKILL_CAP = 5
BROKEN_SKILL_CAP = 8

# Secret arts: not trained, AWAKENED by deeds. Passive powers.
#   foxfire  — the kami reward virtue: +1 banish power, banish costs 1 less MP
#   moongaze — Aiko teaches it at bond 5: searches yield +1 find, sense local dread
#   ironwill — the body learns from collapse: +4 max HP and MP, once
SECRET_ARTS = ("foxfire", "moongaze", "ironwill")


def morality_rank(moral: int) -> str:
    if moral <= -30:
        return "Feared"
    if moral <= -10:
        return "Shady"
    if moral <= 9:
        return "Unknown"
    if moral <= 29:
        return "Trusted"
    return "Virtuous"


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
    dialogue: list[dict] = Field(default_factory=list, description="Recent exchanges {who, target, text}, newest last")
    flags: list[str] = Field(default_factory=list, description="One-shot markers, e.g. searched:Kyoto:1582-06-01")
    hp: int = Field(default=MAX_HP, description="Vitality; collapse at 0")
    max_hp: int = Field(default=MAX_HP)
    mp: int = Field(default=MAX_MP, description="Spirit power; rituals spend it")
    max_mp: int = Field(default=MAX_MP)
    skills: dict[str, int] = Field(default_factory=dict, description="Onmyoji arts 0-8")
    skills_xp: dict[str, int] = Field(default_factory=dict)
    secret_skills: list[str] = Field(default_factory=list, description="Awakened arts: foxfire, moongaze, ironwill")
    limits: dict[str, int] = Field(default_factory=dict, description="Per-skill caps (5, or 8 once broken)")
    morality: int = Field(default=0, description="-100..+100; good deeds raise, cruelty lowers")


def new_journey() -> JourneyState:
    state = JourneyState(
        inventory=["salt", "salt", "ofuda", "ofuda", "sake", "cord", "coin", "coin"],
        skills={s: 0 for s in ALL_SKILLS},
        skills_xp={s: 0 for s in ALL_SKILLS},
        secret_skills=[],
        limits={s: BASE_SKILL_CAP for s in ALL_SKILLS},
    )
    ensure_encounters(state)
    return state


def ensure_encounters(state: JourneyState) -> list[Entity]:
    """Introduce someone to meet at the current location.

    Guarantees at least one human and one spirit is present wherever the
    player arrives (first unused candidate per kind), so Talk/Ritual always
    have targets. Returns the newly met entities for the narrator's
    arrival beat. Capped by MAX_ENTITIES (oldest passing fade first).
    """
    pool = ENCOUNTERS.get(state.location, {})
    added: list[Entity] = []
    known_ids = {e.id for e in state.entities}
    loc_slug = state.location.strip().lower().replace(" ", "-") or "road"
    for kind in ("human", "spirit"):
        if any(e.kind == kind and e.location in ("", state.location)
               for e in state.entities):
            continue
        cands = [c for c in pool.get(kind, [])
                 if f"{loc_slug}-{c['suffix']}" not in known_ids]
        if not cands:
            continue
        c = cands[0]
        ent = Entity(
            id=f"{loc_slug}-{c['suffix']}",
            kind=kind,
            name=c.get("name", ""),
            role=c.get("role", ""),
            disposition=c.get("disposition", ""),
            details=list(c.get("details", [])),
            dread=int(c.get("dread", 0)) if kind == "spirit" else 0,
            tier="passing",
            last_seen=state.date,
            location=state.location,
        )
        state.entities.append(ent)
        known_ids.add(ent.id)
        added.append(ent)
    while len(state.entities) > MAX_ENTITIES:
        for i, e in enumerate(state.entities):
            if e.tier == "passing":
                del state.entities[i]
                break
        else:
            break
    return added


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
    arrived = ensure_encounters(state)
    note = f"🧭 Travel to {dest} (1 day, now {state.date}). Aiko scouts ahead and reports the road clear."
    if arrived:
        names = ", ".join(
            f"{e.name or e.id} ({e.role or e.kind})" for e in arrived)
        note += f" You meet: {names}."
    if faded:
        names = ", ".join(e.name or e.id for e in faded)
        verb = "goes" if len(faded) == 1 else "go"
        note += f" Farewells: {names} {verb} their separate way."
    return True, note


def do_rest(state: JourneyState) -> tuple[bool, str]:
    """Rest a day. Aiko keeps watch. Blessed rest for the virtuous."""
    advance_day(state, 1)
    hp_gain = 1 if state.morality <= -10 else 2
    mp_gain = 4 + (1 if state.morality >= 10 else 0)
    state.hp = min(state.max_hp, state.hp + hp_gain)
    state.mp = min(state.max_mp, state.mp + mp_gain)
    note = f"😴 You rest in {state.location} ({state.date}). Aiko keeps watch through the night."
    if state.morality >= 10:
        note += " Blessed rest."
    elif state.morality <= -10:
        note += " Restless dreams."
    return True, note


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
    note = f"🔍 Searching {state.location} turns up: {found}."
    if "moongaze" in (state.secret_skills or []):
        found2 = (rng or _random).choice(pool)
        state.inventory.append(found2)
        note += f" Moongaze reveals more: {found2}."
        sensed = [e for e in state.entities if e.kind == "spirit" and e.location in ("", state.location)]
        if sensed:
            note += " You sense: " + ", ".join(
                f"{e.name or e.id} (dread {e.dread})" for e in sensed) + "."
    return True, note


def do_ritual(state: JourneyState, ritual: str, target_id: str, rng=None) -> tuple[bool, str]:
    """Cast with MP cost, skill bonus/XP, morality shifts, collapse risk."""
    import random as _random
    name = (ritual or "").strip().lower()
    secrets = state.secret_skills or []
    mp_cost = RITUAL_MP.get(name, 0)
    if name == "banish" and "foxfire" in secrets:
        mp_cost = max(1, mp_cost - 1)
    if name in RITUAL_MP and state.mp < mp_cost:
        return False, (
            f"not enough spirit for {name} (need {mp_cost}, have {state.mp}) — rest first"
        )
    skill = SKILL_FOR_RITUAL.get(name, "")
    bonus = skill_bonus(state, skill) if skill else 0
    if name == "banish" and "foxfire" in secrets:
        bonus += 1
    if name in RITUAL_MP:
        state.mp = max(0, state.mp - mp_cost)
    ok, note = resolve_ritual(state, ritual, target_id, bonus)
    rank_before = morality_rank(state.morality)
    target = next((e for e in state.entities if e.id == target_id), None)
    target_dread = target.dread if target is not None and target.kind == "spirit" else 0
    if ok:
        if skill and gain_xp(state, skill):
            note += f" Your {skill} art rises to {skill_level(state, skill)}!"
        if target_dread >= 3 and skill and break_limit(state, skill):
            note += f" Triumph over terror — your {skill} LIMIT BREAKS to 8! ★"
        if name == "purify":
            shift_morality(state, 3)
            note += " Easing suffering (+morality)."
            state.bond += 1
            note += f" Bond with Aiko deepens ({state.bond})."
        elif name == "bind":
            shift_morality(state, 1)
            note += " Restraint over destruction (+morality)."
        elif name == "banish":
            if target is not None and target.kind == "spirit" and target.dread >= 2:
                shift_morality(state, 1)
                note += " Protecting people (+morality)."
            else:
                shift_morality(state, -5)
                note += " Cruelty stains you (−morality)."
            state.bond += 1
            note += f" Bond with Aiko deepens ({state.bond})."
        if morality_rank(state.morality) != rank_before:
            note += f" You are now {morality_rank(state.morality)}."
    elif "backlash" in note:
        state.hp = max(0, state.hp - 2)
        note += f" The backlash wounds you (−2 vitality, {state.hp} left)."
        if state.hp <= 0:
            advance_day(state, 3)
            state.hp = 3
            dropped = None
            if state.inventory:
                dropped = (rng or _random).choice(state.inventory)
                state.inventory.remove(dropped)
            if learn_secret(state, "ironwill"):
                note += " Your body learns IRONWILL from the ordeal (+4 vitality & spirit, forever)."
            note += (
                f" You collapse! Aiko drags you to safety. Three days pass ({state.date})."
                + (f" Lost: {dropped}." if dropped else "")
            )
    for awakening in check_awakenings(state, skill):
        note += f" {awakening}"
    return ok, note


def do_talk(state: JourneyState, target_id: str) -> tuple[bool, str]:
    """Greet someone (or something) you know. First meetings build standing."""
    target = next((e for e in state.entities if e.id == target_id), None)
    if target is None:
        return False, f"no known entity '{target_id}'"
    target.last_seen = state.date  # talking keeps the bond fresh (no silent fade)
    target.location = target.location or state.location
    name = target.name or target_id
    role = target.role or ("spirit" if target.kind == "spirit" else "traveler")
    disp = target.disposition or "hard to read"
    met_key = f"met:{target_id}"
    if met_key not in (state.flags or []):
        state.flags = list(state.flags or []) + [met_key]
        st = state.standing if isinstance(state.standing, dict) else {}
        st[target_id] = int(st.get(target_id, 0)) + 1
        state.standing = st
        template = FIRST_MEETING_NOTES["spirit" if target.kind == "spirit" else "human"]
        return True, "💬 " + template.format(name=name, role=role, disp=disp) + " They will remember you (+standing)."
    n = sum(1 for f in (state.flags or []) if f == met_key)
    state.flags = list(state.flags or []) + [met_key]
    line = RETURN_NOTES[n % len(RETURN_NOTES)].format(
        name=name, disp_cap=disp[:1].upper() + disp[1:] if disp else "Wary")
    return True, f"💬 {line}"


def do_train(state: JourneyState, skill: str) -> tuple[bool, str]:
    """Drill one art for a day. Slow, honest XP outside of live rituals."""
    skill = (skill or "").strip().lower()
    if skill not in ALL_SKILLS:
        return False, f"unknown art '{skill}' ({', '.join(ALL_SKILLS)})"
    advance_day(state, 1)
    leveled = gain_xp(state, skill)
    note = f"🎴 You drill {skill} through the day ({state.date})."
    if leveled:
        note += f" Your {skill} art rises to {skill_level(state, skill)}!"
    return True, note


def do_work(state: JourneyState) -> tuple[bool, str]:
    """A day's odd jobs for coin. Only where merchants gather."""
    if state.location not in WORK_TOWNS:
        return False, f"no odd jobs to be had in {state.location} — try a merchant town"
    advance_day(state, 1)
    state.inventory.append("coin")
    state.inventory.append("coin")
    return True, f"💰 A day's labor in {state.location} earns 2 coin ({state.date})."


AIKO_GREETING = (
    "Master… the pact is sealed, and I am yours. 🐱 "
    "Kyoto sleeps, but the roads whisper. What are your orders?"
)


def push_dialogue(state: JourneyState, who: str, target: str, text: str, cap: int = 12) -> None:
    """Append one exchange line, keeping only recent context."""
    state.dialogue.append({"who": who, "target": target, "text": text[:500]})
    del state.dialogue[:-cap]


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


def resolve_ritual(state: JourneyState, ritual: str, target_id: str, bonus: int = 0) -> tuple[bool, str]:
    """Cast a ritual deterministically. Returns (success, narrator-ready note).

    - Unknown ritual / missing target / unpaid cost: fail quietly with reason.
    - Success needs power + skill bonus >= target dread; overmatched casts
      fail LOUDLY (note says so; narrator plays the backlash) and still
      consume cost.
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
    power = spec["power"] + max(0, bonus)
    if target.kind == "spirit" and target.dread > power:
        return False, (
            f"{spec['use']} overmatched: {target.name or target_id} "
            f"(dread {target.dread}) shrugs off a power-{power} {ritual} — backlash"
        )
    return True, f"{ritual} holds on {target.name or target_id} ({spec['use']})"


def skill_level(state: JourneyState, skill: str) -> int:
    cap = skill_cap(state, skill)
    return max(0, min(cap, int(state.skills.get(skill, 0))))


def skill_cap(state: JourneyState, skill: str) -> int:
    try:
        return max(BASE_SKILL_CAP, min(BROKEN_SKILL_CAP, int(state.limits.get(skill, BASE_SKILL_CAP))))
    except (TypeError, ValueError):
        return BASE_SKILL_CAP


def skill_bonus(state: JourneyState, skill: str) -> int:
    """Effective ritual power bonus: +1 at 3+, another +1 at 6+ (broken)."""
    level = skill_level(state, skill)
    return (1 if level >= SKILL_POWER_AT else 0) + (1 if level >= LIMIT_BREAK_AT else 0)


def gain_xp(state: JourneyState, skill: str) -> bool:
    """Record one successful use. Returns True on level-up."""
    if skill not in ALL_SKILLS:
        return False
    cap = skill_cap(state, skill)
    xp = int(state.skills_xp.get(skill, 0)) + 1
    level = max(0, min(cap, int(state.skills.get(skill, 0))))
    ups, xp = divmod(xp, XP_PER_LEVEL)
    level = min(cap, level + ups)
    state.skills_xp[skill] = xp if level < cap else 0
    leveled = level > int(state.skills.get(skill, 0))
    state.skills[skill] = level
    return leveled


def learn_secret(state: JourneyState, art: str) -> bool:
    """Awaken a secret art. Returns True if newly learned."""
    if art not in SECRET_ARTS or art in (state.secret_skills or []):
        return False
    if art == "ironwill":
        state.max_hp += 4
        state.max_mp += 4
        state.hp = min(state.max_hp, state.hp + 4)
        state.mp = min(state.max_mp, state.mp + 4)
    state.secret_skills = list(state.secret_skills or []) + [art]
    return True


def break_limit(state: JourneyState, skill: str) -> bool:
    """Raise a skill's cap 5 → 8 after a dread-3 triumph. Returns True if new."""
    if skill not in ALL_SKILLS or skill_cap(state, skill) >= BROKEN_SKILL_CAP:
        return False
    state.limits = dict(state.limits or {})
    state.limits[skill] = BROKEN_SKILL_CAP
    return True


def check_awakenings(state: JourneyState, last_skill: str = "") -> list[str]:
    """Triumphs that teach. Call after ritual resolution. Returns notes."""
    notes: list[str] = []
    if "foxfire" not in (state.secret_skills or []) and state.morality >= 30:
        if learn_secret(state, "foxfire"):
            notes.append("🦊 The kami reward your virtue: secret art FOX FIRE learned! Banishing costs less and strikes harder.")
    if "moongaze" not in (state.secret_skills or []) and state.bond >= 5:
        if learn_secret(state, "moongaze"):
            notes.append("🌙 Aiko teaches you MOONGAZE under the night sky: searches yield more, and you sense dread nearby.")
    return notes


def shift_morality(state: JourneyState, delta: int) -> str:
    """Move morality, clamped; returns the new rank (for notes)."""
    state.morality = max(-100, min(100, int(state.morality) + delta))
    return morality_rank(state.morality)
