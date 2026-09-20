"""Onmyoji game state — hard state lives here, in code, never in the LLM.

The narrator/Aiko prompts may *read* these structures (via gatekeeper.py)
but only this module mutates them. Dates are ISO strings (game calendar);
the spoiler rule is simple: anything dated after the player's current date
is invisible to the narrator.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SceneCard:
    """Where/when the player is and what is visible right now."""
    location_id: str = "sakai"
    date: str = "1576-01-15"  # game calendar, ISO
    present_npc_ids: list[str] = field(default_factory=list)
    present_spirit_ids: list[str] = field(default_factory=list)
    visible_detail: str = ""  # short, code-authored scene note (not LLM prose)


@dataclass
class PlayerState:
    """Who the player is and what they have done / carry."""
    name: str = "Onmyoji"
    bond: float = 0.5  # Aiko pact bond 0..1
    morality: float = 0.0  # -1 (dark) .. +1 (virtuous); evil acts work, doors close
    reputation: dict[str, float] = field(default_factory=dict)  # faction -> standing
    inventory: list[str] = field(default_factory=list)
    completed_quests: list[str] = field(default_factory=list)
    met_npc_ids: list[str] = field(default_factory=list)
    met_spirit_ids: list[str] = field(default_factory=list)
    hp: int = 10
    max_hp: int = 10
    mp: int = 10
    max_mp: int = 10
    skills: dict[str, int] = field(default_factory=dict)  # name -> level


@dataclass
class QuestState:
    quest_id: str
    stage: str = "available"  # available|active|done|failed
    log: list[str] = field(default_factory=list)


@dataclass
class Entity:
    """A persisted NPC or spirit: generated once on first meeting, then fixed."""
    entity_id: str
    kind: str  # "human" | "spirit" | "historical"
    name: str
    role: str
    disposition: str = "neutral"
    speech_style: str = ""  # persona card: diction, honorifics, temperament
    voice_seed: str = ""  # TTS timbre seed so tones are audible
    details: list[str] = field(default_factory=list)
    first_met_date: str = ""
    first_met_place: str = ""


@dataclass
class GameState:
    scene: SceneCard = field(default_factory=SceneCard)
    player: PlayerState = field(default_factory=PlayerState)
    quests: dict[str, QuestState] = field(default_factory=dict)
    entities: dict[str, Entity] = field(default_factory=dict)


def _default_state() -> GameState:
    return GameState()


class GameStore:
    """JSON-file store for one playthrough (slice simplicity; SQLite later)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            self.state = self._load()
        else:
            self.state = _default_state()
            self.save()

    def _load(self) -> GameState:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        scene = SceneCard(**raw.get("scene", {}))
        player = PlayerState(**raw.get("player", {}))
        quests = {qid: QuestState(quest_id=qid, **{k: v for k, v in q.items() if k != "quest_id"})
                  for qid, q in raw.get("quests", {}).items()}
        entities = {eid: Entity(entity_id=eid, **{k: v for k, v in e.items() if k != "entity_id"})
                    for eid, e in raw.get("entities", {}).items()}
        return GameState(scene=scene, player=player, quests=quests, entities=entities)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scene": asdict(self.state.scene),
            "player": asdict(self.state.player),
            "quests": {qid: asdict(q) for qid, q in self.state.quests.items()},
            "entities": {eid: asdict(e) for eid, e in self.state.entities.items()},
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    # ── mutations (the ONLY way state changes) ──────────────────────────

    def move(self, location_id: str, date: str = "") -> SceneCard:
        self.state.scene.location_id = location_id
        if date:
            self.state.scene.date = date
        self.state.scene.present_npc_ids = []
        self.state.scene.present_spirit_ids = []
        self.state.scene.visible_detail = ""
        self.save()
        return self.state.scene

    def meet(self, entity: Entity) -> Entity:
        known = self.state.entities.get(entity.entity_id)
        if known is not None:
            return known
        entity.first_met_date = entity.first_met_date or self.state.scene.date
        entity.first_met_place = entity.first_met_place or self.state.scene.location_id
        self.state.entities[entity.entity_id] = entity
        if entity.kind == "spirit":
            if entity.entity_id not in self.state.player.met_spirit_ids:
                self.state.player.met_spirit_ids.append(entity.entity_id)
            if entity.entity_id not in self.state.scene.present_spirit_ids:
                self.state.scene.present_spirit_ids.append(entity.entity_id)
        else:
            if entity.entity_id not in self.state.player.met_npc_ids:
                self.state.player.met_npc_ids.append(entity.entity_id)
            if entity.entity_id not in self.state.scene.present_npc_ids:
                self.state.scene.present_npc_ids.append(entity.entity_id)
        self.save()
        return entity

    def give(self, item: str) -> list[str]:
        if item not in self.state.player.inventory:
            self.state.player.inventory.append(item)
            self.save()
        return self.state.player.inventory

    def take(self, item: str) -> list[str]:
        if item in self.state.player.inventory:
            self.state.player.inventory.remove(item)
            self.save()
        return self.state.player.inventory

    def complete_quest(self, quest_id: str, note: str = "") -> QuestState:
        quest = self.state.quests.get(quest_id) or QuestState(quest_id=quest_id)
        quest.stage = "done"
        if note:
            quest.log.append(note)
        self.state.quests[quest_id] = quest
        if quest_id not in self.state.player.completed_quests:
            self.state.player.completed_quests.append(quest_id)
        self.save()
        return quest

    def shift_morality(self, delta: float, faction: str = "", amount: float = 0.0) -> PlayerState:
        player = self.state.player
        player.morality = max(-1.0, min(1.0, player.morality + delta))
        if faction:
            player.reputation[faction] = max(-1.0, min(1.0, player.reputation.get(faction, 0.0) + amount))
        self.save()
        return player

    def shift_bond(self, delta: float) -> float:
        player = self.state.player
        player.bond = max(0.0, min(1.0, player.bond + delta))
        self.save()
        return player.bond

    def spend_mp(self, cost: int) -> bool:
        """Spend spirit power; False when exhausted (action still narrated)."""
        player = self.state.player
        if player.mp < cost:
            return False
        player.mp -= cost
        self.save()
        return True

    def harm(self, amount: int) -> int:
        player = self.state.player
        player.hp = max(0, player.hp - amount)
        self.save()
        return player.hp

    def recover(self) -> PlayerState:
        player = self.state.player
        player.hp = player.max_hp
        player.mp = player.max_mp
        self.save()
        return player
