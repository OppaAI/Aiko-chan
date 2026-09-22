# Fly Stage 2 — teach → memory → agent loop

Builds on Stage 1 (#171). Goal: when you teach Aiko to prefer or avoid
something, **retrieval scores and agent abort behavior change**, and Studio
can show a short causal trail.

## What Stage 2 adds

| Piece | Behavior |
|-------|----------|
| **Teach API** | `teach_preference(topic, direction=prefer|avoid)` — explicit do/don’t |
| **Online cues** | “prefer X”, “avoid X”, “stop talking about X” → teach API + MB |
| **Recall rank** | `cognition.memory.fly_rank.adjust_recall_score` — MB bias adjusts scores; avoidance weighted harder |
| **Agent abort** | Main ReAct loop polls `should_abort_plan` each step (not only Needle) |
| **Studio causal** | `/studio/fly/api/causal` returns recent influence events as a trail |
| **Config** | Slightly higher LTM rank weight; `MEMORY_FLYMB_AVOID_MULT` |

## Teach do / don’t (operator)

```python
from cognition.flymemory.teach_api import teach_preference

teach_preference("fruit tarts", direction="avoid", user_id=uid)
teach_preference("short direct answers", direction="prefer", user_id=uid)
```

Chat cues (when `MEMORY_FLYMB_MODE=live`):

- `stop talking about X` / `avoid X` → avoid
- `prefer X` / `I like when you X` → prefer

Requires `MEMORY_FLYMB_MODE=live`. SOUL bootstrap remains separate (`FLY_SOUL_TEACH_ON_BOOT`).

## Causal trail (Studio)

`GET /studio/fly/api/causal` → last influence events, including:

- `teach_preference` / `online_teach`
- `recall_rank` (bias, delta, memory_id)
- `gf` / `mb_valence` / `dn`

## Config

```yaml
MEMORY_FLYMB_MODE: live
MEMORY_FLYMB_LTM_W: "0.04"       # was 0.01 — more visible rank effect
MEMORY_FLYMB_AVOID_MULT: "1.8"   # avoidance hurts more than approach helps
MEMORY_FLYGF_MODE: live
```

## Still not in Stage 2

- Full RL / policy network / delayed credit assignment
- 166k-neuron MaleCNS / spiking / MaleVNC
- Automatic SOUL.md full-document learning

## Rollback

Set `MEMORY_FLYMB_MODE=off` (stops teach + rank). Set `MEMORY_FLYGF_MODE=off` (stops aborts).
