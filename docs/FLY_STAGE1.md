# Fly Stage 1 — closed loops + SOUL teaching

Solo-engineer PR. Observe in normal use via **Fly Circuit Studio** (`/studio/fly/`).

## What this is

Make existing MaleCNS-derived circuits the **authority** for continuous behavioral signals, teach the mushroom body from `persona/SOUL.md` + live outcomes, and surface everything in Studio.

**Not** a full 166k-neuron simulation.

## What changed

| Area | Behavior |
|------|----------|
| **GF interrupt** | Multi-source urgency (keywords + safety + system_error + priority + subliminal + motion). `should_abort_plan(user_id)` for agent loops. |
| **MB valence** | Each turn publishes MB `valence_bias` into NeuralState when `MEMORY_FLYMB_MODE=live`. |
| **SOUL teach** | On first turn (or SOUL.md change), curated situations are reinforced into KC→MBON plasticity (`FLY_SOUL_TEACH_ON_BOOT`). |
| **Online teach** | User praise / correction / “stop talking about X” → small `reinforce()`. |
| **DN vigor** | Turn priors publish DN rate/volume proxies into NeuralState. |
| **Influence log** | NeuralState keeps a ring buffer of last ~48 fly events for Studio. |
| **LH / CX / sleep** | Unchanged modes; still live. Diversity/anti-loop remains memory-side. |

## Config (config/memory.yaml)

```yaml
MEMORY_FLYMB_MODE: live
MEMORY_FLYCX_MODE: live
MEMORY_FLYLH_MODE: live
MEMORY_FLYGF_MODE: live
MEMORY_FLYSLEEP_MODE: live
MEMORY_FLYDN_MODE: live
MEMORY_FLYAL_MODE: live

FLY_SOUL_TEACH_ON_BOOT: live   # off | shadow | live
FLY_SOUL_TEACH_FORCE: 0        # 1 = re-bootstrap even if SOUL hash unchanged
```

Existing weights (`MEMORY_FLYMB_W`, `MEMORY_FLYLH_W`, …) still apply to ranking.

## How to observe without a formal review

1. Use Aiko normally.
2. Open `/studio/fly/` → NeuralState sidebar.
3. Watch `influence` entries: `mb_valence`, `gf`, `lh`, `dn`, `online_teach`.
4. Say **STOP** mid-task → urgency/interrupt should rise; route already forces localchat when interrupt is set.
5. Praise or “stop talking about fruit tarts” → online teach events appear; sticky topics should soften over time via avoidance teaching.

## Agent abort

`cognition.fly_behavior.should_abort_plan(user_id)` returns True when live GF interrupt is set. Needle aborts the crew when this is true. Route already maps interrupt → localchat.

## Rollback

Set any `MEMORY_FLY*_MODE` or `FLY_SOUL_TEACH_ON_BOOT` to `off` (or `shadow` for log-only). Plasticity DB is per-user; delete `fly_plasticity.db` under the user state dir to reset learned MB deltas.

## Honest limits

- Synthetic sensory projection into MB (8-D `text_features`).
- SOUL alignment is associative conditioning, not loading prose into neurons.
- Emotion emoji labels remain LLM-side.
- Language, LTM storage, and reasoning remain LLM/memory systems.
