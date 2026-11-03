# Fly Stage 3 — close remaining A–H gaps

Builds on #171 + #172. Does **not** add full RL, 166k neurons, or spikes.

| A–H | Stage 3 change |
|-----|----------------|
| **A** | Studio Causal trail panel (`#causal` + `/api/causal`) |
| **B** | Recall rank blends MB first, subliminal valence second |
| **C** | Already wired (#171/#172); unchanged |
| **D** | `cx_topic.apply_topic_drive` on turn priors |
| **E** | `antiloop.apply_antiloop` freshness penalty after diversify |
| **F** | `dn_tts.apply_dn_prosody` on `Speaker.speak` / `speak_synced` |
| **G** | `preference_store` durable JSON + rank delta |
| **H** | `tests/eval/fly_stage3_ab.py` + this doc |

## Observe

1. `/studio/fly/` → **Causal trail** panel
2. Teach `avoid fruit tarts` → `fly_preferences.json` + `recall_rank` events
3. Same-motif memories should drop after being shown (`MEMORY_ANTILOOP_W`)
4. TTS rate/volume follows `motor_vigor` when `MEMORY_FLYDN_MODE=live`

## Config

```yaml
MEMORY_ANTILOOP_W: "0.04"
MEMORY_ANTILOOP_HALF_LIFE_S: "900"
MEMORY_FLY_SUBLIMINAL_W: "0.2"   # secondary blend vs MB
```

## Still out of scope

Full RL, delayed credit assignment, 166k MaleCNS, spiking, MaleVNC, whole-document SOUL learning.
