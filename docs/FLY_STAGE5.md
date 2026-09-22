# Fly Stage 5 — Semantic fly senses

Replaces SHA-1 CX topic hash with Harrier-driven features; upgrades LH
familiarity; feeds T4/T5 motion into CX pen drive.

| Piece | Module |
|-------|--------|
| Semantic 8-D features | `fly_behavior/semantic_features.py` |
| CX feature blend | `fly_behavior/cx_features.py` |
| CX topic (semantic) | `fly_behavior/cx_topic.py` |
| LH embedding familiarity | `fly_behavior/lateral_horn.py` |
| Motion → pen | via `NeuralState.motion_salience` in `blend_cx_features` |

## Config

```yaml
MEMORY_FLYCX_MODE: "live"
MEMORY_FLYLH_MODE: "live"
FLY_CX_SEMANTIC_W: "0.55"
FLY_CX_MOTION_W: "0.35"
```

## Observe

1. Causal trail: `cx_topic` with `mode: semantic` (not `observe`)
2. Same topic paraphrased → similar heading; LH method `embedding`
3. High `motion_salience` → stronger pen drive on next CX step

## Still later

DN→body (Stage 6), LIF / full CNS (7–8).
