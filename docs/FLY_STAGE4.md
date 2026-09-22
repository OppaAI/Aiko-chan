# Fly Stage 4 — Temporal learning

Builds on #171–#173. Does **not** add 166k neurons, spikes, or MaleVNC.

| Piece | Module | Default |
|-------|--------|---------|
| Eligibility trail | `flymemory/eligibility.py` (existing + DA path) | `FLY_ELIGIBILITY=1` |
| Dopamine PAM/PPL1 | `flymemory/dopamine.py` | always used when pulsing |
| Delayed credit | `assign_credit` → dopamine-weighted reinforce | decay `0.7` |
| Reversal boost | `FLY_REVERSAL_MULT=1.6` when bias opposes reward | on |
| Continuous trail | `record_step` every turn in `turn.py` | on when MB live |
| Sleep consolidation | `flymemory/consolidate_mb.py` | `FLY_MB_CONSOLIDATE=1` |

## Observe

1. Causal trail: `dopamine`, `eligibility` / credit, `mb_consolidate`
2. Praise after a multi-step task → prior trail steps get decayed credit
3. Teach like X then avoid X → stronger plastic flip (`reversal: true`)
4. High `sleep_pressure` + maintenance → weak plastic mass shrinks

## Config

```yaml
FLY_ELIGIBILITY: "1"
FLY_ELIGIBILITY_STEPS: "5"
FLY_ELIGIBILITY_DECAY: "0.7"
FLY_REVERSAL_MULT: "1.6"
FLY_MB_CONSOLIDATE: "1"
FLY_MB_CONSOLIDATE_MIN_SLEEP: "0.55"
MEMORY_FLYMB_MODE: "live"      # required for learning and consolidation
MEMORY_FLYSLEEP_MODE: "live"   # required for maintenance-triggered consolidation
```

## Still later

CX semantic population code (Stage 5), DN→body (Stage 6), LIF / full CNS (7–8).
