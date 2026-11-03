# Fly-brain integration roadmap (post PR #165)

## Isolation contract (must stay true)

```
user A → fly state A
user B → fly state B
         ↓
    no cross-talk
```

All MB/CX/plasticity access goes through `cognition.fly_registry`
(`get_flymb` / `get_flycx` / `get_fly_store` / `flush_*`).
No module-local `FlyMB()` / `FlyCompass()` singletons.

## NeuralState bus

`cognition.neural_state.get_neural_state(user_id)` is the single read model:

valence, approach/avoidance, focus heading/sharpness, decisiveness,
sleep_pressure, sensory_gain, motion_salience, action_drive, motor_vigor,
context_familiarity, urgency, interrupt, circadian_phase.

Publishers: MB (grasp/promote/recall/memory), CX (attention), DN (prosody),
LH, GF. Consumers: memory rank, agent cadence, voice, persona tone, studio.

## Layers

| Motif | Module | Role | Status |
|-------|--------|------|--------|
| MB | flymemory + registry | Learned association | Wired (shadow default) |
| CX | centralcomplex + registry | Focus + sleep pressure | Wired (shadow) |
| AL / T4-T5 / DN | flysense | Gain, motion, prosody | Wired (mode-gated) |
| LH | fly_behavior.lateral_horn | Context familiarity prior | Scaffold (off) |
| GF | fly_behavior.giant_fiber | Interrupt prior | Scaffold (off) |
| Clock | (planned) | Circadian phase | Wall-clock proxy on NeuralState |
| MaleVNC | (later) | Motor only with embodiment | Deferred |

## Persona

NeuralState can bias persona *tone* (approach → warmer, avoid → cautious,
high sleep → quieter, high urgency → short) without changing identity text.
Keep persona constitution in `persona/`; use NeuralState as soft style prior only.

## Explicit non-goals

- Full MaleCNS spike simulation in-chat
- Courtship-specific male circuits for female Aiko
- MaleVNC before physical body
- Claiming "Aiko is a fly brain"

These are connectome-derived functional abstractions around the LLM.
