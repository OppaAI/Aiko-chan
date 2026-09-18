# Fly stack complete (except MaleVNC)

## Wired

| Circuit | Live path |
|---------|-----------|
| MB | grasp / recall / promote / dream / forget |
| CX | attention, needle cadence, NeuralState |
| LH | turn priors + **grasp score bias** |
| GF | turn priors + system_note + force localchat |
| Sleep | needle maintenance + **dream boost multiplier** |
| AL / DN | available; default shadow (sensory/motor) until presence/avatar eval |

## Not included

- MaleVNC (needs physical body)
- Full 166k-neuron spike simulation (use pre-extracted slices + motifs)

## A/B

```bash
MEMORY_FLYMB_MODE=off MEMORY_FLYGF_MODE=off python -m tests.eval.fly_ab_harness
MEMORY_FLYMB_MODE=live MEMORY_FLYGF_MODE=live MEMORY_FLYLH_MODE=live MEMORY_FLYSLEEP_MODE=live python -m tests.eval.fly_ab_harness
```

Then chat with `/studio/fly/` open and compare STOP / recall / long-agent turns.
