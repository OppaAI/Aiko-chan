# Aiko biological layer II — central-complex compass

Aiko's focus compass: a ring-attractor focus/heading plus R5 sleep-drive
pressure, grafted from the real **MaleCNS v1.0** central complex
(Berg et al. 2025, CC-BY, https://male-cns.janelia.org). In the fly, EPG
neurons hold a bump of activity = head direction, PENs rotate it with
angular velocity, PFL outputs steer action, and EB R5 neurons track sleep
drive. Here: bump heading/sharpness = focus direction/confidence, PFL
L/R imbalance = decisiveness, ER5 accumulator = drowsiness pressure.

## What's real vs synthetic

| Part | Status | Source |
|---|---|---|
| Neurons, edges, weights (EPG/PEN/PEG/PF/RING/INPUT) | REAL | `data/cx_circuit.npz`, MaleCNS v1.0 `minconf-0.5`, weight ≥ 5 |
| EPG wedge order (16 wedges from `_R1..8/_L1..8` suffixes) | REAL | parsed from instance names |
| PFL L/R steering split (25/25) | REAL | somaSide column |
| ER5 units (21) as sleep-drive proxy | REAL neurons, proxy role | R5 sleep literature (Liu et al. 2016); v1.0 has no dFB-annotated sleep types |
| Feature→INPUT/RING projections, PEN drive, fatigue scalars | SYNTHETIC | fixed seeded matrices / caller inputs |
| One-step recurrent update, bump rotation gain, sleep rate/decay | functional abstraction | real weights, synthetic dynamics |

Slice stats: 1,282 neurons (50 EPG · 42 PEN · 18 PEG · 506 PF · 308 RING ·
358 top EPG inputs), 50,515 synapses. See `data/summary.json`.

## GitHub artefacts used

Same set as the mushroom-body layer (`cognition/flymemory/README.md`):
`flyconnectome/2025malecns`, `connectome-neuprint/neuprint-python`,
`natverse/malecns` (R reference), `janelia-flyem/male-cns`. No token, no
runtime network; 172KB ships here.

## Regenerating

`tools/extract_cx.py` is the exact one-off script (needs pandas + pyarrow).

## Wiring

`FlyCompass.step(features, pen_drive, fatigue)` per turn; readouts consumed
by `EdgeCognitiveState.record()` behind `MEMORY_FLYCX_MODE` (off | shadow |
live, default off; see `config/memory.yaml`). Shadow is log-only; live adds
a small drowsiness coupling to `_energy`.
