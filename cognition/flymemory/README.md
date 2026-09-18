# Aiko biological layer — fruit-fly mushroom body

Aiko's first living circuit: a mushroom-body (MB) associative memory grafted
from the real **MaleCNS v1.0** connectome (Berg et al. 2025, CC-BY,
https://male-cns.janelia.org). In the fly, the MB is the associative memory
center: odors fan out over ~2,000 Kenyon cells at ~5% sparsity, dopamine
teaching signals gate KC→MBON plasticity, and opponent approach/avoid MBONs
steer behaviour. This module replays that motif with real connectivity.

## What's real vs synthetic

| Part | Status | Source |
|---|---|---|
| Neurons, edges, weights (INPUT/PN, KC, DAN, MBON, APL/DPM) | REAL | `data/mb_microcircuit.npz`, sliced from MaleCNS v1.0 `minconf-0.5` flat connectome, synapse weight ≥ 5 (paper-standard threshold) |
| MBON valence signs (approach/avoid) | REAL, derived | per-MBON PAM-vs-PPL1 innervation: PAM≈reward, PPL1≈punishment. Cross-checks known biology (MBON01/γ5β'2a→+, MBON14/α3→−, MBON11/γ1pedc→−) |
| KC sparsity (~5%), APL divisive inhibition, MBON recurrence | REAL motif | topology + weights from data; normalisation constants functional |
| Feature→INPUT sensory projection | SYNTHETIC | fixed seeded random matrix (documented in `circuit.py`) |
| DAN reward source | SYNTHETIC interface | call `reinforce()` with Aiko's `valence_score / 2` |
| Plasticity rule | functional abstraction | opponent Hebbian on a KC→MBON overlay; base weights stay fixed |

Slice stats: 4,673 neurons (4,064 KC · 97 MBON · 332 DAN · 4 APL/DPM ·
176 top KC-input units), 121,706 synapses. See `data/summary.json`.

## GitHub artefacts used (no token, no GB downloads at runtime)

- `flyconnectome/2025malecns` — quantification method + release-bucket layout
- `connectome-neuprint/neuprint-python` — query reference (token path, optional upgrade)
- `natverse/malecns` — R reference package for the same dataset
- `janelia-flyem/male-cns` — project portal code
- Bulk flat connectome via public HTTPS (`storage.googleapis.com/flyem-male-cns`),
  sliced locally; only the 451KB slice ships here.

## Regenerating the slice

`tools/extract_mb.py` is the exact one-off script (needs pandas + pyarrow).
It downloads the 14MB annotations + 1.05GB weights to a scratch dir, keeps
the MB slice, and deletes the weights. Re-run to refresh from a new release.

## Wiring

`FlyMB.valence_bias(features, reward)` returns opponent bias in [-1, 1].
Grasp/consolidation integration lives behind `MEMORY_FLYMB_*` config flags
(see `config/memory.yaml`) and starts in shadow (log-only) mode.

## Naive readout is uncalibrated

Before any `reinforce()` call, MBON biases are arbitrary-signed per pattern
(the connectome gives topology, not meaning). Valence semantics enter only
through DAN teaching — same as the fly, which must also learn what predicts
reward. Downstream wires should treat untaught biases as weak priors.
