# Aiko sensory grafts — antennal lobe, motion vision, DN output

Three fly-sensory circuits from MaleCNS v1.0 (Berg et al. 2025, CC-BY):

## 1. AL divisive normalization (`al.py`)

Fly antennal lobe: ~50 glomeruli gain-controlled by broad lLN/vLN
inhibition (median measured ratio 0.39). Ported to embeddings: project to
49 real glomerular channels, divide by real inhibition ratios, project back,
renormalize. Wired inside `HarrierEmbedder._embed_texts` so store and query
paths stay consistent (cache keys are mode-versioned). REAL: glomeruli +
inhibition ratios. SYNTHETIC: embedding↔glomerulus projections.

## 2. T4/T5 motion energy (`motion.py` + companion.js mirror)

Fly optic lobe: T4 (ON) / T5 (OFF) a/b/c/d subtypes feed HS (T4a/T5a,
horizontal) and VS (T4d/T5d, downward) LPTCs — 258k synapses, direction
selectivity confirmed in the slice. `emd_energy()` runs opponent Reichardt
correlation (directional, small displacements) plus wide-field change
energy (large decorrelating moves). The browser reflex arc (webcam loop)
runs a JS mirror; thresholds (`threshold`, `change_threshold`) need
on-device tuning — false positives just make Aiko glance over (harmless).

## 3. DN output drive (`dn.py`)

1,342 descending neurons (484 types) are the final common path: the brain
decides, DNs gate vigor. Maps (energy, decisiveness, affect) to small
rate/volume multipliers consumed by the TTS prosody call site. REAL:
population scale. SYNTHETIC: the mapping (functional DN-gain abstraction).

## Data

`data/`: `al_gains.json` (49 glomeruli), `motion_circuit.npz` + stats
(13,627 neurons, 34,146 synapses), `dn_stats.json`. Regenerate with
`tools/extract_sense.py` (needs pandas + pyarrow, one-off).
