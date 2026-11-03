# Third-party notices — biological circuit data

Aiko ships **derived, reduced circuit slices** extracted from public
connectome releases. The full connectomes are **not** redistributed.

## MaleCNS v1.0 (mushroom body, central complex, sensory grafts)

- **Dataset:** MaleCNS v1.0 — complete adult male *Drosophila melanogaster*
  central nervous system (brain + ventral nerve cord)
- **Portal:** https://male-cns.janelia.org/
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- **Citation:** Berg et al., *Sexual dimorphism in the complete connectome of
  the Drosophila male central nervous system*, Cell (2026).
  See also the bioRxiv preprint lineage and Janelia project page.
- **Producers / collaborators:**
  - Janelia FlyEM Project Team (HHMI Janelia Research Campus)
  - Drosophila Connectomics Group, University of Cambridge
    (Dept. of Zoology) / MRC Laboratory of Molecular Biology
  - Google Research (segmentation / flood-filling networks)
- **Bulk data:** `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`
- **Reference tooling (not runtime dependencies of Aiko):**
  - https://github.com/natverse/malecns (R convenience package)
  - https://github.com/natverse/malevnc (male VNC / MANC access)
  - https://github.com/connectome-neuprint/neuprint-python
  - https://github.com/flyconnectome/2025malecns (supplemental derived tables)
  - https://github.com/janelia-flyem/male-cns (project portal)

### What Aiko redistributes

Only compact NumPy/JSON slices under:

- `cognition/flymemory/data/`
- `cognition/centralcomplex/data/`
- `cognition/flysense/data/`

Base synapse weights in those files are **read-only**. Learned KC→MBON
plastic deltas and compass sleep pressure are stored separately in
per-identity SQLite files and are **not** part of the upstream dataset.

### What Aiko does *not* claim

These modules are **heuristic feature extractors** grafted onto chat memory
and attention. Topology and weights are data-derived; text features, reward
signals, learning rules, and most dynamics are synthetic functional mappings.
They are not a validated biophysical simulation of a living fly, and they are
not a source of factual world knowledge.

## Additional datasets (not currently shipped)

If future work adds FlyWire/FAFB, Hemibrain, MANC/MaleVNC, or larval CNS
slices, document each source, license, and citation in this file before
merge.
