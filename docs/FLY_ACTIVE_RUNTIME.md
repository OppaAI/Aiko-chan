# Active MaleCNS runtime

This is a bounded, connectome-derived runtime—not a claim of a full fly-brain
simulation or biological sentience. The catalog can contain large source data,
but a turn activates a deterministic, size-limited forward subgraph.

## Contract

1. `ConnectomeCatalog` retains node IDs, type, region, edge weight/sign, and
   source/version/SHA-256 metadata. A declared checksum is verified on load.
   Missing biological information is represented as
   `unknown`, never guessed.
2. `subgraph()` applies a fixed node budget and hop limit.
3. `ActiveDynamics` is a small rate model for that active set only.
4. `FlyRuntime` supports `shadow` (trace only) and `live` (bounded
   `NeuralState` publication) modes, and records an inspectable active trace
   including observation consent, cells, and edges.
5. Sensory adapters must have explicit consent before emitting a drive.
6. Avatar output is restricted to expression, viseme, and pose intents;
   external actions require an approval-capable scheduler/policy.

## Embodiment and autonomy boundary

The active graph can propose only reversible VRM body language (expression and
pose) from its current output plus the identity's existing affect state. These
proposals are traceable in Studio and are not automatically translated into
speech, messages, purchases, browser actions, hardware control, or new sensor
access. Those consequential capabilities remain behind their existing consent,
conscience, permission, and approval layers; a learned conscience is not a
safe replacement for independent technical controls.

The repository does **not** bundle the full MaleCNS release. The public release
is large (the complete weights artifact alone is about 1 GB) and this runtime
does not yet provide a validated, redistributable 166k-neuron importer. Its
catalog interface is ready to index such an extract once a verified source
artifact and its license/checksum are supplied. Shipping a made-up or partial
file as "all 166k neurons" would be misleading.

## Wiring after importing a catalog

A first verified catalog already exists (built 2026-09-18, not committed —
512MB, gitignored under `data/fly_catalog/`):

- File: `data/fly_catalog/male-cns-v1.0-w5.json`
- Contents: 211,577 nodes / 6,300,108 synapse edges at weight≥5 (paper
  standard), MaleCNS v1.0 minconf-0.5, CC-BY Berg et al. 2025.
- SHA-256: `27ba5e5d50a758ddb0f5df7b4b7c00e39435375e1f441750a8dbf40914c4e561`
  (verified on load by `ConnectomeCatalog.from_path`).
- Honest gaps, per contract: every node `region` and every edge `sign` is
  `"unknown"` (annotations carry no neuropil column; flat weights carry no
  sign — joining neurotransmitter predictions is future work).
- Rebuild: `python3 cognition/fly_runtime/tools/build_catalog.py
  --annotations <annot.feather> --weights <weights.feather>
  --out data/fly_catalog/male-cns-v1.0-w5.json` (streams; ~1GB sources).

Set `AIKO_FLY_CATALOG_PATH` to the verified JSON catalog produced from the
download and choose `AIKO_FLY_RUNTIME_MODE=shadow` before `live`. Optionally
set `AIKO_FLY_ACTIVE_BUDGET` (1–20000). The WebUI's explicitly submitted camera
or screen images create a consented, privacy-preserving visual observation;
only its scalar salience and modality enter the active graph, never image bytes.
The service selects matching catalog types (`sensory`, `T4`, `T5`, or `visual`)
and does nothing if the catalog, matching seed, or runtime mode is absent.

## Operations

Use the Fly Studio endpoint `/studio/fly/api/trace` to inspect the active node
count, edge count, source/version/checksum, truncation flag, rates, and
DN/VNC-style outputs for the current identity. It is read-only and has no-cache
headers. Its default visual projection is capped at 160 cells / 260 edges; this
does not alter the runtime's full counts.

Run the target-device smoke benchmark after changing the runtime:

```bash
python -m tests.eval.fly_runtime_harness
```

The benchmark is intentionally synthetic. Production acceptance must also
measure full ASR + vision + LLM + TTS latency and memory on the Jetson.
