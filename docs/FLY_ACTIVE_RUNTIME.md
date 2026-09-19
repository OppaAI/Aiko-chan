# Active MaleCNS runtime

This is a bounded, connectome-derived runtime—not a claim of a full fly-brain
simulation or biological sentience. The catalog can contain large source data,
but a turn activates a deterministic, size-limited forward subgraph.

## Contract

1. `ConnectomeCatalog` retains node IDs, type, region, edge weight/sign, and
   source/version metadata. Missing biological information is represented as
   `unknown`, never guessed.
2. `subgraph()` applies a fixed node budget and hop limit.
3. `ActiveDynamics` is a small rate model for that active set only.
4. `FlyRuntime` converts bounded readouts into the existing per-user
   `NeuralState`, and records an inspectable active trace.
5. Sensory adapters must have explicit consent before emitting a drive.
6. Avatar output is restricted to expression, viseme, and pose intents;
   external actions require an approval-capable scheduler/policy.

## Operations

Use the Fly Studio endpoint `/studio/fly/api/trace` to inspect the active node
count, edge count, source/version, truncation flag, rates, and DN/VNC-style
outputs for the current identity. It is read-only and has no-cache headers.

Run the target-device smoke benchmark after changing the runtime:

```bash
python -m tests.eval.fly_runtime_harness
```

The benchmark is intentionally synthetic. Production acceptance must also
measure full ASR + vision + LLM + TTS latency and memory on the Jetson.
