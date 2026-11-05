# Mem0 benchmark adapter

Copy `aiko_mem0_adapter.py` into the benchmark repository and import
`AikoMemClient as Mem0Client` from it. Set `AIKO_ROOT` to the Aiko checkout,
`LLM_BASE_URL` to the configured model endpoint, and `EXTRACT_MODEL` (or
`LLM_MODEL`) to the extraction model.

```sh
export AIKO_ROOT=/path/to/Aiko-chan
export LLM_BASE_URL=http://localhost:8080/v1
export EXTRACT_MODEL=your-extraction-model
```

Each client uses a new private temporary database by default and removes it on
close. Use the client as an async context manager, or call `await client.close()`.
`SQLITE_MEMORY_PATH` optionally selects a **new** scratch file for inspection;
existing files are refused to prevent a benchmark from clearing a live store.
Do not point this variable at a production database. The selected path is
available as `client.db_path`.

`AIKO_BENCH_DRY_RUN=1` tests protocol wiring without model calls; it cannot
produce meaningful accuracy scores. `AIKO_BENCH_USER` can explicitly force one
benchmark row identity. Distinct user IDs otherwise remain row filters within
the isolated database, without switching production profile/episodic stores.

This adapter measures semantic extraction/retrieval through MemoryBackend. It
preserves final result order and returns actual inserted IDs. Native calls run
outside the event loop and finish before cancellation can release the operation
lock. Model or storage errors fail the benchmark instead of returning fake empty
results. Closing releases both database and HTTP clients.

Episodic consolidation, the full AikoMemorize facade, persona behavior and a
historical clock are outside this adapter's scope. Observation timestamps and
custom-instruction arguments are accepted for protocol compatibility but do not
simulate historical time. Treat results as a new semantic-backend baseline;
older adapter results may reflect different or broken storage wiring.

See [Evaluation and tooling](../../docs/EVALUATION_TOOLING.md) for local checks.
