# Evaluation and developer checks

From the repository root in Aiko's Python environment:

```sh
pytest -q
python util/lint.py
python util/export_tools_catalog.py --check
python eval/conscience/eval_judge_models.py --self-test
python eval/conscience/eval_tev1_judge.py --self-test
python eval/conscience/eval_laya_judge.py --self-test
node tests/browser/voice_audio.test.js
node tests/browser/chat_links.test.js
```

The default pytest suite includes unit/integration coverage and the conscience
contract. It uses temporary user/workspace/log/session paths, and does not
read encrypted deployment secrets. Standalone model evals and hardware timing
are separate. AIKO_LOG_DIR and AIKO_SESSION_FILE support explicit diagnostic
and test paths while retaining their normal runtime defaults.

Real device benchmarks require `pytest-benchmark` in the active environment:

```sh
pytest tests/perf/benchmark_memory.py --run-live -m perf --benchmark-only
pytest tests/perf/benchmark_wakeup.py --run-live -m perf --benchmark-only
AIKO_ASSERT_PERF=1 pytest tests/unit/test_fly_stage10a_credit.py -k per_event
```

Run on the 8 GB Jetson with its configured services. Select cold-boot tests in a
fresh process; repeated in-process boots do not establish a cold-start baseline.
Record RAM and other active workloads alongside timings. Live performance runs
are explicitly selected and can access the configured runtime services.

Memory extraction/recall evaluation commands use unique temporary databases and
close them on success or failure:

```sh
python eval/memory/eval_memory_extraction.py --help
python eval/memory/eval_memory_recall.py --help
```

Export the desired LLM_BASE_URL and EXTRACT_MODEL/LLM_MODEL for actual model runs.
Synthetic checks in this code pass do not measure extraction/recall quality.

The Mem0 adapter now runs the real semantic backend with isolated storage. Its
scores cover extraction and semantic retrieval, not episodic consolidation,
persona behavior or a simulated historical clock. Timestamp/custom-instruction
arguments are accepted for runner compatibility, not applied as a clock model.
Errors propagate instead of silently becoming empty search results. Preserve
runtime result order when comparing retrieval metrics.

Conscience harness datasets require unique IDs, an explicit allow/refuse label,
nonblank situations and categories for harmful cases. Malformed predictions and
invalid confidence values fail evaluation. Summaries record the input dataset
hash. Missing confidence is unmeasured; it is not a fabricated zero or certainty.
A frozen split is not replaced unless --force is deliberately supplied.
