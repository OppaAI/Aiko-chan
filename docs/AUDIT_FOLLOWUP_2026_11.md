# Repository audit follow-up — November 2026

This pass updates the earlier BUG_AUDIT.md review against current `dev`. It
combines repository-wide syntax/undefined-name checks with the default test
suite and targeted review. Physical-device and live-service behavior remains
outside the cloud results.

## Confirmed fixes

- Cognitive-state `persist()` now updates its shared debounce timer rather than
  raising UnboundLocalError before any state can be saved.
- Entity spreading activation initializes adjacency, seed/frontier strengths,
  decay, and cutoff; it handles undirected weighted hops and excludes invalid
  edge weights.
- Workflow fallback logs the caught exception rather than raising another
  NameError while handling an invalid spec.json.
- GRASP debug tracing references the real capacity constant.
- Go/Shogi/Koi-Koi self-play locks can create their threads/locks; Koi-Koi's
  diagnostic references its actual card constants.
- Lingo vocabulary creation imports its card class, social reply-claim failure
  logging has a logger, and Codebase Studio markdown export resolves identity.
- Web fetch reaches available Tavily/Crawl4AI fallback strategies even when
  Trafilatura is absent.
- Codebase indexing supports embedding providers without a batch-size
  attribute while retaining bounded batches. An unavailable embedder still
  explicitly defers vectors and preserves lexical rows for a later pass.

## Test maintenance

New regressions cover debounced persistence, weighted entity propagation, and
invalid-workflow fallback. Existing indexing tests now expect explicit deferred
vector status during outages. The embedding context-retry test supplies its
own character cap and ordinary prose so deployment truncation/redaction cannot
silently bypass the retry case.

## Cloud audit closure and deployment boundary

Final combined default and standalone evaluation result: **2,851 passed,
6 skipped, 34 passing subtests**. There were 12 warnings, principally the threaded-fork deprecation in the
existing Fly process regression tests.

All 570 Python files parse. Production and test Python pass Ruff F821/F823 checks. The default suite is run in an
isolated environment with its declared test/runtime libraries and the sibling
Aiko-Playground worker contract. Tests using local event-loop sockets require
an executor that permits loopback; the restricted sandbox can otherwise stall
TestClient rather than establish an application defect.

The cloud static/regression audit and follow-up fixes are complete for this
checkout. Real inference/audio/camera services, Jetson 8 GB measurements,
installed packaging, optional platform accounts and SQLCipher still require
deployment validation. A passing suite does not prove every user journey or
third-party service works in deployment.

## Final follow-up fixes

- Comparison parsing accepts contrast/differences/pros-and-cons phrases and
  single-letter subjects. Equal graph scores prefer a structurally parsed
  comparison over a generic question trigger.
- The standalone route diagnostic delegates to the actual four-way policy,
  respects the configured LLM key/timeout, initializes models only on execution,
  and validates prompt files before initialization. Removed the obsolete
  ternary decision tree, fake predicted execution steps and undefined helpers.
- Migrated evaluation fixtures to current per-user chunk storage, embedding
  dimensions, automatic FTS triggers, graph APIs, report locations and verifier
  schema. Graph execution tests retain real orchestration with deterministic
  tools to avoid external side effects.
- Replaced process-randomized test hashing and the invalid constant-direction
  evaluation vectors with reproducible word features. This fixes spurious
  semantic matches without weakening production relevance thresholds.

Validation used an isolated virtual environment with declared lightweight
runtime/test libraries; no live model, camera, microphone or account was used.
The known threaded-fork deprecation remains a deployment concern for the Fly
worker tests and is not counted as proof of safe hardware concurrency.
