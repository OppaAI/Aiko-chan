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

## Verification and remaining work

Final default-suite result: **2,800 passed, 6 skipped, 34 passing subtests**.
There were 13 warnings, principally the threaded-fork deprecation in the
existing Fly process regression tests.

Production Python passes Ruff F821/F823 checks. The default suite is run in an
isolated environment with its declared test/runtime libraries and the sibling
Aiko-Playground worker contract. Tests using local event-loop sockets require
an executor that permits loopback; the restricted sandbox can otherwise stall
TestClient rather than establish an application defect.

The broader audit remains open: real inference/audio/camera services, Jetson
8 GB measurements, packaging, optional platform accounts and SQLCipher need
deployment validation. Legacy standalone routing/evaluation helpers also have
stale undefined references and need their own migration pass. A passing default
suite does not establish that those standalone commands or every user journey
has been exercised.
