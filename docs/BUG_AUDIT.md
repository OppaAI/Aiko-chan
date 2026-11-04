# Aiko codebase bug audit

This checklist breaks the repository into reviewable areas. Each area should be
audited against its callers, configuration, failure paths, and existing tests;
record reproducible bugs, fix them, and then mark the area reviewed. A passing
test suite is useful evidence, but does not by itself mean an area is bug-free.

## Review order

| # | Area | Main code | Status |
|---|---|---|---|
| 1 | Startup, shutdown, and session orchestration | `main.py`, `system/wakeup.py`, `system/orchestrate.py`, `system/schedule.py` | Code review completed; final validation recorded below; physical Jetson checks pending |
| 2 | Conversation routing, context assembly, and response streaming | `cognition/think.py`, `cognition/deep_think.py`, `cognition/inner_speech.py` | Current code pass completed; combined validation recorded below; real-model/device checks pending |
| 3 | Conscience, refusal gates, and user preferences | `cognition/conscience/`, `cognition/policy.py`, `cognition/attention.py` | Focused false-refusal fixes shipped; broad review not started |
| 4 | Agentic tasks, tools, and self-coding | `agentic/`, `Aiko-Playground` integration | Focused self-coding fixes shipped; broad review not started |
| 5 | Memory, retrieval, knowledge, and persistence | `cognition/memory/`, `cognition/knowledge/` | Not started |
| 6 | Fly-brain circuits and state wiring | `cognition/fly*`, `cognition/centralcomplex/`, `cognition/neural_state.py` | Focused wiring fixes shipped; broad review not started |
| 7 | Speech, listening, and sensory pathways | `sensory/`, `cognition/flysense/` | Not started |
| 8 | Web, desktop, mobile, and external interfaces | `interface/` | Not started |
| 9 | Configuration, security boundaries, storage, and deployment | `system/`, `config/`, `backend/`, `scripts/` | Focused path/TLS fixes shipped; broad review not started |
| 10 | Evaluation, observability, and developer tooling | `tests/`, `eval/`, `conscience-lab/`, `util/` | Not started |

## Already fixed in the current `dev` history

- Self-coding path confinement, worker context size, tool authorization, and
  generated-script result handling.
- False GF cancellation and overly broad conscience blocks; Fly-brain state
  routing and identity isolation.
- Capability embedding cache isolation, shadow CX bus leakage, zero-valued
  attention inputs, atomic coding-tool writes, and TLS CA fallback.

## Area 1 progress

- Memory readiness is published after the startup result is available.
- Thinking and the returned boot result use the same memory startup decision.
- Memory that finishes after its startup timeout is closed after cleanup.
- Fatal cognition failure before memory attachment also disposes unused memory
  after cleanup, using a shared guard against duplicate closes. Attached memory
  is conservatively retained because workers may already be using it.
- Fatal cognition startup skips the optional Fly catalog warmup.
- Fatal cognition startup also bypasses the optional MCP bridge wait, avoiding
  up to 120 seconds of delay before reporting the failure to the supervisor.
- Scheduler startup occurs after speech/listening configuration, so overdue
  jobs cannot use partially wired backends and wiring failures do not leave
  a newly started scheduler running.
- Transition from startup to chat uses a bounded spinner join so a stalled
  UI spinner cannot prevent the input loop from starting.
- Shutdown continues remaining cleanup when an individual subsystem fails.
- A shared setup guard stops the startup spinner even when frontend wiring
  raises before the session input loop begins.
- Partial frontend setup and ready-display failures use the shared shutdown
  path to stop the scheduler and created adapters and drain memory writes.
  Cleanup tolerates missing resources and runs once per session.
- WebUI boot and the plain CLI share a line-based console progress renderer:
  parallel steps retain their own state, completed steps show timings, and
  redirected/dumb terminals receive plain lines. Animation is disabled when
  console logging is enabled to avoid competing with diagnostic output.
- A single session-loop cleanup guard covers normal returns, input failures,
  command failures, and turn exceptions, while preserving error propagation.
- Proactive check-in failures clear the active-session marker.
- The shared turn gate is released when memory switching fails or turn setup
  raises, including failures before inference begins.
- Session shutdown requests scheduler stop before draining memory writes and
  detaches the stopped scheduler from the notification registry. In-flight jobs
  are not forcibly cancelled.
- Guest scheduler startup defers reflection and consolidation catch-up checks
  until a real owner is resolved, avoiding redundant checks against guest state.
- Scheduler start/stop operations are serialized; concurrent starts cannot
  duplicate the polling worker, and stopped runners cannot spawn new catch-up
  workers through start().
- Owner assignment is atomic, and owner resolution cannot launch catch-up
  workers after shutdown, including when shutdown arrives during its scan.
- Startup and owner resolution share one guarded catch-up launch path,
  preventing duplicate workers regardless of which happens first.
- Daily catch-up checks shutdown after acquiring the turn gate and retains
  pending dates when stopped or when switching memory users fails.
- Monthly catch-up acquires the shared turn gate, binds the owner's memory
  and identity, and skips work if the gate times out or shutdown arrives.
- System and user dispatch recheck shutdown after waiting for the shared
  turn gate, releasing it without starting queued jobs when stopped.
- Monthly verification is included in scheduler wakeup timing, so it cannot
  be delayed simply because the next daily or monthly job is later.
- The RSS fallback no longer opens an unused proc file, and regression coverage
  checks conversion with a 64 KB system page size.
- WebUI startup distinguishes bound/ready from failed or timed-out HTTP startup;
  configuration errors close the event loop and release startup waiters.
- WebUI exits request HTTP server shutdown, and failed post-boot WebUI startup
  stops the scheduler and idle learner and drains memory writes.
- Idle-learner waits are interruptible; shutdown prevents new idle studies.
- Proactive generation uses the shared turn gate and skips a contended check-in;
  replies finishing after a stop request are not displayed.
- Karaoke pacing is interruptible and cannot replace a still-running worker.
- Replacing the registered scheduler stops the old runner, and registry access
  is synchronized with shutdown and notifications.
- Launcher exit paths share one Fly-state flush in finally. SIGTERM unwinds
  through normal cleanup, while the existing two-stage SIGINT behavior remains.
- Reviewed launcher/configuration contracts: corrected the --name help text,
  documented override=True precedence, removed duplicated exit and catch-up
  launch logic, and checked imports for unused names in the four core files.
- Version remains 0.2.0. No device performance claims are inferred from cloud
  tests; Jetson measurement instructions are in JETSON_AREA1_VALIDATION.md.

## Area 1 completion boundary

The code-side pass covers startup success/failure, frontend setup, shared job
gates, repeated worker startup, owner resolution, shutdown, console rendering,
launcher/configuration consistency, and resource telemetry. It includes the
supporting frontend, idle-learning, and resource call sites required by those
lifecycles. This is a completed review pass, not a guarantee of zero bugs.

The only remaining validation item in this pass requires the physical Jetson:
peak RAM/swap, concurrent model loading, latency, and restart observations under
the actual 8 GB device's models and configuration. The cloud host is x86_64 and
cannot supply those results. Follow [the device checklist](JETSON_AREA1_VALIDATION.md).

Deliberate lifecycle limits: in-flight model work may finish after a stop request;
the runtime does not forcibly kill Python/C workers, close SQLite underneath
them, or unload externally managed inference servers. Unrecoverable boot failure
requires process exit/restart. Unattached memory is disposed where ownership is
known; attached or potentially active memory is retained to avoid unsafe closes.

Final combined verification for this code-side pass: 2,160 unit tests passed,
5 unit tests skipped, and 30 integration tests passed. There were 13 warnings
and 34 passing subtests. One existing web-fetch test used a fragile 50 ms
wall-clock assertion; it now verifies concurrency using a four-worker barrier.
These results are not a claim that the repository has no remaining bugs or
that the physical Jetson has been validated.

## Area 2 review

The code pass covers normal/greeting/web/task routing, context assembly, the
no-history and soft-gate paths, deep-think guidance, shared context workers,
primary/retry/fallback streams, and inner-speech single-flight/persistence.
See [conversation context](CONVERSATION_CONTEXT.md) for the request-fitting and
recovery contracts.

Confirmed issues fixed:

- No-history requests retain the current task; style directives reach the model
  without becoming recall/history text.
- The fully assembled normal-chat request is fitted to LLM_CTX_SIZE, preserving
  the latest request and required instructions. Optional evidence omissions are
  explicit, and impossible requests fail before the network call.
- Context workers carry each request's identity and mode flags. Nested work in
  the bounded pool cannot starve waiting parent workers.
- Retrieval uses retrieval-instructed embeddings rather than classifier vectors.
- Early route exceptions release active-user tracking.
- Failed model requests do not become conversation history or durable memories.
- Failed streams cancel pending early audio, retries filter private thinking,
  and both stream resources are closed. Thinking-only flags are request-local.
- Inner speech remains single-flight through persistence and snapshot updates.
- Inner-speech SQLite close shares the operation lock, preventing concurrent
  close/write from crashing the interpreter.
- Removed unreachable response-review code left after the correction pass was
  removed. Updated sentence-stream test fixtures to supply a valid current user
  request while preserving their existing assertions.

Token fitting is estimated, not exact model tokenization. The physical 8 GB
Jetson and real inference endpoints have not been exercised in this cloud pass.

Full unit verification: 2,180 passed, 5 skipped, 13 warnings, and 34 passing
subtests. Integration verification: 30 passed.
The remaining audit areas are paused at the user's request after this batch.
