# Aiko codebase bug audit

This checklist breaks the repository into reviewable areas. Each area should be
audited against its callers, configuration, failure paths, and existing tests;
record reproducible bugs, fix them, and then mark the area reviewed. A passing
test suite is useful evidence, but does not by itself mean an area is bug-free.

## Review order

| # | Area | Main code | Status |
|---|---|---|---|
| 1 | Startup, shutdown, and session orchestration | `main.py`, `system/wakeup.py`, `system/orchestrate.py`, `system/schedule.py` | In progress; memory readiness and shutdown cleanup fixes shipped; Jetson 8 GB review ongoing |
| 2 | Conversation routing, context assembly, and response streaming | `cognition/think.py`, `cognition/deep_think.py`, `cognition/inner_speech.py` | Not started |
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
- Version remains 0.2.0. Jetson Orin Nano 8 GB performance measurements and the
  remaining lifecycle and duplication review are still pending.

Remaining review blocks (not counts of undiscovered bugs):

1. Failures during session setup before the input-loop cleanup guard, and
   ownership of resources created during failed boot.
2. Remaining dead code and configuration inconsistencies in the launcher and
   orchestration paths.
3. Peak RAM, concurrent model loading, and latency measurements on the actual
   Jetson Orin Nano 8 GB.

Combined verification at `abee4b82` completed with 2,112 unit tests and 30
integration tests passing (5 unit tests skipped). These results cover the
changes through that commit. Later catch-up and wakeup changes passed their
focused scheduler suites; these checks are not a claim that the repository
has no remaining bugs.
