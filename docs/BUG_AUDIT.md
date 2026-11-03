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
- Shutdown continues remaining cleanup when an individual subsystem fails.
- A single session-loop cleanup guard covers normal returns, input failures,
  command failures, and turn exceptions, while preserving error propagation.
- Proactive check-in failures clear the active-session marker.
- The shared turn gate is released when memory switching fails or turn setup
  raises, including failures before inference begins.
- Session shutdown requests scheduler stop before draining memory writes and
  detaches the stopped scheduler from the notification registry. In-flight jobs
  are not forcibly cancelled.
- Version remains 0.2.0. Jetson Orin Nano 8 GB performance measurements and the
  remaining lifecycle and duplication review are still pending.

The latest full run completed with 2,095 unit tests and 30 integration tests
passing. These results are a baseline for the systematic review, not a claim
that the repository has no remaining bugs.
