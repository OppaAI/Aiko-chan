# Aiko codebase bug audit

This checklist breaks the repository into reviewable areas. Each area should be
audited against its callers, configuration, failure paths, and existing tests;
record reproducible bugs, fix them, and then mark the area reviewed. A passing
test suite is useful evidence, but does not by itself mean an area is bug-free.

## Review order

| # | Area | Main code | Status |
|---|---|---|---|
| 1 | Startup, shutdown, and session orchestration | `main.py`, `system/wakeup.py`, `system/orchestrate.py`, `system/schedule.py` | In progress; initial scan found no confirmed new defect |
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

The latest full run completed with 2,095 unit tests and 30 integration tests
passing. These results are a baseline for the systematic review, not a claim
that the repository has no remaining bugs.
