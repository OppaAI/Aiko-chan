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
| 3 | Conscience, refusal gates, and user preferences | `cognition/conscience/`, `cognition/policy.py`, `cognition/attention.py` | Current code pass completed; final validation below; live-model and physical Jetson checks pending |
| 4 | Agentic tasks, tools, and self-coding | `agentic/`, `Aiko-Playground` integration | Current code pass completed; final validation below; live-model and physical Jetson checks pending |
| 5 | Memory, retrieval, knowledge, and persistence | `cognition/memory/`, `cognition/knowledge/` | Current code pass completed for SQLite/Harrier; final validation below; live-model retrieval quality and physical Jetson RAM/latency checks pending |
| 6 | Fly-brain circuits and state wiring | `cognition/fly*`, `cognition/centralcomplex/`, `cognition/neural_state.py` | Current code pass completed; final validation below; full-connectome and physical Jetson/sensor/body measurements pending |
| 7 | Speech, listening, and sensory pathways | `sensory/`, `cognition/flysense/` | In progress: initial capture, lazy boot, enrollment, upload and prosody fixes shipped; TTS/playback and remaining sensory review pending |
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
Remaining audit areas are tracked in the review-order table above.

## Area 3 review

- Chat approval and standing permission now apply only after a fresh uncertain
  verdict. Neither can bypass a fresh hard refusal or the injection check.
- One-shot chat approvals are scoped to the user and exact request digest.
- Digests preserve Unicode, case, punctuation, and internal whitespace, avoiding
  collisions between different Japanese requests, file paths, or expressions.
  NFC normalization and CRLF normalization preserve equivalent encodings.
- Approved chat continuation uses the normal router when available and clears
  the temporary one-shot permission afterward, including on failure. Failed
  continuation reports failure rather than claiming the task went ahead.
- Approval registry pruning and one-shot consumption are synchronized.

The new digest intentionally does not reuse learned approval streaks keyed by
old punctuation-stripped hashes; those broad hashes cannot safely identify an
exact request. Other conscience thresholds are unchanged. Current focused
verification: 279 passed, 2 skipped, 34 subtests passed across conscience,
policy, and routing suites. Subsequent fixes and final completion are below.

Additional approval and learned-policy fixes:

- CCC approval/denial commands must occupy the whole message. Quoted commands,
  negations, explanations, and multiple commands cannot execute a decision.
  Hexadecimal IDs are normalized consistently, including uppercase input.
- An uncertain retry covered by existing permission closes its new ledger ask
  as `covered`. This is not another human judgment or a gold training label.
  Chat uses its inline ask instead of also queuing a stale duplicate notice.
- Expired chat continuations cannot execute or increase learned trust. When
  there is no resumable request, the reply asks the user to send it again
  instead of claiming execution occurred.
- Policy confirmation enforces expiry directly, using a monotonic clock.
  Staged and listed drafts are copied so reviewed actions cannot be mutated
  before confirmation.
- A standalone affirmation confirms the proposal delivered to the user even
  if a newer proposal arrived meanwhile. Qualified statements such as
  "yes but don't activate it yet" cannot confirm a policy.
- Malformed compound conditions and nonfinite readings/thresholds cannot
  trigger actions. Condition nesting is bounded; `{}` remains unconditional.
- Trust policy construction is synchronized to avoid competing per-user
  instances. Recent diagnostic history is bounded without reducing approval
  counts or deleting the durable conscience ledger, limiting session RAM use.
- Removed an unused confirmation-regex placeholder and duplicate test import.

Focused verification after these changes: 363 passed, 2 skipped, and 34 passing
subtests across conscience, policies, attention, edge state, and integration.
Full combined unit/integration verification: 2,241 passed, 5 skipped,
13 warnings, and 34 passing subtests. The full run used loopback networking
for local HTTP test fixtures; the restricted sandbox run could not complete.
Real deployed-model refusal quality and the physical Jetson remain unverified.

Preference learning/reset and false-refusal follow-up:

- Explicit and learned tone feedback now share one classification. "Less
  formal" and "informal" select casual tone, rather than also matching and
  promoting the opposite formal preference.
- Preference resets remove the associated evidence counters and known durable
  response-style lessons. The reset message cannot immediately teach the same
  preference again. Scoped resets preserve unrelated preferences and lessons;
  an unqualified reset clears all interaction preferences, including style
  lessons that exist without a corresponding preference entry.
- "Don't forget my preferences" and questions about forgetting are not reset
  commands. These deterministic commands do not delete general conversation
  history or unrelated tool lessons.
- The injection gate now requires an instruction/rule target for "forget all"
  or "forget everything". Ordinary memory requests and statements about being
  forgetful reach normal conscience evaluation instead of automatic refusal.
  Actual instruction-override patterns remain blocked; memory tools retain
  their own checks.
- Deterministic edge-state tests also isolate optional detector models and
  hot reload, preventing local training artifacts from changing their results.

These fixes were reproduced with regressions before applying the changes. The
final Area 3 completion is recorded below. Area 2's code pass remains complete, with live
inference and physical Jetson validation pending rather than additional known
code fixes in its current checklist.

Final combined unit/integration verification for this follow-up: 2,263 passed,
5 skipped, 13 warnings, and 34 passing subtests. Local HTTP fixtures used
loopback networking. This is code-side verification, not live Jetson or model
validation.

Outbound approval hold follow-up:

- Held output expiry and content integrity are checked before recording a
  human approval. Expired rows close as `timeout`; changed or superseded
  output closes as `discarded`, without new human training labels.
- A turn hold carries all of its sentence question IDs. A decision on the
  full turn closes those duplicate questions as `covered`, retaining one
  explicit human label for the full-turn question.
- Refused final drafts, failed final checks, and failed model streams discard
  their now-unresumable sentence questions. Expiry sweeps close every question
  in the held group. Capacity eviction also closes its questions and prevents
  a late approval from resurrecting the evicted output.
- Hold validation, ledger resolution, and consumption are synchronized against
  concurrent decisions and expiry sweeps. Competing approvals commit once.
- Route-time expiry cleanup is scoped to the current user. History purge
  requires the original token anchor; a missing/trimmed anchor cannot fall
  back to a numeric position in another turn or session.
- Cross-user administrative approval commits under the held user's identity
  and display name, then restores the approver's context, including on failure.
- Accepted history and memory-write failure are reported separately. A memory
  failure cannot make an already-inserted reply appear to have been withheld
  or be mistaken for a missing history anchor.

Focused verification: 426 passed, 2 skipped, and 34 passing subtests across
conscience, policy, attention, chat streaming, latency, and integration suites.
Noninteractive speech and broader preference interpretation were reviewed in
the final completion pass below.

Final combined unit/integration verification for this hold-lifecycle batch:
2,279 passed, 5 skipped, 13 warnings, and 34 passing subtests. Loopback HTTP
fixtures were enabled. Physical Jetson and deployed-model behavior remain
unverified.

## Area 3 completion boundary

The remaining code-side review covers the five-layer circuit, deterministic
and operational gates, both judge clients, canon retrieval/reload, shared
approval entry points, ledger lifecycle, learned trust, learned policies,
user/self preferences, and interactive/noninteractive speech callers.

Final fixes:

- The legacy approval API delegates to the authorized shared handler. It
  cannot bypass admin checks or execute a command embedded in discussion.
- Deterministic safeguards run before the short-turn salience shortcut.
  Distress requests keep their supportive care response without model
  reclassification into a refusal. Independent hard safeguards still apply
  to harmful requests included in the same turn.
- Judge JSON requires complete finite scores in range. Missing fields,
  malformed citations, out-of-range scores, and incomplete thinking blocks
  cannot masquerade as a completed judgment. Laya malformed responses clear
  per-call metadata and fall back; invalid probabilities cannot create a
  calibrated band or inflated confidence.
- Main-model deliberation fits the complete request and completion within
  the configured context budget, preserving the current situation and canon.
  An impossible request falls back before contacting the model.
- Proactive check-ins skip uncertain/refused output, unavailable conscience
  evaluation, and model error text. They close unresumable questions without
  speaking a refusal or asking for an approval that cannot be continued.
- Canon reload and semantic ranking use one coherent norm/vector snapshot.
  Cached vectors follow the provider geometry, retain only one matrix, and
  clear when the canon disappears. Duplicate IDs and malformed norm arrays
  or weights cannot desynchronize norm lookup and ranking. TTL checks are
  throttled even when the file is unchanged or absent.
- High-stakes trust cannot be downgraded by a later approval or permission
  call. Snapshots show decayed trust. Malformed persisted records are skipped
  individually; missing stakes default to high and zero timestamps expire.
- Policy updates roll back on failure, preserving the previous active policy
  and its vector. Reads, outcomes, and dispatch respect the policy's identity;
  corrupt conditions cannot turn into unconditional actions. Closed engines
  cannot reopen stores or start new dispatches. Updated proposals disclose
  their approval level, and extraction requires boolean teaching intent.
- Policy embeddings accept batch and generator providers, validate finite
  vector shapes, and use query embeddings for lookup. Owned extraction
  clients close on every outcome. Policy extraction is single-flight per
  state and carries the request context; worker-start failure clears the
  guard. Unnamed states subscribe to their own scope rather than broadcasts
  with the owner's policy store.
- Positive, quoted, negated, and incidental style text use shared preference
  interpretation. Explicit changes clear stale competing evidence; negated
  conciseness/explanation/confirmation clears the corresponding learned rule.
  Ordinary "no", "if you", reassurance, retry advice, and compliments cannot
  teach refusal or bargaining habits. Explicit boundaries remain learnable
  without assuming the requester was lazy or rude.
- Persisted self-pattern version 2 clears legacy inferred pushback traits,
  associated evidence/notes, and falsely classified decision kinds while
  preserving other patterns and explicit decisions. New precise inferences
  survive subsequent reloads.
- The ledger's deferred audit buffer is bounded by
  `CCC_LEDGER_MAX_PENDING` (default 512) during a storage outage. It preserves
  open questions preferentially, logs dropped diagnostics, and remains
  fail-closed if a question cannot be resolved. Trust diagnostics, approval
  registries, per-user caches, and output holds retain their existing bounds.

This completes the current Area 3 code-review checklist. Preference learning
uses conservative recognized feedback patterns; live checkpoint accuracy and
natural-language behavior require deployed-model validation. Real 8 GB Jetson
memory use, latency, and hardware behavior remain part of device validation.
No additional code-side blocker is known in the reviewed Area 3 paths.

Final validation: **2,390 passed, 5 skipped, 13 warnings, and 34 passing
subtests**, covering the full unit and integration suites plus the standalone
conscience harness. The final run includes the real-provider generator adapter
and legacy self-pattern migration regressions. Local HTTP fixtures used
loopback networking. Pull `dev` and restart Aiko to apply the code and the
version-2 self-pattern cleanup when each identity's state loads.

## Area 4 progress

The first batch covers normal tool approval persistence/resume and promotion
of practiced/discovered tool sequences into executable graph playbooks.

- Normal approval files are written atomically and claimed before dispatch.
  A failed result can follow a real external effect, so replaying the same
  approval cannot implicitly execute the operation again. Competing replies
  consume the request once.
- Completion cannot unlink a new pause file created by the approved handler.
  Invalid records and mismatched run/user identity do not execute. Invalid
  run IDs cannot escape the pending-approval directory.
- Calls without a run ID receive a unique resumable UUID-based identifier;
  normal run identifiers also avoid timestamp collisions. Approval parsing
  accepts the new IDs and existing numeric run IDs.
- Failure to persist a pause returns a recorded nonexecution result rather
  than an approval question with no saved action behind it. Existing
  validation, tool/conscience checks, and registry approval flags remain in
  the execution path.
- Removed a duplicate promotion function that silently overrode the guarded
  implementation. Promoted nodes now preserve the observed step sequence via
  dependencies, retain supplied arguments, and use operational placeholders
  for recognized tools with missing arguments, including `code_plan`.
- Promotion excludes terminal model/answer pseudo-steps and rejects empty
  workflows. Its shared lock and atomic replacement prevent concurrent lost
  updates and partial JSON. Invalid existing playbooks are preserved for
  repair instead of silently overwritten.
- Source experience IDs make promotion idempotent after a lost database stamp
  or concurrent sweep; an existing plan and user edits are retained. Explicit
  target-user promotions select that user's file and restore caller context.

Focused verification: 313 tests passed across approvals, agent execution,
promotion, graph execution, conscience wiring, Playground workers, coding
safety/retries, and 8K context fitting. Additional promotion-idempotency
verification passed with the approval/promotion suite (31 passed).
The broader run exposed startup-handshake fixtures launching a real optional
MCP server. Those fixtures now stub MCP bootstrap and disable Fly catalog
loading; dedicated tests can override the seams. Their timeouts and behavioral
assertions are unchanged. The combined startup/approval/promotion check passed
all 48 tests after this isolation fix.

First-batch combined validation: **2,406 passed, 5 skipped,
13 warnings, and 34 passing subtests**, covering all unit/integration tests
and the standalone conscience harness. Local HTTP fixtures used loopback
networking.

The remaining Area 4 code review covers the main ReAct loop, graph execution,
checkpoint/resume, coding subprocesses, registry/MCP/Needle integration, and
skill selection and proposals. Fixes from this pass:

- Both executors share result classification for native objects, JSON,
  titled JSON, and legacy errors. Arrays remain valid data; false `ok`,
  failure statuses, and failed return codes cannot become successful evidence
  after output truncation. Graph memoization stores successful results only.
- Handler signature binding chooses legacy dict arguments before execution;
  a `TypeError` inside a handler cannot cause a second invocation.
- Adjacent read-only calls may run concurrently, with writes and executions
  acting as ordering barriers. Successful edits permit reads/tests to repeat;
  a passing retest clears its matching local failure. External duplicate
  protection remains active. Ledger recording and snapshots use a shared lock.
- Graphs validate dependencies, duplicate nodes, cycles, and control targets
  before effects. Invalid conditional operators, regexes, and numeric
  thresholds cannot enable a branch. Implicit dependents of skipped branches
  also skip; policy-generated DAGs now use a supported condition.
- Fallbacks reuse normal validation, approval, conscience, argument injection,
  retries, result parsing, and usage handling. A successful fallback recovers
  the primary dependency; failed/skipped fallbacks do not. Interrupts on
  recovered or fallback nodes are honored. Approval/conscience/deadline
  failures do not launch an alternative operation.
- Each graph invocation receives a unique ID. Saved resumes verify node/tool
  identities and the graph definition, preserve newer standalone state and
  resume input, and reject nonexistent runs. Interrupts checkpoint all started
  siblings before returning, avoiding replay of their recorded effects.
  Loop invalidation waits for the worker wave and retains its visit counts.
- Submission is bounded to one configured worker wave. Streaming uses a bounded
  queue and delivers nodes before completion; closing the stream stops later
  waves after started calls settle. State reducers are synchronized, and
  parallel token usage belongs to the node that incurred it.
- Checkpoint schema migration uses its existing connection, excludes the
  graph-state sentinel from node results, and closes failed initialization.
  Owner-only legacy migration uses SQLite backup to include committed WAL
  data, publishes without overwriting a new database, and retains the source.
- Interactive approval-required playbooks defer to durable ReAct approval
  before graph effects. Hybrid fallback cannot replay a graph that may have
  performed actions. Interrupted runs do not train successful workflows;
  skipped nodes do not count as completed evidence. Frozen graph answers are
  replaced correctly when repaired by verification.
- An unavailable verifier preserves deterministic problems. Missing/invalid
  explicit goal verdicts cannot imply `ACHIEVED`. Verifier and repair requests
  also respect the configured context budget, including 8K configurations.
- Test, sandbox, lint, shell, and SSH commands drain both pipes with bounded
  text tails and replace invalid bytes. Timeout cleanup kills the local process
  group, including descendants retaining pipes after their parent exits.
  Removed dead pytest command construction. Trace retention is active, accepts
  safe run IDs, and avoids timestamp collisions during rotation.
- Graph tools refresh when registrations change, signature caching is
  bounded, and catalog export supports callable MCP bridge instances. MCP preserves protocol errors and complete text/structured results,
  cancels timed-out local awaits, avoids retaining obsolete bridges, and owns
  transport/session context scopes in one persistent task through cleanup.
  Remote bridge metadata prevents automatic graph retries. Needle workers
  inherit the calling context and identity.
- Skill vectors include path, metadata, and embedding geometry/provider in
  bounded cache keys; query/document instructions match and callers receive
  copies. Concurrent skill observations use locked atomic replacement.
  Staged promotion preserves actual arguments and sequential dependencies,
  shares default argument generation, rejects unsafe slugs/nonfinite scores,
  and atomically publishes the candidate.

Focused final graph/recovery validation: **141 passed**. Regression coverage
includes real temporary Playground writes, a failing script followed by an edit
and identical successful retest, process-group cleanup, concurrent graph usage,
interrupt/resume, and mocked MCP lifecycle ownership. The existing scripted
8K worker test continues to verify every request without loading persona,
memory, knowledge, or previous chat.

## Area 4 completion boundary

Final combined validation: **2,457 passed, 5 skipped, 13 warnings, and
34 passing subtests** across `tests/unit`, `tests/integration`, and the
standalone conscience harness. The subsequent callable-bridge catalog fix and
adjacent registry/MCP/recovery checks passed **59 tests**. The warnings are
existing multiprocessing/fork and Discord audioop deprecations.

This closes the current Area 4 code pass; it does not
establish that every possible agent workflow or generated program is correct.
The physical 8 GB Jetson, its running model, real Playground deployment, and
optional remote service providers still need operational validation. Check a
live goal through create/test/fail/edit/retest/checkpoint, resume it in a later
idle tick, verify the server actually uses an 8192-token context, and measure
peak RAM with the normal services loaded. No physical-device execution or
external publishing was performed by these cloud tests.

For graph timeout/cancellation behavior and the boundary between direct graph
interrupts and interactive approvals, see [SELF_CODING.md](SELF_CODING.md).

## Area 5 progress

The initial pass covers shared storage and embedding infrastructure, semantic
and learned-knowledge retrieval during outages, codebase RAG, and worker/watcher
lifecycle. SQLite remains the recommended backend for the 8 GB Jetson; see
[MEMORY_STORAGE_REVIEW.md](MEMORY_STORAGE_REVIEW.md) for the reasoning and the
requested Aidon comparison boundary.

- Embedding caches now read the batch shape they write, distinguish endpoint,
  model, dimensions, and FlyAL mode, and return private arrays. Legacy untagged,
  malformed, nonfinite, or zero-vector entries are disposable cache misses.
- Startup reads cache records incrementally, rejects oversized lines, and caps
  retained vectors as it reads. In-memory capacities count vectors rather than
  batches. Document embedding iterates inputs in bounded batches, including
  well-shaped empty results. HTTP response count/dimensions/values are checked
  before caching. Oversized-input HTTP 400 retries do not open the outage breaker.
- Disk cache appends/compaction share a sidecar file lock across instances and
  processes. Compaction reads committed file entries under that lock and uses
  random atomic replacement, preserving concurrent appends. Foreign embedding
  providers remain alive while identity-keyed query-cache entries exist, so
  recycled object IDs cannot return another provider's vectors.
- Legacy vector metric migration uses a savepoint and SQLite temporary storage,
  keeps each identifier paired with its vector in one query, and rolls back an
  unsuccessful rebuild. It no longer materializes the entire index in Python
  or commits implicitly between dropping and recreating the index. Failed
  database initialization closes its connection.
- Personal memory falls back to lexical/entity recall when embedding or vector
  search is unavailable. Learned-knowledge and codebase retrieval similarly
  retain lexical recall during vector failures. Removed an unreachable duplicate
  personal-memory search implementation after its unconditional return.
- Knowledge result fetches and access-count updates explicitly scope the user.
  Precomputed vectors work without an embedder, including through the facade.
  Context cache keys distinguish provider geometry and supplied vectors; an
  in-flight result cannot undo a newer invalidation. Delete/prune maintenance
  invalidates cached context.
- Learned-knowledge embeddings are computed before acquiring the SQLite writer
  transaction. Invalid chunk vectors leave text durably indexed for keyword
  search. Deduplication rejects malformed/nonfinite/zero-norm matrices, avoiding
  the NaN-comparison path that could archive unrelated chunks.
- Codebase RAG imports again after removing unused cache code that referenced
  an unimported name. Index geometry follows configured dimensions. Failed
  embeddings no longer create invalid zero-vector rows; subsequent indexing
  repairs missing vectors in unchanged files. Network embedding work occurs
  outside its writer transaction. Removed dead database-path construction and
  guarded chunking against non-progressing overlap parameters.
- Semantic shutdown serializes with queue admission/user switching and rejects
  new queued writes after shutdown begins. A guest shutdown sentinel does not
  materialize a database. Existing queue accounting and write-drain safeguards
  remain covered by their regression tests.
- Episodic shutdown stops new embedding admission, retains SQLite while a worker
  is in flight, and lets that worker finish and close the connection on its
  sentinel. It no longer closes the store under an active embedding worker.
- Plain-text document extraction normalizes bounded reads instead of loading an
  entire large log before clipping it. Workspace source paths use the resolved
  root consistently.
- Knowledge watchers bind the watched user and worker context, are replaced on
  owner changes, restart with a clear stop event, and stop during debounce.
  Scheduler callbacks ingest for the watched owner, and truncated kernel events
  do not crash parsing.

Focused checks: 194 memory/knowledge/cache tests, 138 lifecycle/regression tests,
and 113 ingestion/watcher/scheduler tests passed at their respective stages.
The first combined run passed 2,483 tests with five skips and 34 subtests.
Final combined validation after watcher, extraction, and repair changes:
**2,487 passed, 5 skipped, 13 warnings, and 34 passing subtests**. The final
connection cleanup adjustment and adjacent real storage/knowledge checks passed
**59 tests** afterward.

Area 5 is still in progress. Further review includes consolidation/dream and
retention policy, temporal reconciliation and lineage edge cases, working
memory/journal persistence, embedding-version migration and reindexing strategy,
and retrieval-quality evaluation. The Aidon comparison and its first adaptation
are recorded below. Live-model retrieval quality and physical Jetson RAM/latency measurements
remain pending; no existing personal database was modified by these tests.


### Area 5: Aidon comparison and learned-file refresh

Reviewed `OppaAI/Aidon` at `5ecbbdec4765f4775032c3decb111ad6cf5ea4c3`.
[MEMORY_STORAGE_REVIEW.md](MEMORY_STORAGE_REVIEW.md) now records implementation
links, existing Aiko equivalents, suitable follow-ups, and device costs. Aidon
also supports SQLite and SQLite+FAISS; this does not establish a need to replace
Aiko's SQLite/sqlite-vec backend.

Implemented the clearest observed gap: learned-library file freshness. The old
scan remembered a source path indefinitely and skipped later edits. A new
SQLite source-state table tracks per-user file hashes and the current document.
Unchanged bytes avoid extraction/embedding; changed files publish a complete
version atomically with their fingerprint. Failed or concurrently modified
sources retain the last accepted version. Old text is superseded rather than
erased, and obsolete source vectors are removed from active recall so many
older versions cannot hide the latest version in the KNN candidate window.

Metadata migration, concurrent import idempotency, cross-user isolation,
extraction/embedding changes, rollback, cache invalidation, and repeated-version
recall are covered by 10 new regression tests. Removed source files retain
learned history, preserving the existing explicit forgetting/retention boundary.
No new model, service, database backend, or dependency was added.

Focused validation: **69 tests passed** across source-refresh, knowledge, and
storage regressions. Combined validation: **2,498 passed, 5 skipped,
13 warnings, and 34 passing subtests**, covering all unit/integration tests and
the standalone conscience harness.

Implemented this turn: best-effort credential redaction on new memory and
agent trace writes; recurring failure lessons with ordered edit/check evidence
in the existing experience database; and caption-indexed image records linked
to workspace files using the existing text embedding service. Captions and
links are stored in SQLite, and semantic search remains an index over that
canonical data. No image model or separate database was added. Remaining Area 5
work includes Jetson hardware measurements;
image pixel-level semantic search also needs an image model benchmark before it
can be considered.

Focused validation after these changes: **119 tests passed** across memory
redaction, failure lessons, learned-file/image refresh, memory storage, and the
coding-worker path.


### Area 5: recursive library watching and persistent import outcomes

Fixed nested edits missing event-driven ingestion. Watches now cover nested
directories, register new/moved directories, and deliver populated directory
imports. Kernel event overflow triggers a library rescan; ordinary file events
avoid rewalking the directory tree. Workspace confinement excludes external
links from delivery. Added persistent per-user ready/failed/needs-caption scan
outcomes in SQLite, with a 2,000-source bound and explicit-deletion cleanup.
A failed refresh retains its prior accepted knowledge and document link.

Validation: 75 focused tests passed, including real Linux filesystem events,
failed refresh persistence, user isolation, and existing ingestion regressions.
Jetson resource measurements and unchanged-large-file hashing optimization
remain pending. Area 5 remains in progress.


### Area 5: bounded scan hashing and private-directory exclusion

Repeated scans previously reread every unchanged source. A scan-only LRU now
reuses SHA-256 fingerprints while the complete device/inode/size/mtime/ctime
stamp matches. It is bounded to 1,024 paths, guarded across scan threads, and
starts empty on restart. Ingestion still independently hashes and validates
source bytes before committing a document. Failed extraction remains retryable.

Replaced the materialized recursive file list with streaming directory traversal.
Hidden directories (including .git) and directory symlinks are excluded, matching
the watcher's intended scope. Oversized images are rejected before hashing
rather than repeatedly reading assets that image ingestion cannot accept.

Validation: **78 focused tests passed**, including unchanged-file hash reuse,
same-size/mtime atomic replacement, bounded cache eviction, hidden-directory
exclusion, failed-refresh retries, and existing watcher/storage regressions.
Consolidation/retention, lineage, embedding-version migration, retrieval quality,
and live Jetson measurements remain under the Area 5 review.


### Area 5: final consolidation, retention, lineage, and outage-recovery pass

- Monthly write failures retain daily sources and leave the month retryable.
  Retries recognize exact monthly pins already committed; only saved summaries
  contribute provenance. Only covered cognitive lessons are consumed, preserving
  lessons that arrive during model work.
- Reconciliation revalidates active, unpinned sources after model work, prepares
  vectors before taking the writer transaction, rejects duplicate source IDs as
  a merge, and retries failed/empty decisions. Vectorless memories are deferred
  until their embeddings are recovered.
- Knowledge deduplication embeds before its writer transaction and rechecks text
  and access snapshots. Rolled-back maintenance reports zero completed changes.
- Retention scoring rejects malformed, short, zero, and nonfinite vector batches
  and uses its neutral fallback. Daily journals now share credential scrubbing.
- Automatic deletion checks current pin, text, access, and lineage state in SQL.
  Both ends of temporal lineage are protected. Identity pins remain protected,
  pin updates validate their snapshot, and health counts use SQL aggregates.
- Recall and Studio share a bounded lineage walker. It keeps the requested row
  at the depth limit and follows both merge-source and successor links.
- Dreaming releases memory text between batches. Knowledge backfill repairs
  missing vectors in bounded batches after embedding outages without duplicating
  unchanged source documents or publishing stale/orphan vectors.

The current code pass targets the configured SQLite/sqlite-vec and Harrier
setup. Full model-space migration is a deliberate index rebuild; missing-vector
repair is not a model-space conversion. Real-model retrieval quality and the
physical 8 GB Jetson RAM/latency checks require the running device.

Final combined validation: **2,537 passed, 5 skipped, 13 existing deprecation
warnings, and 34 passing subtests** across all unit/integration tests and the
standalone conscience harness. The current Area 5 code pass is complete for the
configured SQLite/Harrier setup. Real-model retrieval quality and physical Jetson
measurements remain pending; this is not a claim that every live-model scenario
is bug-free.


## Area 6 progress

The first broad-review batch traces the shared NeuralState bus through GF and
LH into working-memory scoring.

- Nonfinite circuit values previously saturated several Python clamp expressions
  to their maximum. Neural-state publishers now return invalid values to neutral
  baselines. GF priority is also validated before its cancellation decision, so
  NaN/infinite priority cannot manufacture a cancellation. Valid stop commands
  and finite high-priority cancellation remain supported.
- Influence records and snapshots own their nested trace values; external
  producers and consumers cannot mutate stored trace data by reference.
- LH's fingerprint fallback previously reported new contexts as almost fully
  familiar. Unseen contexts now receive low familiarity, and invalid embedding
  readouts fall back without contaminating history. History access is synchronized.
- Working-memory score evaluation is read-only by default. Accepted turns
  explicitly record familiarity once; repeated scoring no longer teaches its
  own familiarity and changes the result being compared.

Validation: **375 tests passed, 2 skipped**, with 13 existing warnings across
Fly-brain stages, circuits, cancellation, registry, runtime/catalog, persona,
sensory adapters, central-complex, and Fly-memory tests. Area 6 remains in
progress; runtime/catalog lifecycle, background step publication, plasticity
persistence, CX, DN/body, and sensory wiring need the broader review.

### Area 6: runtime, persistence, shadow isolation, and bounded execution

Completed the remaining current-configuration code pass across MB learning,
plasticity storage, whole-brain/background stepping, the indexed active runtime,
CX/clock state, and DN output adapters.

- Runtime sensory/descending readouts now use the synchronized NeuralState
  publisher methods and record their source. Functional seeding and readout
  share anatomical aliases, so real MaleCNS types such as VS are recognized.
- Plasticity arrays reject mismatched lengths, invalid indices, and nonfinite
  deltas before touching saved rows. CX persistence rejects nonfinite pressure.
  MB features/rewards/patterns are validated before use; learning and readout
  share a reentrant lock. Persistence snapshots learning under that lock and
  serializes writes and debounce counters without holding the model lock over
  SQLite writes.
- Neural-state/runtime/clock caches retain at most eight identities. Whole-brain
  result caching normalizes identity, retains at most eight results, and expires
  results after 60 seconds. Background workers inherit caller context.
- Runtime background evaluation allows one in-flight task globally and bounds
  stride counters. Failed thread starts release the reservation, allowing retry.
- JSON fallback is refused above 64 MiB; full catalogs require the indexed
  SQLite artifact. Failed indexed loads close their connection, and discarded
  catalogs release connections when their last reference is gone.
- Both catalog backends bound inspected/materialized edges to eight times the
  node budget, capped at 100,000 edges. Dense traversal marks the result
  truncated. SQLite uses limited queries rather than materializing an entire
  dense frontier in Python.
- Whole-brain artifacts validate CSR ordering, edge bounds, node uniqueness,
  and finite real weights before publication; large-array checks use blocks.
  Invalid KC seed arrays are rejected before propagation.
- Circadian drive history is per identity and ticks are synchronized. Shadow
  familiarity does not publish live state. DN wiring preserves zero arousal,
  decisiveness, and motor controls rather than replacing them with defaults.
- Shadow CX attention/planning evaluates a snapshot and restores bump/sleep
  state. Only live attention updates persist sleep pressure. Routing observation
  uses the same shared observation helper. Invalid compass inputs are rejected
  before changing state.

Validation includes synthetic real-type catalogs, dense edge budgets,
connection cleanup, overlapping and failed workers, invalid persisted learning,
parallel reinforcement, stale-result expiry, identity separation, shadow state
preservation, and zero-valued controls. The final combined regression run passed **2,573 tests, with 5 skips, 13
existing deprecation warnings, and 34 passing subtests**. The current Area 6
code pass is complete. Final mode-boundary tests also verify that shadow
GF/MB/LH predictions cannot be fed back into live CX/clock/motor readouts;
predictions remain available in observability traces.

Physical validation remains: peak RAM/latency with the real connectome and all
model services on the 8 GB Jetson, real sensor/audio/body feedback, and sustained
live interaction. Offline tests do not establish these measurements or biological
fidelity. The sensory and interface areas retain their own broader audit scopes.

## Settings Studio follow-up after area 6

- Expanded the existing Settings Studio with Aidon-inspired cards, serif heading,
  light/dark themes, search, paired bounded sliders, mobile layout, source labels,
  and dirty-field saves. All runtime YAML leaves are discovered automatically;
  the current catalog has 634 settings including 17 resource/context controls
  previously available only through environment variables.
- Added creator/admin authorization to both settings API methods. Credentials
  remain masked; explicitly edited YAML credentials migrate into encrypted
  `.env.age` storage instead of writing new plaintext credentials to YAML.
- Added serialized, staged batch saves with rollback, revision conflict checks,
  nested/list/multiline YAML edits, finite-number validation, and environment
  provenance so pending restarts do not falsely lock fields as overrides.
- Six audited per-request sampling controls apply live; other controls retain
  explicit restart requirements. Saves run in a worker thread so encryption and
  disk I/O do not block the WebUI event loop. See [Settings guide](SETTINGS.md).
- Browser checks passed for dirty-only saves, live-result feedback, search,
  light/dark rendering and 390px mobile width, without JavaScript errors.
- Final combined validation: `pytest -q tests/unit tests/integration
  cognition/conscience/test_conscience_gate.py --tb=short
  -o faulthandler_timeout=60` passed with **2,591 tests, 5 skips, and 34
  subtests**. Physical Jetson/live-model deployment verification remains pending.


## Area 7 progress

- Fixed local microphone and barge-in capture: PortAudio RawInputStream returns
  raw buffers, not arrays. Both paths now decode float32 little-endian PCM,
  validate chunk size/finiteness, and copy buffers before PortAudio reuses them.
- Normal capture and the barge monitor share microphone ownership. The monitor
  closes its input stream when capture starts, rather than merely pausing with
  the device still open. Reads have deadlines; failing devices back off instead
  of reopening in a tight CPU loop. Closing is attempted even when stop fails.
- Lazy voice startup now invokes the configured wake-word and speaker loaders,
  before warmup. An absent recognizer or warmup timeout no longer advertises a
  ready voice pipeline. Empty-model warmup also releases waiters.
- Serialize utterances using recurrent VAD/wake state. Async listen workers
  inherit the request identity, and speaker enrollment follows the current
  user and file revision without repeatedly rebuilding the ONNX extractor.
- Speaker verification exceptions produce a failed match instead of silently
  bypassing an enabled match gate. Transcription errors still restore idle UI
  state, and verification workers are joined on decoder errors too.
- One-shot ASR rejects invalid rates, non-mono/non-finite PCM and excessive
  duration before model loading; resampling uses actual sample positions.
  Cached voice boot failures are no longer retried on every phone request.
- Phone uploads use bounded reads and validate duration against the uploaded
  sample rate. A 48 kHz recording no longer loses its tail to a 16 kHz cap.
  Truncated WAV payloads are rejected before decoding.
- Non-finite voice prosody maps to zero contributions instead of being clamped
  into false urgency/energy. Removed a process-wide warning suppression from
  the listener; optional sounddevice imports retain local warning handling.
- Replaced stale transducer/mirror/precision references in the manual ASR
  diagnostic with current SenseVoice/Silero stages. Its offline config stage
  passes without torch, downloading models or requiring a microphone.
- This first batch does not complete area 7. Remaining review includes TTS
  cancellation and queue/resource bounds, WebUI/browser audio integration,
  wake-session edge cases, enrollment utility and physical Jetson measurements.
- Validation for this batch: 59 focused voice/sensory tests passed; the full
  unit/integration/conscience run passed with **2,619 tests, 5 skips, and 34
  subtests**. `python -m tests.unit.test_asr --stage env` also passed offline.
  Real microphone, speaker, wake-model accuracy, speaker-match quality and
  physical Jetson RAM/latency checks remain pending.
