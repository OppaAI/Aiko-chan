# Flybrain and conscience gating

Fly signals are functional priors. The per-identity NeuralState bus carries
MB valence, CX focus/urgency, DN vigor, and GF cancellation to their consumers.
Their live/shadow modes remain separate from the whole-connectome telemetry
mode; a shadow observation must not cancel output.

## GF: urgency versus cancellation

Previously, a single occurrence of `help`, `now`, `urgent`, `wait`, or `stop`
scored 0.7 against a hard-coded 0.65 cancellation line. That could abort a coding
request, mute speech, and defer scheduled work. High decaying CX urgency could
also re-arm cancellation even when the current turn's interrupt flag was false.

Cancellation now requires an explicit, whole-turn stop command or an actual
critical system/priority event. `MEMORY_FLYGF_CANCEL_THRESHOLD` defaults to 0.85
for those events; explicit STOP remains effective regardless of that threshold.
Motion, loudness, subliminal cues, and general urgency still guide attention,
but do not independently cancel work. A later ordinary turn clears the stop
flag even if its CX urgency trace remains high. STOP stays latched until a new
turn or an explicit clear; it does not expire behind the user's back.

Examples: `Please stop.` cancels; `Can you help me fix this code now?`,
`How can I stop this Python process?`, and `Don't stop coding.` do not.

The background fullbrain worker now consults the existing `AIKO_FULLBRAIN_EVERY_N`
and `AIKO_FULLBRAIN_MIN_INTERVAL_S` controls. Previously it bypassed them. Its motor-vigor readout also now reaches
the DN bus as a bounded live-mode modulation, instead of being overwritten
by the DN stage; shadow mode remains observational.

## Conscience: five levels, with a way to resolve false suspicion

- L0: content guardrails still run before routine classifications.
- L1: complete personal apologies and the ordinary activity-choice example
  have narrow, whole-turn classifications. An appended harmful request does
  not inherit them. Actual local read-only tool names now reach the existing
  allowlist. Writes, execution, and external side effects are not read-only.
- L2: each Laya evaluation returns an atomic snapshot of its bands/confidence.
  Failures clear all metadata; one user's negative reading must not leak into
  another request. Response evaluation describes Aiko's proposed reply rather
  than adjudicating the user's past conduct; speech evaluation includes the
  user turn as context.
- L3: confident, nonnegative deliberation can resolve a weak retrieval-only
  suspicion. Trigger-strength prohibitions, guardrail findings, unresolved
  doubt, and uncertain deliberation retain their existing treatment.
- L4: approvals for consequential actions remain. The change does not disable
  the conscience or globally lower all of its model thresholds.

Learned fly aversion still records tool votes. For explicitly enumerated
information-retrieval tools it is advisory, not a hard veto. Those tools still
pass the normal content/judge checks. Other external actions retain aversion
checks and their existing approval policy.

Chat approvals are scoped to one user and an exact normalized request. They
resolve uncertainty only after a fresh conscience check; a fresh hard refusal
still wins. Use a standalone `approve ccc-<id>` or `deny ccc-<id>` message.
Quoting or discussing those commands does not execute them. Covered retries
close their new ledger question without creating another human training label.
Expired continuation text must be sent again rather than executed automatically.

Learned behavior proposals also expire at confirmation. A standalone affirmation
applies to the proposal actually delivered to the user, and changes to a draft
outside the proposal cannot change what is confirmed. Invalid conditions and
nonfinite sensor values do not authorize an action.

"Forget all my memories" and ordinary statements about forgetting are not
instruction overrides. They still pass through conscience and any memory-tool
checks. Interaction preference resets clear related learning evidence and
known durable style rules, so forgotten preferences cannot reappear from those
paths. Negated reset commands preserve preferences. Both immediate and learned
tone feedback treat "less formal" and "informal" as casual tone.

Regression coverage includes the reported `Sorry I did something wrong.` and
outings-versus-coding conversation, appended harmful requests, explicit STOP,
urgency carryover, identity isolation, judge failures/concurrency, and bounded
fullbrain stepping. Model HTTP responses and connectome dynamics use fixtures
where needed; tests do not establish the quality of a deployed Laya checkpoint
or the availability of real connectome artifacts.

Restart Aiko after updating `dev`. Runtime environment variables override YAML;
check any service-level override of `MEMORY_FLYGF_CANCEL_THRESHOLD` if behavior
differs from these defaults.
