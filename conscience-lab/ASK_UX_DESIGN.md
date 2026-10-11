# Ask-UX design: what "ask the owner" does in the turn pipeline

Design draft for the approval band's consumer side. Builds ONLY on
machinery that already exists — no new ledger, no new commands.

## Existing pieces (no changes)

- Escalation ledger: `core.py::resolve_escalation(id, approved, note)`,
  `pending()`, states HITL_APPROVED / HITL_DENIED.
- Owner commands: `approve ccc-<id>` / `deny ccc-<id>` (hooks.py),
  resume/discard flows already wired.
- Gate outcome ASK: `gate.py` already returns
  `GateOutcome(action=ASK, reason="conscience escalated")` for
  ESCALATE verdicts — the turn pipeline knows how to pause on ASK.
- Band signal: `laya_judge.last_band_decision == "ask"` (+ `last_p_violation`,
  `last_uncertain=True`) from the triage wiring.

## Proposed flow (triage ask in chat)

1. `_score_triage` returns (0, 0) + uncertain (already merged).
2. `decide()` check #3 → ESCALATE (already merged).
3. Gate → ASK with reason carrying the situation + P(violation) +
   suggested category for owner context (NEW: include all three; today
   the reason carries the verdict only).
4. Turn pauses; owner is notified through the existing notice path
   (same bus as tool-approval notices — one inbox, not two).
5. Owner `approve ccc-<id>` → turn proceeds as allow (log both the
   approval and the P value for later band calibration).
   Owner `deny ccc-<id>` → turn refuses with the standard refusal
   template. Log likewise.
6. Timeout with no owner response → fail CLOSED to refuse (matches the
   existing default-closed invariant; needs an explicit timeout value —
   suggest 10 minutes, env-tunable).

## Non-goals / explicit exclusions

- CSAM/self-harm never enter this flow (hard refuse, policy).
- No second approval queue: reuse the escalation ledger and the
  approve/deny commands verbatim.
- Social (Threads/Bluesky/Mastodon) replies: band-ask there means
  SKIP (don't publish, log for review) — there is no owner online to
  ask within a poll cycle. Owner review happens via the ledger's
  pending list, async.

## Open knobs (env, defaults suggested)

- `ASK_TIMEOUT_S` = 600 (fail-closed refuse on expiry).
- Band thresholds stay [0.40, 0.60] until a frozen re-measurement says
  otherwise (dev-fit caps do not hold off-distribution — budget for
  ~10–16% ask rates).

## Integration checklist

- [ ] Gate ASK reason carries situation + P + category.
- [ ] Notice bus covers band-ask (same inbox as tool approvals).
- [ ] Approve/deny handlers accept band escalation ids (same
        `ccc-<id>` namespace — no new id space).
- [ ] Timeout path defaults to refuse + logs.
- [ ] Social pollers map ask → skip+log (no publish, no owner wait).
- [ ] Approval/denial + P logged per decision (feeds band recalibration).
