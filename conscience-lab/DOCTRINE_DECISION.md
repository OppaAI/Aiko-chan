# Doctrine decision: replace vs alongside for the triage verdict

Decision needed from doctrine owners. Facts both options share:

- Champion (ft3-2xlr) speaks triage (binary, ECE ≤0.09, generalizes) +
  category (16-way, 61% dev → 38% locked, does not generalize).
- Production speaks (v,h): vertical/horizontal through blend → ladder →
  decide, plus lexical judge, canon filter, autonomy gate.
- Measured band (triage P in [0.40, 0.60] → ask) catches 34–51% of errors
  at ~5–16% ask rates depending on distribution.
- The wiring already merged is additive: `CCC_JUDGE_SCHEME=triage` asks
  the champion triage+category and maps verdicts onto the existing
  (v,h) tuple contract (refuse→(-1,-1), allow→(+1,+1), ask→(0,0) +
  uncertain). Nothing downstream changed.

## Option A — replace

Triage verdict drives refusal outright; blend/ladder/decide retire for
this judge.

- For: single source of truth; the trained binary head beats the
  doctrine axes on every measured set; simpler pipeline, one threshold
  to own.
- Against: discards the commandments/neighbour semantics the rest of
  the conscience (canon, autonomy, ledger language) is written in;
  loses the lexical veto interplay; category signal (weak but
  non-zero) thrown away with the pipeline; largest blast radius —
  every downstream consumer re-verified.

## Option B — alongside with triage-first precedence (recommended)

Triage refuse → refuse. Triage ask → escalate (owner). Triage allow →
fall through to the existing (v,h) pipeline unchanged.

- For: keeps the doctrine investment; adds trained competence exactly
  where it's measured strongest (binary head) and leans on doctrine
  where the model is weak (category nuance, novel territory); the
  already-merged wiring implements exactly this mapping; reversible
  per-row by moving thresholds; fail-safe layering (either side can
  still refuse).
- Against: two judges to maintain; one precedence rule to own and
  document; one extra model call per turn (already paid — triage is
  asked anyway; the cost is the band logic, which is arithmetic).

## Option C — (v,h) primary, triage advisory

Not recommended: pays the full champion cost while ignoring its only
measured advantage. Listed for completeness.

## Recommendation

B. It is also the only option consistent with what's already merged:
the additive wiring maps triage onto (v,h) precisely so blend/ladder
keep working. A means: flip the default scheme to triage and keep
(v,h) as the fall-through. Revisit A if the category head ever
generalizes (it currently degrades 61% → 38% off-distribution).

## Decision

**B — alongside, triage-first** (decided 2026-10-10, provisional: "try B first").

B is what is already merged and running (`CCC_JUDGE_SCHEME=triage` maps
triage verdicts onto the (v,h) contract: refuse→refuse, ask→escalate,
allow→fall through to the doctrine pipeline), so this decision changes
zero code. Revisit A (replace) if the category head ever generalizes
off-distribution.
