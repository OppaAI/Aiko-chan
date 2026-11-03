# Phase 3 training dataset — spec

## Phase 1 outcome (measured 2026-10-05, dev n=463)

Vanilla Laya multilingual q8 via `laya serve` `/v1/decide`, threshold mode:

| metric | measured | bar | verdict |
|---|---|---|---|
| harmful recall | 59.8% | ≥90% | fail |
| benign FP | 45.4% | <10% | fail |
| latency p50/p95 | 84/88 ms | — | fine |
| ECE (10-bin) | 0.140 | — | honest confidences |

Threshold sweep (0.05–0.95): recall≈FP at every cutoff — the confidence
signal does not separate harmful from benign. No threshold clears both
bars. This is a knowledge gap, not a calibration problem: **training is
required.**

## Misses by category (dev, n=16 each)

| category | recall | misses |
|---|---|---|
| self-harm | 37.5% | 10 |
| weapons | 37.5% | 10 |
| harassment | 43.8% | 9 |
| doxxing | 50.0% | 8 |
| csam | 56.2% | 7 |
| deception | 56.2% | 7 |
| degrading-speech | 56.2% | 7 |
| fraud | 56.2% | 7 |
| sexual-broad | 56.2% | 7 |
| animal-cruelty | 62.5% | 6 |
| defamation | 62.5% | 6 |
| theft-broad | 62.5% | 6 |
| idolatry | 68.8% | 5 |
| impersonation | 68.8% | 5 |
| credential-theft | 87.5% | 2 |
| cyberattack | 93.8% | 1 |
| **total** | | **103** |

Benign false positives: ~94 of 207 dev benign (45.4%).

## Design principles

1. **Narrow.** Target the measured gap; do not rebuild a general dataset.
2. **Train on dev, test on locked.** The 100-case locked test stays blind
   until Phase 5. Training on dev misses is legitimate.
3. **Benign contrast is half the dataset.** The FP rate (45.4%) is the
   worse failure. Harmful-only training will fix recall and keep the FP.
4. **Retention.** Keep what works: cyberattack/credential-theft are near
   the bar and must not regress.
5. **Egg tests stay.** Destruction-per-se (cracking eggs, cutting paper,
   demolishing one's own shed) must remain `none`.

## Composition (~270 rows)

| split | rows | source |
|---|---|---|
| misses | ~103 | all dev harmful misses, as-is (`extract_misses.py` kind=miss) |
| targeted variants | 52 | 10 fresh rephrasings each for self-harm, weapons, harassment, doxxing — same category, new wording (`eval/phase3_variants.jsonl`); plus dx-03 relabeled `none` (people-finder site ≠ doxxing content), 4 minimal-pair `none` twins for cue-dependent/borderline rows (sh-08, dx-05, dx-08, wp-04), and 8 extra output-draft rows (2 per weak category) for egress coverage. 47 refuse / 5 allow; checkpoints 32 input / 8 context / 12 output. |
| hard negatives | ~94 | all dev benign FPs, labeled `none` (kind=fp) |
| retention | 32 | 2 correctly-refused hits per category (kind=hit) |
| egg tests | included above | verify the egg-test benigns are in tn, not fp; if any egg test was flagged, it becomes a priority hard negative |

## Decision — none-labeling: two-question (c)

**Decided 2026-10-05 (Oppa).** The fine-tune uses two questions per
`/v1/decide` call, mirroring the old vertical/horizontal judge:

- **Q1 "triage"** (binary), criteria:
  - `violation`: "The text trips one of the refusal categories."
  - `none`: "Ordinary content that trips no refusal category."

  Trains on **all** rows: harmful → `violation`, benign (including the 94
  hard negatives) → `none`. Q1 *is* the binary refuse/allow metric, so the
  ≥90% / <10% bar is optimized directly.
- **Q2 "category"** (16-way): the 16 categories from `categories.md`.
  Trains on harmful rows only.
- **Serving rule — Q1 gates:** Q1=`none` → allow, Q2 not consulted.
  Q1=`violation` → refuse, category = Q2 argmax. Category accuracy is
  secondary per doctrine; the binary call is the metric.
- **Why not the others:** (a) leaves the hard negatives without a training
  target and leans on weak OOD-uncertainty — rejected; (b)+(d) needs a
  GGUF recompile — revisit only if (c) underperforms; per-option sigmoids
  need serving changes beyond Phase 3 scope.
- **Doctrine note:** two forwards in one HTTP call stretches the letter of
  "one semantic inference" but keeps its intent — the call stays in the
  typed-decision plane, no chat model, no chain.
- **Harness:** a two-question eval mode will be added to measure the
  fine-tuned model the way it will serve.

## Row format (two-question training rows — confirm serialization against v5 tooling)

```json
{"state": "<situation text>",
 "questions": {
   "triage": {"type": "choice",
     "criteria": {"violation": "The text trips one of the refusal categories.",
                  "none": "Ordinary content that trips no refusal category."}},
   "category": {"type": "choice",
     "criteria": {"weapons": "<definition from categories.md>", "...": "..."}}},
 "answers": {"triage": {"choice": "violation"},
             "category": {"choice": "weapons"}}}
```

Benign rows: `answers.triage.choice` = `none`, no Q2 answer. Q2 criteria
text comes from `categories.md` (single source of truth).

## Format risk (reviewer-raised)

The `situation` strings are descriptions ("Tool result (web fetch): a
forum thread where..."), not raw text. Train/eval/serve are consistent
with each other — the harness and the judge both consume this format —
**provided Aiko's runtime constructs situations the same way.** That is a
pipeline-consistency requirement, not a data bug: if the deployed
situation builder ever feeds raw forum text / untruncated tool output /
full drafts where training saw descriptions, the gap reopens. The 4
output-draft rows ending in "..." follow the dev-set truncation
convention; keep the convention aligned across train, eval, and runtime.

## Generation rules

- Variants are rephrasings, not copies: new verbs, new objects, same
  category and checkpoint mix (input/context/output).
- Every variant gets a second pair of eyes (Oppa or the harness's
  egg-test rule) before entering the set.
- No locked-test content: the locked 100 stay sealed; generate variants
  from dev misses only.
- Japanese rows stay out until the naturalness review lands (Phase 0D).
- **Label convention.** The none class is spelled `refusal_category: null`
  + `expect: allow` in eval-schema files (`phase3_variants.jsonl`, the dev
  set) and `label: "none"` in extraction-schema files
  (`phase3_candidates.jsonl`). The training builder MUST map both to one
  none class — never let `null`/`none`/`allow` split it across spellings.
- **Metadata is not input.** `why` and `tags` are review metadata. They
  name categories in plain language ("Primary: doxxing") — if the builder
  ever concatenates them with `situation`, the label leaks into the
  features. Model input is `situation` (+ criteria) only.

## Validation

1. Fine-tune on the Phase 3 set.
2. `screen_model.py --backend laya` (size + smoke).
3. `run_eval.py --backend laya` on dev: recall ≥90%, benFP <10%.
4. Sweep: confirm a threshold clears both bars with margin.
5. Locked test stays sealed until Phase 5.

## Tooling

- `harness/extract_misses.py` — pulls kind-labeled candidates
  (miss/hit/fp/tn) from a results dir + dev set.
- `harness/sweep_threshold.py` — picks the deployed threshold on dev.
