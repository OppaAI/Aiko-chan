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
| targeted variants | 40 | 10 fresh rephrasings each for self-harm, weapons, harassment, doxxing — same category, new wording |
| hard negatives | ~94 | all dev benign FPs, labeled `none` (kind=fp) |
| retention | 32 | 2 correctly-refused hits per category (kind=hit) |
| egg tests | included above | verify the egg-test benigns are in tn, not fp; if any egg test was flagged, it becomes a priority hard negative |

## Row format (proposed — confirm against v5 training tooling)

```json
{"state": "<situation text>",
 "question": "refusal",
 "criteria": {"weapons": "<definition from categories.md>", ...},
 "answer": "weapons"}
```

Criteria text comes from `categories.md` (single source of truth); the
harness already loads it. Benign rows carry `"answer": "none"`.

**Open question — none-labeling vs the 16-opt cap.** Serving caps at 16
options (`laya.max_opts` is compiled into the GGUF), so `none` cannot
ride along as a 17th criterion at serve time. Candidate resolutions:

- (a) Train 16-way; benign rows teach low-confidence-everywhere and the
  deployed threshold reads it (matches serving exactly).
- (b) Train 17-way (16+none); serve 16+threshold. Train/serve mismatch,
  but the none-concept is learned explicitly.
- (c) Two questions per call (binary any-category/none + 16-way which),
  mirroring the old vertical/horizontal judge. Fits the cap; needs a
  disagreement rule.
- (d) Recompile the GGUF with max_opts=17+ and serve explicit none.

Recommendation: (a) for the first Phase 3 run — smallest change, serving
already works that way. Revisit (d) if the fine-tune rebuilds the
artifact anyway.

## Generation rules

- Variants are rephrasings, not copies: new verbs, new objects, same
  category and checkpoint mix (input/context/output).
- Every variant gets a second pair of eyes (Oppa or the harness's
  egg-test rule) before entering the set.
- No locked-test content: the locked 100 stay sealed; generate variants
  from dev misses only.
- Japanese rows stay out until the naturalness review lands (Phase 0D).

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
