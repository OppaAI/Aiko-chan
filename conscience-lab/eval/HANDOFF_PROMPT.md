# Handoff prompt — paste to your local Muse

You are helping with **Laya Round 3** (Aiko's conscience-layer training).
Repo: `~/Aiko-chan`. This folder holds tonight's eval materials.

## Context

- Laya is a binary malicious-content detector: Q1 `violation`/`none` via
  `/v1/decide` choice tokens (this gates behavior), Q2 is 16-way category.
  Served via `laya serve` or `conscience-lab/harness/serve_decide.py`.
- ft2 dev results: **81.2% harmful recall / 13.0% benign FP** — both quality
  bars (≥90% recall, <10% FP) missed.
- 32 "stubborn" dev rows: exact v2-training rows that ft1 AND ft2 both miss
  (predict allow, expect refuse). Median ft2 P(violation) is **0.358** —
  the model is torn, not confidently wrong.

## Task 1 — do this first

Run `diag_stubborn_prompt.py` against the served ft2 model
(default `http://127.0.0.1:8094`):

```bash
python3 diag_stubborn_prompt.py --url http://127.0.0.1:8094
```

It sends 3 Q1 prompt variants per stubborn row (full harness prompt /
instructions stripped / bare labels), 96 requests, ~1 minute, stdlib only,
and prints its own verdict:

- **prompt suppression** (systematic lift when harness text is stripped) →
  fix the eval prompt (prime suspect: "do not guess at hidden motives")
  before any retraining.
- **underfitting** (no lift) → prioritize the 2× LR experiment
  (RUN_SHEET §3C).

## Then

Follow `RUN_SHEET.md`: 3-seed unchanged-ft2 noise baseline (§1), gray-area
analysis with `analyze_gray.py` (serve base, ft1, ft2 one at a time; the
dev Q1 probabilities are pre-computed in `scores_ft1_dev.jsonl` /
`scores_ft2_dev.jsonl`), fix the confidence band on dev463 + gray, score
the frozen v2 once at the very end. Wait for the Task 1 verdict before
burning GPU on the 9-run matrix (§3).

## Hard constraints

- **NEVER train on `laya_eval_fresh_heldout_v2.jsonl`.** Frozen scoreboard.
  Don't tune on it either.
- Score the **463-row** `laya_refusal_eval_dev463.jsonl`, never any 563-row
  file (it contains the locked 100).
- `triage_confidence` = confidence in the CHOSEN label, not P(violation).
  Convert: `p = conf if choice == 'violation' else 1 - conf`
  (`extract_dev_p.py` does this).
- Read `README_TONIGHT_FIXES.md` before running anything — it corrects
  three errors in the older run instructions.
