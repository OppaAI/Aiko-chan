# Laya Round 3 — Run Sheet (2026-10-07)

> 2026-10-07: v2 frozen (Claude's revision) — 400 rows, 200/200, decontam clean locally (max 0.591 vs dev). v1 archived in v1_superseded/. Use the v2 file below.
Goal: decide whether ft2's problems are noise, underfitting (LR), or hard-negative
overcorrection — measured on the frozen fresh held-out set, not just dev.

## 0. Stage the frozen set

- Copy `laya_eval_fresh_heldout_v2.jsonl` (this dir) → `conscience-lab/eval/laya_eval_fresh_heldout_v2.jsonl`
  (or point `run_eval.py --dev` at it directly).
- Verify: 400 rows, 200 `refuse` / 200 `allow`, ids `fh-*` / `fb-*`.
- **NEVER train on it.** It is the scoreboard now.

## 1. Noise baseline — do first (~10 min)

Rerun the **unchanged** ft2 config (v2 data, 8 epochs, best-val checkpoint) with **3 different seeds**.
Score each run on **dev + fresh** with the harness two-question adapter.

Record per seed: of the 18 ft2-fixed and 16 ft2-broken rows (vs ft1), how many reproduce.
This is the noise floor — every later result is judged against it.

Deriving the row lists from per_case files (dev harmful rows only):
- stubborn 32: `ft1.predicted_refuse == false AND ft2.predicted_refuse == false`
- new misses 16: `ft1 == true AND ft2 == false` (9 of these are the regressions of interest)

## 2. Diagnostic checks

a. **Training loss on the 32 stubborn rows.** Pull per-row loss from
   `runs/phase3-refusal-1k/` (or re-run training with per-row loss logging).
   - Loss ≈ 0 but eval says "none" → train/eval pipeline mismatch. Diff the
     training input format against what `harness/serve_decide.py` sends in its
     `/v1/decide` calls (fields, instructions, criteria shape).
   - Loss high → genuine underfitting → the LR experiment (§3C) is the fix.
b. ~~Near-duplicate opposite-label scan~~ — done remotely: only 4 pairs ≥ 0.6
   Jaccard in all of v2 training, all four intentional twin pairs. No hidden
   label conflicts.

## 3. Experiment matrix — one change per run, 3 seeds each (~27 min)

- **A. Baseline** = the §1 reruns. Reuse them.
- **B. Narrow ablation.** v2 data MINUS these credential-themed hard negatives only:
  - `b-breach-own-email` (fp-mined) — "check whether my email appeared in a breach"
  - `b-password-audit-own` (fp-mined) — "check whether any of my passwords are weak"
  - `p3r2-v2-hn-04` (hn-written) — "forgot my own router's admin password… factory reset"
  
  Keep the twin allow-halves (`p3r2-v2-twin-ct-*`) and all r1 TNs in place.
  Count BOTH the misses that come back AND the FPs that come back.
  (Wider variant if B is inconclusive: also drop the twin ct allow-halves.)

- **C. Higher LR.** `layaft` hardcodes encoder LR 2.5e-5 / head LR 1e-4 — monkeypatch
  to 2× and retrain v2 unchanged. (ft1 profile was full-model, `freeze 0`, so LoRA
  rank is not a lever here; LR is.)

Score every run on **dev AND fresh** via the two-question adapter.

## 4. Reading the results

- C fits the 32 stubborn → underfitting confirmed; LR was the binding lever.
- B brings back the 9 regressions without returning FPs → overcorrection confirmed;
  the mined FPs were too aggressive in credential-theft territory.
- Anything smaller than the §1 seed-noise floor is noise, not signal.
- Report dev and fresh numbers separately. Dev numbers are fit-check only —
  never present them as held-out results.

## Explicitly not doing

- No new training rows until fresh-set scores exist (no more mining dev).
- Locked 100 (`laya_eval_locked_test.jsonl`) stays untouched until Phase 5.
