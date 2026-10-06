# Eval split (Phase 0C)

- Source: `laya_refusal_eval.jsonl` (563 cases, pre-split master — do not eval against this directly)
- Dev: `laya_eval_dev.jsonl` (463) — prompts, calibration, model selection
- Locked test: `laya_eval_locked_test.jsonl` (100) — **one run at the end** (Phase 5)

Method: stratified by (refusal_category or benign) x checkpoint, seed 20261005,
~20% to test. Cells with <5 rows stay in dev (all harmful context/output cells:
3 rows each). Consequence: the locked test's harmful cases are all `input`
checkpoint (48); benign covers all three checkpoints (52).

Script: `split_eval.py` (deterministic; re-running reproduces the split).
Twin co-location is best-effort — pairs are not explicitly marked in the data.

Locked test sha256:
376177d8dbe6341f9170f18b3849b5f04149ef2f2688bc2e52c73013214d9b3e
