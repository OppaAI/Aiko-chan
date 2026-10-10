# Approval band — ship doc

Triage-confidence routing for the ft3-2xlr champion: P(violation) inside
[0.40, 0.60] routes to owner approval instead of binary refuse/allow.

## Policy

`harness/adapter.py::apply_band(p, lo=0.40, hi=0.60)` → refuse / ask / allow.
`p=None` (older servers) degrades to `ask-needs-probs`, never to `ask` —
a missing signal must not nag the owner. The harness records `suggested`
per case (additive field; binary metrics unchanged).

## Measured behavior (band fixed on dev, verified frozen + locked)

| set    | ask rate | errors caught | benign asked | harmful asked |
|--------|----------|---------------|--------------|---------------|
| dev    | 7.8%     | 18/35 (51%)   | 4.8%         | 10.2%         |
| frozen | 10.2%    | 24/70 (34%)   | 10.5%        | 10.0%         |
| locked | 16.0%    | 7/19 (37%)    | 15.4%        | 16.7%         |

The caps (5% benign / 15% harmful) were fit on dev and do NOT hold on
unseen sets — benign-asked runs 2–3× over off-distribution. Size owner
interrupt budget for ~10–16% ask rates, not the dev numbers. The band
catches a third to a half of errors everywhere; it is a safety net for
coin-flips, not a gray solver (only ~24% of gray rows land in-band).

## Consumer contract (Aiko side)

1. Ask the checkpoint the questions it was TRAINED on (triage +
   category, phase-3 shape). A 2x2/3x2/legacy scheme against this
   checkpoint is the documented misfire mode — band thresholds are
   meaningless on any other question shape.
2. Read P(violation) from the triage `probabilities` dict, never from
   argmax choice or `confidence` (entropy semantics differ by type).
3. Band on triage ONLY. Q2 exact-match degrades off-distribution
   (61% dev → 38% locked); a category-gated band inherits that.
4. `ask` must escalate to the owner with the situation attached — never
   fail open to allow, never fail closed to refuse silently.
5. CSAM/self-harm rows bypass the band (refuse path); the gray set
   deliberately excludes them and so must the policy.

## Provenance

Thresholds from `eval/analyze_gray.py` on ft3-2xlr dev+gray
(`eval/scores_ft3_dev_gray.jsonl`, AUROC 0.682). Frozen/locked
re-scores added `p_violation` only — no model selection or tuning
touched either set after the single sanctioned runs.
