#!/usr/bin/env python3
"""Per-sample dump from a Laya server for FP-by-category analysis.

Runs every row in the eval set through the model, records predicted vs
expected per sample. Output feeds the per-category FP computation
(Claude asked for this) and FP clustering.

Usage:
    python3 dump_ft1_persample.py <eval.jsonl> <output.jsonl> [--url http://127.0.0.1:8093]

Uses the harness protocol (`laya serve` POST /v1/decide, Q1 triage + Q2
category) via ../harness/adapter.py — the same path as
`run_eval.py --backend laya --laya-mode two-question`. The old
/v1/systemone shape returned HTTP 422 on every row; do not go back to it.

NOTE: if you already have harness/results/<run>/per_case.jsonl for this
model, you don't need this script at all — per_case.jsonl already holds
per-sample predicted_refuse / predicted_category / triage_confidence.
Join it with the dev set for situation text (see extract_misses.py).
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "harness")))
from adapter import LayaDecideBackend, LayaTwoQuestionAdapter, load_categories


def query_one(adapter, backend, rng, situation: str) -> dict:
    """Query the Laya server via /v1/decide (two-question shape)."""
    pred = adapter.classify(backend, situation, rng)
    return {
        "predicted": "refuse" if pred["predicted_refuse"] else "allow",
        "category": pred["predicted_category"] or "none",
        "confidence": pred.get("triage_confidence"),
        "triage_choice": pred.get("triage_choice"),
        "category_confidence": pred.get("category_confidence"),
        "raw": pred.get("raw"),
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    eval_path, out_path = sys.argv[1], sys.argv[2]
    url = "http://127.0.0.1:8093"
    for i, a in enumerate(sys.argv):
        if a == "--url" and i + 1 < len(sys.argv):
            url = sys.argv[i + 1]

    with open(eval_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    here = os.path.dirname(os.path.abspath(__file__))
    cats = load_categories(os.path.join(here, "..", "categories.md"))
    adapter = LayaTwoQuestionAdapter(cats)
    backend = LayaDecideBackend(url)

    print(f"Dumping {len(rows)} rows via {url}/v1/decide ...")
    with open(out_path, "w") as out:
        for i, r in enumerate(rows):
            try:
                res = query_one(adapter, backend, random.Random(20261005 + i),
                                r["situation"])
            except Exception as e:
                res = {"predicted": "ERROR", "category": None,
                       "confidence": None, "error": str(e)[:200]}
            out.write(json.dumps({
                "id": r.get("id"),
                "refusal_category": r.get("refusal_category"),
                "checkpoint": r.get("checkpoint"),
                "expect": r.get("expect"),
                **res,
            }) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)}")

    # Quick summary: FP by *predicted* category (benign rows carry
    # refusal_category=null by design, so grouping by it is useless) and
    # by checkpoint. Mirrors the harness, which only reports recall by
    # category.
    fps, totals = {}, {}
    fp_by_pred, fp_by_ckpt = {}, {}
    with open(out_path) as f:
        for line in f:
            d = json.loads(line)
            if d["expect"] != "allow" or d["predicted"] == "ERROR":
                continue
            cat = "benign"
            totals[cat] = totals.get(cat, 0) + 1
            if d["predicted"] == "refuse":
                fps[cat] = fps.get(cat, 0) + 1
                pred = d.get("category") or "?"
                fp_by_pred[pred] = fp_by_pred.get(pred, 0) + 1
                ck = d.get("checkpoint") or "?"
                fp_by_ckpt[ck] = fp_by_ckpt.get(ck, 0) + 1
    print("\nBenign FP overall:")
    for cat in sorted(totals):
        fp = fps.get(cat, 0)
        print(f"  {cat}: {fp}/{totals[cat]} = {fp / totals[cat]:.1%}")
    print("\nFP by predicted (wrong) category:")
    for pred in sorted(fp_by_pred, key=lambda c: -fp_by_pred[c]):
        print(f"  {pred}: {fp_by_pred[pred]}")
    print("\nFP by checkpoint:")
    for ck in sorted(fp_by_ckpt):
        print(f"  {ck}: {fp_by_ckpt[ck]}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
