#!/usr/bin/env python3
"""Extract Phase 3 training candidates from a laya eval run.

Joins the run's per_case.jsonl with the dev set (for situation text)
and labels every case with its CORRECT label:

    python3 extract_misses.py --results results/laya-q8/ \\
        --dev ../eval/laya_eval_dev.jsonl --out ../phase3_candidates.jsonl

Output rows:
    {"id", "kind", "label", "situation", "checkpoint",
     "model_choice", "model_confidence"}

kinds: miss (harmful, model allowed), hit (harmful, model refused),
       fp (benign, model refused), tn (benign, model allowed).
label is the refusal_category for harmful rows, "none" for benign.
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = {}
    for line in open(args.dev, encoding="utf-8"):
        r = json.loads(line)
        dev[r["id"]] = r

    out_rows = []
    counts: dict[str, int] = {}
    for line in open(os.path.join(args.results, "per_case.jsonl"),
                     encoding="utf-8"):
        r = json.loads(line)
        d = dev.get(r["id"], {})
        harmful = r["expect_refuse"]
        refused = r["predicted_refuse"]
        if harmful and not refused:
            kind = "miss"
        elif harmful and refused:
            kind = "hit"
        elif not harmful and refused:
            kind = "fp"
        else:
            kind = "tn"
        label = r["refusal_category"] if harmful else "none"
        key = f"{kind}:{label}"
        counts[key] = counts.get(key, 0) + 1
        out_rows.append({
            "id": r["id"],
            "kind": kind,
            "label": label,
            "situation": d.get("situation", ""),
            "checkpoint": r["checkpoint"],
            "model_choice": r.get("choice"),
            "model_confidence": r.get("choice_confidence"),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(out_rows)} candidate rows -> {args.out}")
    print("\nkind:label counts:")
    for k in sorted(counts):
        print(f"  {k:28s} {counts[k]}")


if __name__ == "__main__":
    main()
