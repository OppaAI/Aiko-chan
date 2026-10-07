#!/usr/bin/env python3
"""Assemble Laya Phase 3 Round 2 training dataset with Q1/Q2 two-question format.

Combines:
  - v3 variant pairs (88 rows): targeted at weakest categories (csam, doxxing, self-harm)
  - v3 hard negatives (25 rows): FP reduction, length-matched to variants
  - v1 draft (56 rows): retention, prevents forgetting round 1 gains

Output format (one JSONL row per sample):
  {
    "id": "...",
    "situation": "...",           # model input — never includes `why`
    "triage": "violation|none",   # Q1: trained on EVERY row
    "category": "doxxing|..."     # Q2: only present on harmful rows
  }

Serving rule: Q1 gates. If triage=none, Q2 is ignored.

Pair integrity: minimal-pair twins are kept adjacent in the output so any
split can keep them together. A --split option is provided for train/dev.

Usage:
    python3 assemble_r2_train.py <output.jsonl> [--split 0.9]
"""

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent

# 16 refusal categories for Q2 (must match LayaTwoQuestionAdapter._category_criteria)
CATEGORIES = [
    "weapons", "cyberattack", "fraud", "theft",
    "harassment", "csam", "doxxing", "self-harm",
    "deception", "defamation", "idolatry", "impersonation",
    "animal-cruelty", "degrading-speech", "violence",
    "none",
]


def load(name):
    with open(DATA_DIR / name) as f:
        return [json.loads(l) for l in f if l.strip()]


def to_train_row(r):
    """Convert a v3 annotation row to Q1/Q2 training format."""
    expect = r.get("expect", "")
    is_harmful = (expect == "refuse")

    out = {
        "id": r["id"],
        "situation": r["situation"],
        "triage": "violation" if is_harmful else "none",
    }
    # Q2 only on harmful rows
    if is_harmful:
        cat = r.get("refusal_category", "none")
        if cat not in CATEGORIES:
            print(f"WARNING: unknown category '{cat}' in {r['id']}, mapping to none")
            cat = "none"
        out["category"] = cat
    return out


def pair_key(r):
    """Sort key that keeps minimal-pair twins adjacent."""
    rid = r["id"]
    # Strip twin- prefix so twins sort next to their refuse row
    base = rid.replace("twin-", "")
    # Twins sort right after their base (0=base, 1=twin)
    is_twin = 0 if "twin-" not in rid else 1
    return (base, is_twin)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path = args[0] if args else "laya_phase3_r2_train.jsonl"
    split = 0.9
    for a in sys.argv[1:]:
        if a.startswith("--split"):
            split = float(a.split("=")[1])

    variants = load("laya_phase3_round2_variants_v3.jsonl")
    hns = load("laya_phase3_round2_hard_negatives_v3.jsonl")
    v1 = load("laya_phase3_round2_variants_draft.jsonl")

    print(f"Variants v3: {len(variants)}")
    print(f"Hard negatives v3: {len(hns)}")
    print(f"v1 retention: {len(v1)}")

    all_rows = variants + hns + v1
    # Sort so minimal-pair twins stay adjacent
    all_rows.sort(key=pair_key)

    # Verify pair integrity
    ids = {r["id"] for r in all_rows}
    orphans = [r["id"] for r in all_rows
               if "twin-" in r["id"]
               and r["id"].replace("twin-", "") not in ids]
    if orphans:
        print(f"WARNING: {len(orphans)} orphan twins: {orphans[:5]}")
    else:
        print("Pair integrity: OK")

    # Convert to Q1/Q2 format (strips `why` — never feed to model)
    train_rows = [to_train_row(r) for r in all_rows]

    # Label balance
    n_violation = sum(1 for r in train_rows if r["triage"] == "violation")
    n_none = len(train_rows) - n_violation
    n_q2 = sum(1 for r in train_rows if "category" in r)
    print(f"Total: {len(train_rows)} (violation={n_violation}, none={n_none}, Q2 rows={n_q2})")

    # Q2 category distribution
    from collections import Counter
    cat_dist = Counter(r["category"] for r in train_rows if "category" in r)
    print("Q2 distribution:")
    for cat in CATEGORIES:
        if cat_dist[cat]:
            print(f"  {cat}: {cat_dist[cat]}")

    # Optional split (keeps pairs together — split on pair base, not individual rows)
    if split < 1.0:
        # Group by pair base
        pairs = {}
        for r in train_rows:
            base = r["id"].replace("twin-", "")
            pairs.setdefault(base, []).append(r)
        bases = sorted(pairs.keys())
        n_train = int(len(bases) * split)
        train_bases = set(bases[:n_train])
        train_out = [r for r in train_rows if r["id"].replace("twin-", "") in train_bases]
        dev_out = [r for r in train_rows if r["id"].replace("twin-", "") not in train_bases]
        print(f"Split {split}: train={len(train_out)}, dev={len(dev_out)} (pairs kept together)")
        with open(out_path, "w") as f:
            for r in train_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dev_path = out_path.replace(".jsonl", ".dev.jsonl")
        with open(dev_path, "w") as f:
            for r in dev_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path} + {dev_path}")
    else:
        with open(out_path, "w") as f:
            for r in train_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
