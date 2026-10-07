#!/usr/bin/env python3
"""Assemble Laya Phase 3 Round 2 training dataset (v2).

Combines:
  - v3 variant pairs (88 rows): targeted at weakest categories
  - v3 hard negatives (25 rows): FP reduction
  - v1 draft (56 rows): retention, prevents forgetting r1 gains

Outputs a single JSONL ready for training. Strips the `why` field
(never feed it to the model) and verifies pair integrity.

Usage:
    python3 assemble_r2_dataset.py <output.jsonl>
"""

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent


def load(name):
    with open(DATA_DIR / name) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "laya_phase3_r2_train.jsonl"

    variants = load("laya_phase3_round2_variants_v3.jsonl")
    hns = load("laya_phase3_round2_hard_negatives_v3.jsonl")
    v1 = load("laya_phase3_round2_variants_draft.jsonl")

    all_rows = variants + hns + v1
    print(f"Variants v3: {len(variants)}")
    print(f"Hard negatives v3: {len(hns)}")
    print(f"v1 retention: {len(v1)}")
    print(f"Total: {len(all_rows)}")

    # Strip `why` — never feed annotator reasoning to the model
    stripped = 0
    for r in all_rows:
        if "why" in r:
            del r["why"]
            stripped += 1
    print(f"Stripped `why` from {stripped} rows")

    # Verify pair integrity: every twin has its refuse row
    ids = {r["id"] for r in all_rows}
    orphans = [r["id"] for r in all_rows
               if "twin-" in r["id"]
               and r["id"].replace("twin-", "") not in ids]
    if orphans:
        print(f"WARNING: {len(orphans)} orphan twins: {orphans[:5]}")
    else:
        print("Pair integrity: OK")

    # Check label balance
    refuse = sum(1 for r in all_rows if r.get("expect") == "refuse")
    allow = sum(1 for r in all_rows if r.get("expect") == "allow")
    print(f"Label balance: {refuse} refuse / {allow} allow")

    # Check ID uniqueness
    assert len(ids) == len(all_rows), "DUPLICATE IDS!"
    print("ID uniqueness: OK")

    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
