#!/usr/bin/env python3
"""Near-duplicate check: new training rows vs the eval set.

Catches rows that are too close to laya_refusal_eval.jsonl — a near-dup
in train that also appears in eval inflates the score without proving
generalization.

Usage:
    python3 check_near_dupes.py <new_rows.jsonl> <eval.jsonl> [--threshold 0.85]

Outputs any new row whose situation is above the similarity threshold
to any eval row, with the matching eval id.
"""

import json
import sys
from difflib import SequenceMatcher


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    new_path, eval_path = sys.argv[1], sys.argv[2]
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--threshold" else 0.85

    with open(new_path) as f:
        new_rows = [json.loads(l) for l in f if l.strip()]
    with open(eval_path) as f:
        eval_rows = [json.loads(l) for l in f if l.strip()]

    print(f"Checking {len(new_rows)} new rows vs {len(eval_rows)} eval rows (threshold {threshold})")
    hits = 0
    for nr in new_rows:
        nsit = nr.get("situation", "")
        for er in eval_rows:
            sim = similarity(nsit, er.get("situation", ""))
            if sim >= threshold:
                hits += 1
                print(f"\nHIT sim={sim:.2f}")
                print(f"  new:  [{nr.get('id')}] {nsit[:100]}")
                print(f"  eval: [{er.get('id')}] {er.get('situation', '')[:100]}")
                break
    print(f"\n{hits} near-duplicates found.")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
