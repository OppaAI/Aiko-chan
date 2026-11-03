#!/usr/bin/env python3
"""Split laya_refusal_eval.jsonl into dev + locked test (Phase 0C).

Stratified by (refusal_category or 'benign') x checkpoint.
Cells with <5 rows stay in dev (too small to split safely).
Seed is fixed and recorded; the locked file's sha256 is printed for the record.
Twin co-location is best-effort: pairs are not explicitly marked in the data.
"""
import json, hashlib, random, sys
from collections import defaultdict

SEED = 20261005
TEST_FRAC = 0.2
SRC = "laya_refusal_eval.jsonl"

def main(write):
    rows = [json.loads(l) for l in open(SRC)]
    cells = defaultdict(list)
    for i, r in enumerate(rows):
        cells[(r["refusal_category"] or "benign", r["checkpoint"])].append(i)

    rnd = random.Random(SEED)
    dev_idx, test_idx = [], []
    print(f"{'cell':28s} {'n':>3s} {'dev':>4s} {'test':>4s}")
    for key in sorted(cells):
        idxs = cells[key]
        rnd.shuffle(idxs)
        n = len(idxs)
        n_test = round(TEST_FRAC * n) if n >= 5 else 0
        if n >= 5 and n_test == 0:
            n_test = 1
        test_idx.extend(idxs[:n_test])
        dev_idx.extend(idxs[n_test:])
        print(f"{str(key):28s} {n:>3d} {n-n_test:>4d} {n_test:>4d}")

    dev_rows = [rows[i] for i in dev_idx]
    test_rows = [rows[i] for i in test_idx]
    # sanity: no id overlap, totals add up
    assert len({r["id"] for r in dev_rows} & {r["id"] for r in test_rows}) == 0
    assert len(dev_rows) + len(test_rows) == len(rows)
    print(f"\ndev={len(dev_rows)} test={len(test_rows)} total={len(rows)}")

    if write:
        with open("laya_eval_dev.jsonl", "w") as f:
            for r in dev_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open("laya_eval_locked_test.jsonl", "w") as f:
            for r in test_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = hashlib.sha256(open("laya_eval_locked_test.jsonl", "rb").read()).hexdigest()
        print("locked test sha256:", h)

if __name__ == "__main__":
    main(write="--write" in sys.argv)
