#!/usr/bin/env python3
"""Split laya_refusal_eval.jsonl into dev + locked test (Phase 0C).

Stratified by (refusal_category or 'benign') x checkpoint.
Cells with <5 rows stay in dev (too small to split safely).
Seed is fixed and recorded; the locked file's sha256 is printed for the record.
Twin co-location is best-effort: pairs are not explicitly marked in the data.
"""
import argparse
import json, hashlib, random, sys
from pathlib import Path
from collections import defaultdict

SEED = 20261005
TEST_FRAC = 0.2
SRC = Path(__file__).resolve().parent / "laya_refusal_eval.jsonl"

def main(write, force=False):
    with open(SRC, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs in split input")
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
        base = Path(SRC).resolve().parent
        dev_path, test_path = base / "laya_eval_dev.jsonl", base / "laya_eval_locked_test.jsonl"
        dev_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dev_rows)
        test_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in test_rows)
        if test_path.exists() and test_path.read_text(encoding="utf-8") != test_text and not force:
            raise ValueError("locked split differs; use --force only when deliberately rebuilding the frozen split")
        dev_path.write_text(dev_text, encoding="utf-8")
        test_path.write_text(test_text, encoding="utf-8")
        h = hashlib.sha256(test_text.encode()).hexdigest()
        print("locked test sha256:", h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--force", action="store_true", help="Deliberately replace a differing frozen split")
    args = parser.parse_args()
    main(write=args.write, force=args.force)
