#!/usr/bin/env python3
"""Step 5: credential ablation — v2 minus 3 topical hard negatives.

Same recipe as step 4 (profile=full, epochs=8, teacher=none, baseline LR,
from base), seeds 20260923/24/25, but trains on
data/phase3_train_v2_ablation.jsonl (525 rows: v2 without
b-breach-own-email, b-password-audit-own, p3r2-v2-hn-04).

If credential-theft recall recovers vs the step-4 seeds, those rows were
poisoning the category; if not, the problem is elsewhere.

Usage (CWD must be conscience-lab/):
    env -u PYTHONPATH .venv-train/bin/python data/train_ablation.py
Checkpoints land in runs/phase3-ablation-<seed>/.
"""
import sys
from pathlib import Path

BASE = "/home/oppa-ai/.cache/huggingface/hub/models--convaiinnovations--laya-multilingual/snapshots/1720e3e3357cfe1e281542e223f8273b0890ca34"
SEEDS = [20260923, 20260924, 20260925]


def main():
    from layaft.task import Task
    from layaft.train.pipeline import TrainPipeline

    task = Task.load("tasks/phase3_refusal.yaml")
    data = Path("data/phase3_train_v2_ablation.jsonl")
    assert data.is_file(), data
    rows = data.read_text().strip().splitlines()
    assert len(rows) == 525, len(rows)
    dropped = {"b-breach-own-email", "b-password-audit-own", "p3r2-v2-hn-04"}
    import json
    assert not (dropped & {json.loads(r)["id"] for r in rows}), "drop rows still present"
    print("ablation set verified: 525 rows, drop rows absent", flush=True)
    for seed in SEEDS:
        out = Path(f"runs/phase3-ablation-{seed}")
        if out.exists():
            print(f"SKIP seed {seed}: {out} exists", flush=True)
            continue
        print(f"=== ablation seed {seed} -> {out} ===", flush=True)
        TrainPipeline(task, model=BASE, profile="full", epochs=8,
                      teacher="none", data=data, out=out, seed=seed).run()
    print("ABLATION DONE")


if __name__ == "__main__":
    sys.exit(main())
