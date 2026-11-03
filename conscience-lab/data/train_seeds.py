#!/usr/bin/env python3
"""Step 4: 3-seed ft2 baseline (noise floor for the 2x LR comparison).

Same config as the ft2 run in every respect except the RNG seed:
  task=phase3_refusal, data=phase3_train_v2.jsonl (frozen), profile=full,
  epochs=8, teacher=none, baseline LR (pristine rlcd.py), from base.
Seeds 20260923 (== ft2's own seed: reproduction check), 20260924, 20260925.

Usage (CWD must be conscience-lab/):
    env -u PYTHONPATH .venv-train/bin/python data/train_seeds.py
Checkpoints land in runs/phase3-seed-<seed>/.
"""
import sys
from pathlib import Path

BASE = "/home/oppa-ai/.cache/huggingface/hub/models--convaiinnovations--laya-multilingual/snapshots/1720e3e3357cfe1e281542e223f8273b0890ca34"
SEEDS = [20260923, 20260924, 20260925]


def main():
    from layaft.task import Task
    from layaft.train.pipeline import TrainPipeline

    task = Task.load("tasks/phase3_refusal.yaml")
    data = Path("data/phase3_train_v2.jsonl")
    assert data.is_file(), data
    for seed in SEEDS:
        out = Path(f"runs/phase3-seed-{seed}")
        if out.exists():
            print(f"SKIP seed {seed}: {out} exists", flush=True)
            continue
        print(f"=== seed {seed} -> {out} ===", flush=True)
        TrainPipeline(task, model=BASE, profile="full", epochs=8,
                      teacher="none", data=data, out=out, seed=seed).run()
    print("ALL SEEDS DONE")


if __name__ == "__main__":
    sys.exit(main())
