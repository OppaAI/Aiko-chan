#!/usr/bin/env python3
"""Add the missing benign population from ETHICS commonsense.

Measured problem: v5 detects clear wrongness at 0.582 but scores 0.592 on
"not clearly wrong" -- and the untuned base scores 0.744 there. The fine-tune
bought moral sensitivity by giving up ordinary-action coverage. Every dataset we
trained on was assembled to collect *moral judgements*, so benign actions were
never sampled. Our own data has no such population either.

ETHICS commonsense supplies it: ~1,100 short first-person scenarios labelled
"not clearly wrong", i.e. everyday unremarkable acts.

LABELS ARE INFERRED, NOT AUTHORED -- this is the important caveat. ETHICS gives
one binary axis; our schema needs two. We therefore only claim what the data
supports and let the rest fall to `unknown`:

    label 1 "clearly wrong"      -> vertical `contrary`
    label 0 "not clearly wrong"  -> vertical `unknown`   (NOT `aligned`:
                                     absence of clear wrongness is not
                                     evidence of virtue)

The horizontal band is set by a fixed, documented rule rather than per-item
judgement, because per-item judgement at this scale is exactly what destroyed
v6 (CE stuck at 1.06 -- unlearnable label noise). The rule is deliberately
biased toward `unknown`, which is the safe direction: an unstated neighbour
impact should escalate, not greenlight.

These rows are weaker evidence than Oppa's own cases, which is precisely why
the 40-60 hand-authored benign cases are still needed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
V_BANDS = ("aligned", "contrary", "unknown")
H_BANDS = ("benefits", "harms", "no-effect", "uncertain")

# Fixed mapping. (vertical, horizontal) -> share of that ETHICS label's rows.
# 4-band horizontal. ETHICS label 0 is "not clearly wrong", which is exactly
# `no-effect`: nobody is materially helped or harmed, so it proceeds. Mapping it
# to `uncertain` instead would escalate a third of all traffic for no safety
# gain. Label 1 is "clearly wrong" -> `harms`.
RULE = {
    ("contrary", "harms"): 1.00,
}


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def assign_h(vertical: str, rng: random.Random) -> str:
    options = [h for (v, h) in RULE if v == vertical]
    weights = [RULE[(vertical, h)] for h in options]
    return rng.choices(options, weights=weights, k=1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/conscience_training_v8_fixed.jsonl")
    ap.add_argument("--ethics", default="/tmp/opencode/ethics/commonsense_train.csv")
    ap.add_argument("--n-per-label", type=int, default=320)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="data/conscience_training_v9.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = load(HERE / args.base)
    before_n = len(base)

    rows = [r for r in csv.DictReader(open(args.ethics, newline="", encoding="utf-8"))
            if r.get("is_short") == "True"]
    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r["input"].strip())
    for k in by_label:
        by_label[k] = sorted(set(by_label[k]))
    print(f"ETHICS short scenarios available: " +
          ", ".join(f"label {k}={len(v)}" for k, v in sorted(by_label.items())))

    existing = "\n".join(r["fields"]["scenario"] for r in base)
    added: list[dict] = []
    for label, texts in sorted(by_label.items()):
        pool = [t for t in texts if t and t not in existing]
        rng.shuffle(pool)
        take = pool[:args.n_per_label]
        vertical = "contrary" if label == "1" else "unknown"
        horizontal = "harms" if label == "1" else "no-effect"
        for text in take:
            added.append({"fields": {"scenario": text},
                          "answers": {"vertical": vertical,
                                      "horizontal": horizontal}})
        print(f"  label {label} -> vertical={vertical}: {len(take)} added")

    # Hold both marginals near-uniform. Every ETHICS row is vertical
    # contrary|unknown, so appending them naively dilutes `aligned` to ~22%.
    # Fill the deficit by resampling OUR OWN rows, which keeps the new
    # information and restores the balance the v3 work established.
    target_v = len(base) // 3
    target_h = len(base) // 3
    have_v = collections.Counter(r["answers"]["vertical"] for r in added)
    have_h = collections.Counter(r["answers"]["horizontal"] for r in added)

    by_v = collections.defaultdict(list)
    by_h = collections.defaultdict(list)
    for r in base:
        by_v[r["answers"]["vertical"]].append(r)
        by_h[r["answers"]["horizontal"]].append(r)

    need_v = {b: max(0, target_v - have_v[b]) for b in V_BANDS}
    need_h = {b: max(0, target_h - have_h[b]) for b in H_BANDS}

    by_v = collections.defaultdict(list)
    by_h = collections.defaultdict(list)
    for r in base:
        by_v[r["answers"]["vertical"]].append(r)
        by_h[r["answers"]["horizontal"]].append(r)

    chosen: list[dict] = []
    for r in base:
        v, h = r["answers"]["vertical"], r["answers"]["horizontal"]
        repeats = 1
        # repeat while this row's cell is short on EITHER marginal, capped at 2x
        # (2x keeps the added information without drowning the set in repeats)
        while repeats < 2 and (repeats <= need_v[v] or repeats <= need_h[h]):
            repeats += 1
        chosen.extend([r] * repeats)

    # top up any band still short, sampling directly from our own rows for it
    TOPUP = 260  # hard cap on extra copies, to avoid memorising a thin band
    for b in V_BANDS:
        cur = sum(1 for r in chosen if r["answers"]["vertical"] == b)
        for _ in range(max(0, min(TOPUP, need_v[b] - cur))):
            chosen.append(rng.choice(by_v[b]))
    for b in H_BANDS:
        cur = sum(1 for r in chosen if r["answers"]["horizontal"] == b)
        for _ in range(max(0, min(TOPUP, need_h[b] - cur))):
            chosen.append(rng.choice(by_h[b]))

    merged = base + added + chosen
    rng.shuffle(merged)
    out = HERE / args.out
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged),
                   encoding="utf-8")

    def grid(rs):
        c = collections.Counter((r["answers"]["vertical"], r["answers"]["horizontal"]) for r in rs)
        return c

    gb, ga = grid(base), grid(merged)
    print(f"\ntrain: {before_n} -> {len(merged)}  (+{len(added)} ethics, +{len(chosen)} resampled)")
    print(f"{'cell':22s} {'before':>7s} {'after':>7s} {'share':>7s}")
    print("-" * 46)
    for v in V_BANDS:
        for h in H_BANDS:
            print(f"{v + '/' + h:22s} {gb[(v,h)]:7d} {ga[(v,h)]:7d} {ga[(v,h)]/len(merged):7.3f}")
    vm = collections.Counter(r["answers"]["vertical"] for r in merged)
    hm = collections.Counter(r["answers"]["horizontal"] for r in merged)
    print(f"\nvertical   {dict((k, round(vm[k]/len(merged),3)) for k in V_BANDS)}")
    print(f"horizontal {dict((k, round(hm[k]/len(merged),3)) for k in H_BANDS)}")
    print(f"largest cell: {max(ga.values())/len(merged):.3f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())