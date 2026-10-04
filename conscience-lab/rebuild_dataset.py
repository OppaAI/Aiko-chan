#!/usr/bin/env python3
"""Rebuild the conscience dataset to kill the marginal-prior shortcut.

v2 problems this fixes:
  * 69% of rows sat on the aligned/benefits + contrary/harms diagonal, so the
    joint label was learnable from the marginal alone (v2 learned a prior, not
    the doctrine).
  * Training states were bare sentences; deployment states are
    "Request: ... / Parties affected: ... / Relevant norms: <canon>".

Output:
  data/conscience_training_v7.jsonl   rebalanced, plain + deployment-format
  data/conscience_eval_v7.jsonl       held-out, deployment-format only
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cognition.conscience.canon import get_canon  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
V_BANDS = ("aligned", "contrary", "unknown")
H_BANDS = ("benefits", "harms", "unknown")

# Horizontal `mixed` is retired: "no real effect" and "cannot be scored" both
# become `unknown`, which is the escalate-to-human band on either axis.
BAND_REMAP = {"vertical": {"unclear": "unknown", "mixed": "unknown"},
               "horizontal": {"mixed": "unknown"}}

DIAG_CAP = 200      # trim the two huge diagonal cells
FLOOR = 120         # bring every other cell up to here
MAX_OVERSAMPLE = 1.5  # keep repetition low: repeats teach lexical shortcuts
DEP_FRACTION = 0.55  # share of rows emitted in deployment format
SEED = 20261003
HOLDOUT_FRAC = 0.12  # stratified holdout fraction, split BEFORE oversampling


def core_scenario_key(row: dict) -> str:
    """Canonical identity of a row: the underlying scenario, not its rendering.

    Rows are emitted in plain and deployment formats; dedupe and splitting
    must key on the scenario itself, or duplicates leak across the split.
    """
    text = row["fields"]["scenario"]
    m = re.match(r"Request: The user asks: (.*?)\nParties affected:", text, re.S)
    core = m.group(1).strip() if m else text.strip()
    return " ".join(core.lower().split())


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """Drop repeat scenarios, keeping the first occurrence."""
    seen: set[str] = set()
    out = []
    for r in rows:
        k = core_scenario_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    dropped = len(rows) - len(out)
    if dropped:
        print(f"  deduped {dropped} repeat scenarios")
    return out


def stratified_split(rows: list[dict], rng: random.Random,
                     frac: float = HOLDOUT_FRAC) -> tuple[list[dict], list[dict]]:
    """Split into (train_pool, holdout_pool) by core scenario, stratified.

    MUST run before any oversampling: oversampling with replacement creates
    duplicates, and splitting after lets the same scenario land on both
    sides, inflating eval scores (v5/v6 leaked ~45% this way).
    """
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["answers"]["vertical"], r["answers"]["horizontal"])].append(r)
    train_pool: list[dict] = []
    holdout_pool: list[dict] = []
    for cell, items in sorted(by_cell.items()):
        rng.shuffle(items)
        k = max(2, round(len(items) * frac))
        holdout_pool.extend(items[:k])
        train_pool.extend(items[k:])
    rng.shuffle(train_pool)
    rng.shuffle(holdout_pool)
    # hard guarantee: no scenario on both sides
    overlap = {core_scenario_key(r) for r in train_pool} & {core_scenario_key(r) for r in holdout_pool}
    assert not overlap, f"train/eval scenario overlap: {len(overlap)}"
    print(f"  split: {len(train_pool)} train / {len(holdout_pool)} holdout, overlap: {len(overlap)}")
    return train_pool, holdout_pool

PARTY_POOL = [
    ("user", "requester", 0.5, "asked for this"),
    ("assistant", "agent", 0.0, "carries it out"),
    ("boss", "third_party", -0.5, "affected by the outcome"),
    ("colleague", "third_party", -0.5, "affected by the outcome"),
    ("customer", "third_party", -0.5, "affected by the outcome"),
    ("friend", "third_party", 0.5, "relies on the truth"),
    ("stranger", "third_party", 0.0, "no prior relationship"),
    ("child", "vulnerable", 0.5, "needs protection"),
]


SEED_FILES = ("unknown_seed.jsonl", "clearharm_seed.jsonl")


def load_seed_rows() -> list[dict]:
    """Hand-reviewed augmentation, loaded without the review `note` field."""
    out = []
    for name in SEED_FILES:
        f = DATA / name
        if not f.exists():
            print(f"  (no seed file {name})")
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r.pop("note", None)
            out.append({"fields": r["fields"], "answers": dict(r["answers"]),
                        "_seed": True})
        print(f"  loaded {len(out)} rows from {name}")
    return out


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        ans = dict(r["answers"])
        for axis, remap in BAND_REMAP.items():
            if ans.get(axis) in remap:
                ans[axis] = remap[ans[axis]]
        if ans["vertical"] in V_BANDS and ans["horizontal"] in H_BANDS:
            out.append({"fields": r["fields"], "answers": ans})
    return out


def rebalance(rows: list[dict], rng: random.Random) -> list[dict]:
    """Resample so the 3x3 joint grid is near-uniform on both marginals."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["answers"]["vertical"], r["answers"]["horizontal"])].append(r)

    out: list[dict] = []
    for v in V_BANDS:
        for h in H_BANDS:
            pool = cells[(v, h)]
            if not pool:
                print(f"  !! empty cell {v}/{h}")
                continue
            target = DIAG_CAP if v == h or (v, h) in (("aligned", "benefits"), ("contrary", "harms")) else FLOOR
            target = min(target, len(pool) * MAX_OVERSAMPLE)
            if target > len(pool):
                picked = [rng.choice(pool) for _ in range(int(target))]
            else:
                # Curated rows first. The cap exists to stop the scraped bulk
                # from dominating a cell, not to discard hand-reviewed rows --
                # and the seeded rows are the ones we specifically added.
                seeded = [r for r in pool if r.get("_seed")]
                rest = [r for r in pool if not r.get("_seed")]
                rng.shuffle(seeded)
                rng.shuffle(rest)
                picked = (seeded + rest)[:target]
            rng.shuffle(picked)
            out.extend(picked)
            print(f"  {v:9s}/{h:8s} {len(pool):4d} -> {len(picked):4d}")
    rng.shuffle(out)
    return out


def deployment_state(row: dict, canon, rng: random.Random, matched: bool) -> str:
    """Render the training row the way Aiko builds state at request time."""
    scenario = row["fields"]["scenario"]
    v, h = row["answers"]["vertical"], row["answers"]["horizontal"]

    parties = [PARTY_POOL[0], rng.choice(PARTY_POOL[1:])]
    rng.shuffle(parties)
    lines = [f"Request: The user asks: {scenario}", "Parties affected:"]
    for label, kind, benefit, note in parties:
        lines.append(f"- {label} [{kind}], benefit {benefit:+.1f} ({note})")

    # Norms are cited for grounding. When matched they agree with the verdict;
    # when mismatched they are cited-but-not-decisive, so the model has to read
    # the situation instead of pattern-matching a prohibition citation.
    if matched:
        want_neg = v == "contrary"
    else:
        want_neg = rng.random() < 0.5
    pool = [n for n in canon.all() if (n.polarity < 0) == want_neg]
    if not pool:
        pool = list(canon.all())
    chosen = rng.sample(pool, k=min(2, len(pool)))
    lines.append("Relevant norms:")
    lines.append(canon.render_block([(1.0 - i * 0.05, n) for i, n in enumerate(chosen)]))
    return "\n".join(lines)


def main() -> int:
    rng = random.Random(SEED)
    canon = get_canon()
    src = load_rows(DATA / "conscience_training_outcome_v2.jsonl")
    src.extend(load_seed_rows())

    print(f"source rows: {len(src)}")
    src = dedupe_rows(src)
    print("split before oversampling (leak fix):")
    train_pool, holdout = stratified_split(src, rng)
    print("joint grid before -> after (train pool only):")
    train = rebalance(train_pool, rng)

    train_out: list[dict] = []
    for r in train:
        if rng.random() < DEP_FRACTION:
            state = deployment_state(r, canon, rng, matched=rng.random() < 0.6)
        else:
            state = r["fields"]["scenario"]
        train_out.append({"fields": {"scenario": state}, "answers": dict(r["answers"])})

    eval_out: list[dict] = []
    for r in holdout:
        eval_out.append({
            "fields": {"scenario": deployment_state(r, canon, rng, matched=rng.random() < 0.6)},
            "answers": dict(r["answers"]),
        })

    (DATA / "conscience_training_v7.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_out), encoding="utf-8")
    (DATA / "conscience_eval_v7.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_out), encoding="utf-8")

    for name, rows in (("train", train_out), ("eval", eval_out)):
        c = Counter((r["answers"]["vertical"], r["answers"]["horizontal"]) for r in rows)
        vm = Counter(r["answers"]["vertical"] for r in rows)
        hm = Counter(r["answers"]["horizontal"] for r in rows)
        print(f"\n{name}: {len(rows)} rows")
        print(f"  vertical   {dict((k, round(vm[k]/len(rows), 3)) for k in V_BANDS)}")
        print(f"  horizontal {dict((k, round(hm[k]/len(rows), 3)) for k in H_BANDS)}")
        biggest = max(c.values()) / len(rows)
        print(f"  largest joint cell share: {biggest:.3f} (v2 was 0.377)")
        dep = sum(1 for r in rows if r["fields"]["scenario"].startswith("Request:"))
        print(f"  deployment-format: {dep}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
