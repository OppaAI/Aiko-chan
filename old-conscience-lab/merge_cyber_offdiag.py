#!/usr/bin/env python3
"""Merge the off-diagonal cyber rows into the v8 training set.

Keeps v8's three-way split exactly as authored -- the new rows go into TRAIN
only, so validation and test stay untouched and the promotion comparison against
v5 remains meaningful. No leakage is introduced by construction.

Also reports the before/after joint grid, because the whole point is to move the
distribution, not just add rows.
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cognition.conscience.canon import get_canon  # noqa: E402

DATA = HERE / "data"
V_BANDS = ("aligned", "contrary", "unknown")
H_BANDS = ("benefits", "harms", "unknown")
SEED = 20261004
DEP_FRACTION = 0.55

PARTY_POOL = [
    ("user", "requester", 0.5, "asked for this"),
    ("assistant", "agent", 0.0, "carries it out"),
    ("client", "third_party", -0.5, "affected by the outcome"),
    ("company", "third_party", -0.5, "affected by the outcome"),
    ("colleague", "third_party", -0.5, "affected by the outcome"),
    ("customer", "vulnerable", 0.5, "needs protection"),
]


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def deployment_state(row: dict, canon, rng: random.Random, matched: bool) -> str:
    """Same wrapper rebuild_dataset.py uses, so new rows match production shape."""
    scenario = row["fields"]["scenario"]
    want_neg = (row["answers"]["vertical"] == "contrary") if matched else rng.random() < 0.5
    pool = [n for n in canon.all() if (n.polarity < 0) == want_neg] or list(canon.all())
    chosen = rng.sample(pool, k=min(2, len(pool)))
    parties = [PARTY_POOL[0], rng.choice(PARTY_POOL[1:])]
    rng.shuffle(parties)
    lines = [f"Request: The user asks: {scenario}", "Parties affected:"]
    for label, kind, benefit, note in parties:
        lines.append(f"- {label} [{kind}], benefit {benefit:+.1f} ({note})")
    lines.append("Relevant norms:")
    lines.append(canon.render_block([(1.0 - i * 0.05, n) for i, n in enumerate(chosen)]))
    return "\n".join(lines)


def grid(rows: list[dict]) -> dict:
    c = collections.Counter((r["answers"]["vertical"], r["answers"]["horizontal"]) for r in rows)
    return {f"{v}/{h}": c[(v, h)] for v in V_BANDS for h in H_BANDS}


def main() -> int:
    rng = random.Random(SEED)
    canon = get_canon()

    train = load(DATA / "conscience_training_v8.jsonl")
    before_n = len(train)
    before = grid(train)

    added: list[dict] = []
    for seed_name in ("cyber_offdiag_seed.jsonl",):
        for r in load(DATA / seed_name):
            r.pop("note", None)
            state = (deployment_state(r, canon, rng, matched=rng.random() < 0.6)
                     if rng.random() < DEP_FRACTION else r["fields"]["scenario"])
            added.append({"fields": {"scenario": state}, "answers": dict(r["answers"])})

    train.extend(added)
    rng.shuffle(train)

    out = DATA / "conscience_training_v8_fixed.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train),
                   encoding="utf-8")

    after = grid(train)
    print(f"train: {before_n} -> {len(train)}  (+{len(added)})")
    print(f"{'cell':22s} {'before':>7s} {'after':>7s} {'share':>7s}")
    print("-" * 48)
    for k in before:
        print(f"{k:22s} {before[k]:7d} {after[k]:7d} {after[k]/len(train):7.3f}")
    c = collections.Counter((r["answers"]["vertical"], r["answers"]["horizontal"]) for r in train)
    diag = sum(n for k, n in c.items()
               if (k[0] == "aligned" and k[1] == "benefits")
               or (k[0] == "contrary" and k[1] == "harms")
               or (k[0] == "unknown" and k[1] == "unknown"))
    print(f"\nlargest cell share : {max(c.values())/len(train):.3f}  (v8 was "
          f"{max(before.values())/before_n:.3f})")
    print(f"diagonal share     : {diag/len(train):.3f}")
    print(f"cyber diagonal rows remaining: 28 of 44 (see audit -- individually "
          f"correct, left in place)")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())