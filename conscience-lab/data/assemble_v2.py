#!/usr/bin/env python3
"""Assemble Phase 3 training dataset v2 (train-from-base).

Buckets (gross, before dedupe):
  r1        281  phase3_train.jsonl (round-1 rows, kept wholesale)
  miss       50  dev harmful rows ft1 allowed (harness/results/laya-ft1)
  v3pair     88  laya_phase3_round2_variants_v3.jsonl (44 refuse + 44 twins)
  fp-mined   64  dev benign rows ft1 refused -> hard negatives (none/none)
  hn-written 25  laya_phase3_round2_hard_negatives_v3.jsonl (hand-written)
  tn        143  dev benign rows ft1 allowed (retention anchors)

Dedupe is by situation text: mined rows re-derive from dev, and r1 already
contains 28 of the misses / 23 of the FPs / 71 of the TNs (ft1 fails on
rows it trained on -- underfitting signal, see manifest). Net new: 248.

Output rows use the exact r1 schema:
  {"id", "fields": {"scenario"}, "answers": {"triage", "category"}}
Provenance goes in the manifest sidecar, not in the rows.

Usage:
  python3 assemble_v2.py   # runs in this dir, writes phase3_train_v2.*
"""

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "..", "eval", "laya_eval_dev.jsonl")
PER_CASE = os.path.join(HERE, "..", "harness", "results", "laya-ft1",
                        "per_case.jsonl")
R1 = os.path.join(HERE, "phase3_train.jsonl")
V3 = os.path.join(HERE, "laya_phase3_round2_variants_v3.jsonl")
HN = os.path.join(HERE, "laya_phase3_round2_hard_negatives_v3.jsonl")
OUT_ROWS = os.path.join(HERE, "phase3_train_v2.jsonl")
OUT_MANIFEST = os.path.join(HERE, "phase3_train_v2_manifest.json")

CANON = {"weapons", "cyberattack", "credential-theft", "fraud",
         "impersonation", "deception", "sexual-broad", "theft-broad",
         "idolatry", "degrading-speech", "harassment", "doxxing",
         "defamation", "animal-cruelty", "self-harm", "csam"}

# Rows dropped even if present in inputs, with reason. Keep minimal and
# documented; prefer dropping over leaking into the locked test.
EXCLUDE = {
    # r1 row sim=0.86 to locked w-pills-lethal (same lethal-overdose
    # question + "Asking for a friend"). Pre-existing r1 leak (ft1 already
    # saw it); v2 must not inherit it. Self-harm stays largest bucket.
    "p3-var-sh-01": "near-dupe of locked test row w-pills-lethal",
}


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def to_train(row_id, situation, expect, refusal_category):
    """Eval-shape row -> r1 train schema."""
    if expect == "refuse":
        assert refusal_category in CANON, f"{row_id}: bad cat {refusal_category}"
        answers = {"triage": "violation", "category": refusal_category}
    else:
        assert refusal_category is None, f"{row_id}: benign with cat"
        answers = {"triage": "none", "category": "none"}
    return {"id": row_id, "fields": {"scenario": situation},
            "answers": answers}


def main():
    dev = {r["id"]: r for r in load(DEV)}
    pc = load(PER_CASE)
    r1 = load(R1)
    v3 = load(V3)
    hn = load(HN)

    # bucket -> list of train rows (with checkpoint kept aside for manifest)
    buckets = {"r1": list(r1), "miss": [], "v3pair": [], "fp-mined": [],
               "hn-written": [], "tn": []}
    meta = {}  # row id -> {checkpoint, expect}

    for r in pc:
        d = dev[r["id"]]
        harmful, refused = r["expect_refuse"], r["predicted_refuse"]
        kind = ("miss" if harmful and not refused else
                "fp-mined" if not harmful and refused else
                "tn" if not harmful and not refused else None)
        if kind is None:  # hit: covered by r1 retention, skip (no new info)
            continue
        t = to_train(r["id"], d["situation"], d["expect"],
                     d["refusal_category"])
        buckets[kind].append(t)
        meta[r["id"]] = {"checkpoint": d["checkpoint"], "expect": d["expect"]}

    for r in v3:
        buckets["v3pair"].append(to_train(r["id"], r["situation"], r["expect"],
                                          r["refusal_category"]))
        meta[r["id"]] = {"checkpoint": r["checkpoint"], "expect": r["expect"]}
    for r in hn:
        buckets["hn-written"].append(to_train(r["id"], r["situation"],
                                              r["expect"],
                                              r["refusal_category"]))
        meta[r["id"]] = {"checkpoint": r["checkpoint"], "expect": r["expect"]}

    gross = {k: len(v) for k, v in buckets.items()}
    assert gross["miss"] == 50, gross
    assert gross["fp-mined"] == 64, gross
    assert gross["tn"] == 143, gross
    assert gross["v3pair"] == 88, gross
    assert gross["hn-written"] == 25, gross
    assert gross["r1"] == 281, gross

    # merge + dedupe by situation; union source tags; conflicts are fatal
    merged = {}  # situation -> {"row", "sources"}
    for bucket in ["r1", "miss", "v3pair", "fp-mined", "hn-written", "tn"]:
        for t in buckets[bucket]:
            if t["id"] in EXCLUDE:
                continue
            sit = t["fields"]["scenario"]
            if sit in merged:
                m = merged[sit]
                assert m["row"]["answers"] == t["answers"], (
                    f"label conflict for situation {sit[:60]!r}: "
                    f"{m['row']['id']} vs {t['id']}")
                m["sources"].append(bucket)
                # prefer the shortest stable id (dev ids over generated ones)
                if len(t["id"]) < len(m["row"]["id"]):
                    m["row"]["id"] = t["id"]
            else:
                merged[sit] = {"row": dict(t), "sources": [bucket]}

    rows = sorted((m["row"] for m in merged.values()),
                  key=lambda r: r["id"])
    with open(OUT_ROWS, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    viol = sum(1 for r in rows if r["answers"]["triage"] == "violation")
    # per-bucket net contribution (rows where this bucket is a source)
    net = Counter()
    for m in merged.values():
        for s in set(m["sources"]):
            net[s] += 1
    manifest = {
        "gross_counts": gross,
        "gross_total": sum(gross.values()),
        "net_rows": len(rows),
        "net_violation": viol,
        "net_none": len(rows) - viol,
        "rows_with_each_source": dict(sorted(net.items())),
        "already_in_r1": {
            # mined rows whose situation was already an r1 row
            k: sum(1 for m in merged.values()
                   if k in m["sources"] and "r1" in m["sources"])
            for k in ["miss", "fp-mined", "tn"]},
        "per_row_sources": {m["row"]["id"]: sorted(set(m["sources"]))
                            for m in merged.values()},
        "train_from": "base (not from ft1)",
        "excluded": {k: v for k, v in EXCLUDE.items()},
        "notes": [
            "28/50 misses, 23/64 FPs, 71/143 TNs were already r1 rows: "
            "ft1 fails on rows it trained on (underfitting, not coverage).",
            "Denied-by-construction: no locked-test rows; mined rows are "
            "dev-derived and dev∩locked=∅; v3/hn cleared vs full eval.",
        ],
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"gross {sum(gross.values())} -> net {len(rows)} rows "
          f"({viol} violation / {len(rows) - viol} none)")
    print(f"already-in-r1: {manifest['already_in_r1']}")
    print(f"wrote {OUT_ROWS} + {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
