#!/usr/bin/env python3
"""Relabel the 40 hand-reviewed `unknown` rows onto the 4-band horizontal axis.

Design objective, set by Oppa: optimise for recall on harm and accept precision
loss on the benign side. Stopping something wrong is worth more than the cost of
asking about something fine. So where an effect is genuinely unstated, the row
goes to `uncertain` (which escalates), never to `benefits`.

The rationale notes already written on these rows determine the band:

  no effect on anyone / protective of me alone   -> no-effect   (proceed)
  they are harmed, size unstated                  -> harms       (stop)
  effect unknowable / offsetting / not given /
    missing context / cannot be scored            -> uncertain   (ask Oppa)

The `uncertain` bucket is deliberately the largest. That is the intended cost of
the asymmetry, and the autonomy counter is what keeps it from becoming a
scruple: after five approvals of an action class, Aiko stops asking.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "unknown_seed.jsonl"
DST = HERE / "data" / "unknown_seed_4band.jsonl"

NO_EFFECT_MARKERS = (
    "no effect on anyone else",
    "protective of me alone",
)
HARMS_MARKERS = (
    "they are harmed",           # the harm is known; only its size is unstated
)


def band_for(note: str) -> str:
    n = note.lower()
    if any(m in n for m in NO_EFFECT_MARKERS):
        return "no-effect"
    if any(m in n for m in HARMS_MARKERS):
        return "harms"
    return "uncertain"


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    tally = collections.Counter()
    for r in rows:
        band = band_for(r.get("note", ""))
        tally[band] += 1
        out.append({
            "fields": r["fields"],
            "answers": {"vertical": r["answers"]["vertical"], "horizontal": band},
            "note": r.get("note", ""),
            "band_reason": band,
        })
    DST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                   encoding="utf-8")

    print(f"{len(rows)} rows relabelled -> {DST.name}")
    print(f"  {dict(tally)}\n")
    print("by vertical band:")
    for band in ("no-effect", "uncertain", "harms"):
        vs = collections.Counter(r["answers"]["vertical"] for r in out
                                 if r["answers"]["horizontal"] == band)
        if vs:
            print(f"  {band:11s} {dict(vs)}")
    print("\nsample of each:")
    for band in ("no-effect", "uncertain", "harms"):
        ex = next((r for r in out if r["answers"]["horizontal"] == band), None)
        if ex:
            print(f"  {band:11s} [{ex['answers']['vertical']:8s}] {ex['fields']['scenario'][:62]}")
            print(f"              reason: {ex['note'][:66]}")


if __name__ == "__main__":
    main()