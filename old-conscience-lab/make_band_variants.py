#!/usr/bin/env python3
"""Emit the 2x2 and 3x2 label schemes from the same 4-band source.

Controlled comparison: both variants come from one dataset, so any difference
between them is the label scheme alone, not the data.

2x2  v: permitted | not-permitted          h: harm | no-harm
3x2  v: permitted | not-permitted | cannot-tell   h: harm | no-harm

The known cost of 2x2, which this is meant to measure: `unknown` on the vertical
axis means "I cannot tell whether this breaks a commandment", and binary has no
way to say that, so it becomes `permitted` and only low confidence distinguishes
it. 3x2 keeps an explicit band for it.

Horizontal `uncertain` maps to `no-harm` in BOTH -- the horizontal question stays
binary because the model is competent at detecting harm. That is a deliberate
loss and it is applied equally to both arms.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "conscience_training_v11b.jsonl"

V_2x2 = {"contrary": "not-permitted", "aligned": "permitted", "unknown": "permitted"}
V_3x2 = {"contrary": "not-permitted", "aligned": "permitted", "unknown": "cannot-tell"}
H_BOTH = {"harms": "harm", "benefits": "no-harm", "no-effect": "no-harm",
          "uncertain": "no-harm"}


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"source: {len(rows)} rows (4-band)\n")

    for tag, vmap, suffix in (("2x2", V_2x2, "2x2"), ("3x2", V_3x2, "3x2")):
        out = []
        unmapped = collections.Counter()
        for r in rows:
            v, h = r["answers"]["vertical"], r["answers"]["horizontal"]
            if v not in vmap:
                unmapped[v] += 1
                continue
            if h not in H_BOTH:
                unmapped[h] += 1
                continue
            out.append({"fields": r["fields"],
                        "answers": {"vertical": vmap[v], "horizontal": H_BOTH[h]}})
        path = HERE / "data" / f"conscience_training_{suffix}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                        encoding="utf-8")
        vc = collections.Counter(r["answers"]["vertical"] for r in out)
        hc = collections.Counter(r["answers"]["horizontal"] for r in out)
        print(f"--- {tag} -> {path.name}: {len(out)} rows")
        print(f"    vertical  : {dict(vc)}")
        print(f"    horizontal: {dict(hc)}")
        if unmapped:
            print(f"    UNMAPPED  : {dict(unmapped)}")
        print()


if __name__ == "__main__":
    main()