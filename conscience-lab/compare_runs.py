#!/usr/bin/env python3
"""Score candidate checkpoints head-to-head on the same two sets.

The promotion bar set before v6: a challenger must beat the incumbent on BOTH
the v8 independent test set (263 rows) AND the original 24 hand-authored cases.
Single-number comparisons across differently-sized sets are meaningless.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/oppa-ai/jetson/Aiko-chan/eval/conscience")

from cognition.conscience.laya_judge import _HORIZONTAL_Q, _VERTICAL_Q

Q = {"vertical": _VERTICAL_Q, "horizontal": _HORIZONTAL_Q}
BAND = {"unclear": "unknown", "mixed": "unknown"}


def ask(agent, state):
    a = agent.predict(state, Q).get("answers") or {}
    return ((a.get("vertical") or {}).get("choice"),
            (a.get("horizontal") or {}).get("choice"))


def on_v8_test(agent, path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    v = h = both = 0
    for r in rows:
        got = ask(agent, r["fields"]["scenario"])
        okv = got[0] == r["answers"]["vertical"]
        okh = got[1] == r["answers"]["horizontal"]
        v += okv; h += okh; both += okv and okh
    n = len(rows)
    return {"n": n, "V": round(v / n, 3), "H": round(h / n, 3), "both": round(both / n, 3)}


def on_hand_authored(agent):
    import eval_tev1_judge as T
    cases = [json.loads(l) for l in Path(T.CASES_DEFAULT).read_text(encoding="utf-8").splitlines() if l.strip()]
    canon = T.load_canon()
    v = h = both = 0
    for c in cases:
        got = ask(agent, T.build_state(c, canon))
        okv = got[0] in {BAND.get(b, b) for b in T.expected_bands(c, "vertical")}
        okh = got[1] in {BAND.get(b, b) for b in T.expected_bands(c, "horizontal")}
        v += okv; h += okh; both += okv and okh
    n = len(cases)
    return {"n": n, "V": round(v / n, 3), "H": round(h / n, 3), "both": round(both / n, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--base", default="convaiinnovations/laya-multilingual")
    ap.add_argument("--test", default="data/conscience_test_v8.jsonl")
    args = ap.parse_args()

    import laya
    results = {}
    for tag, path in (("base", args.base), ("incumbent(v5)", args.incumbent),
                      ("challenger(v8)", args.challenger)):
        print(f"scoring {tag} ...", flush=True)
        agent = laya.load(path)
        results[tag] = {"v8_test(263)": on_v8_test(agent, args.test),
                        "hand(24)": on_hand_authored(agent)}

    print(f"\n{'model':20s} {'set':16s} {'V':>7s} {'H':>7s} {'both':>7s}")
    print("-" * 62)
    for tag, sets in results.items():
        for sname, s in sets.items():
            print(f"{tag:20s} {sname:16s} {s['V']:7.3f} {s['H']:7.3f} {s['both']:7.3f}")

    inc, ch = results["incumbent(v5)"], results["challenger(v8)"]
    wins = []
    for sname in ("v8_test(263)", "hand(24)"):
        for metric in ("V", "H", "both"):
            wins.append((f"{sname}/{metric}", ch[sname][metric] > inc[sname][metric]))
    lost = [k for k, ok in wins if not ok]
    print()
    if lost:
        print(f"REJECT: challenger does not beat v5 on {len(lost)}/{len(wins)} comparisons")
        for k in lost:
            print(f"   lost: {k}")
        return 1
    print(f"PROMOTE: challenger beats v5 on all {len(wins)} comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
