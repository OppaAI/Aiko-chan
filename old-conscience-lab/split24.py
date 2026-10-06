#!/usr/bin/env python3
"""Where did v8 lose to v5? Split the 24 hand-authored cases by cyber-overlap.

If v8's regression is concentrated in cases whose scenarios resemble its cyber
training rows, the loss is a labelling artifact. If it dropped on general cases
too, v8 has genuinely narrowed.
"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, "/home/oppa-ai/jetson/Aiko-chan/eval/conscience")
from cognition.conscience.laya_judge import _HORIZONTAL_Q, _VERTICAL_Q
import eval_tev1_judge as T

Q = {"vertical": _VERTICAL_Q, "horizontal": _HORIZONTAL_Q}
BAND = {"unclear": "unknown", "mixed": "unknown"}
CYBER_WORDS = ("hack", "password", "account", "email", "access", "unauthor", "ex's", "phish",
               "credential", "database", "token", "session", "keylog", "malware", "breach",
               "reference letter", "impersonat", "competitor", "surveil", "monitor")

def is_cyber(text):
    low = text.lower()
    return any(w in low for w in CYBER_WORDS)

def main():
    import laya
    cases = [json.loads(l) for l in Path(T.CASES_DEFAULT).read_text().splitlines() if l.strip()]
    canon = T.load_canon()
    agents = {"v5": laya.load("runs/conscience-laya-v5"), "v8": laya.load("runs/conscience-laya-v8")}
    rows = []
    for c in cases:
        state = T.build_state(c, canon)
        want_v = {BAND.get(b, b) for b in T.expected_bands(c, "vertical")}
        want_h = {BAND.get(b, b) for b in T.expected_bands(c, "horizontal")}
        rec = {"id": c["id"], "cat": c["category"], "cyber": is_cyber(c["situation"]),
               "sit": c["situation"][:52], "wv": want_v, "wh": want_h}
        for tag, ag in agents.items():
            a = ag.predict(state, Q).get("answers") or {}
            gv = (a.get("vertical") or {}).get("choice"); gh = (a.get("horizontal") or {}).get("choice")
            rec[f"{tag}_v"] = gv in want_v; rec[f"{tag}_h"] = gh in want_h
        rows.append(rec)

    for grp, name in ((True, "CYBER-OVERLAP cases"), (False, "GENERAL cases")):
        sub = [r for r in rows if r["cyber"] == grp]
        print(f"\n=== {name}: {len(sub)} ===")
        for ax in ("v", "h"):
            v5 = sum(r[f"v5_{ax}"] for r in sub); v8 = sum(r[f"v8_{ax}"] for r in sub)
            d = v8 - v5
            print(f"   {ax}: v5={v5:2d}/{len(sub)}  v8={v8:2d}/{len(sub)}  delta={d:+d}")
        lost = [r for r in sub if (r["v5_v"] + r["v5_h"]) > (r["v8_v"] + r["v8_h"])]
        for r in lost:
            print(f"     LOST {r['id']:22s} [{r['cat']:12s}] {r['sit']}")

if __name__ == "__main__":
    main()
