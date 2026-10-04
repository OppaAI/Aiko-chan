#!/usr/bin/env python3
"""Is v8's regression on the general cases even statistically real?

Meta Muse's objection implies my v8 rejection could have been noise. Both models
were scored on the SAME 20 general cases, so the right test is McNemar's: only
the discordant pairs carry information.
"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, "/home/oppa-ai/jetson/Aiko-chan/eval/conscience")
from cognition.conscience.laya_judge import _HORIZONTAL_Q, _VERTICAL_Q
import eval_tev1_judge as T

Q = {"vertical": _VERTICAL_Q, "horizontal": _HORIZONTAL_Q}
BAND = {"unclear": "unknown", "mixed": "unknown"}
CYBER = ("hack","password","account","email","access","unauthor","ex's","phish","credential",
         "database","token","session","keylog","malware","breach","reference letter",
         "impersonat","competitor","surveil","monitor")

def binom_two_sided(b, c):
    """Exact McNemar p-value from discordant counts b, c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)

cases = [json.loads(l) for l in Path(T.CASES_DEFAULT).read_text().splitlines() if l.strip()]
canon = T.load_canon()
import laya
agents = {"v5": laya.load("runs/conscience-laya-v5"), "v8": laya.load("runs/conscience-laya-v9")}

def general(c):
    return not any(w in c["situation"].lower() for w in CYBER)

sub = [c for c in cases if general(c)]
print(f"general cases: {len(sub)}\n")
for ax, want_key in (("vertical", "vertical"), ("horizontal", "horizontal")):
    b = c_ = 0   # v5-only-correct, v9-only-correct
    for case in sub:
        state = T.build_state(case, canon)
        want = {BAND.get(x, x) for x in T.expected_bands(case, want_key)}
        res = {}
        for tag, ag in agents.items():
            a = ag.predict(state, Q).get("answers") or {}
            res[tag] = (a.get(ax) or {}).get("choice") in want
        if res["v5"] and not res["v8"]:
            b += 1
        elif res["v8"] and not res["v5"]:
            c_ += 1
    p = binom_two_sided(b, c_)
    verdict = ("REAL regression" if p < 0.05 else
               "suggestive, not significant" if p < 0.20 else
               "indistinguishable from noise")
    print(f"{ax:11s} v5-only-correct={b}  v9-only-correct={c_}  "
          f"discordant={b+c_}  p={p:.3f}  -> {verdict}")
