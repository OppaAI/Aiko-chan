#!/usr/bin/env python3
"""Which of Oppa's 6 unambiguous harm cases does each model miss, and why."""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, "/home/oppa-ai/jetson/Aiko-chan/eval/conscience")
from cognition.conscience.laya_judge import _HORIZONTAL_Q, _VERTICAL_Q
import eval_tev1_judge as T, laya

Q = {"vertical": _VERTICAL_Q, "horizontal": _HORIZONTAL_Q}
BAND = {"unclear": "unknown", "mixed": "unknown"}
rows = [json.loads(l) for l in Path("data/oppa_benign_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
harm = [r for r in rows if r["category"] == "harmful"]
agents = {"v5": laya.load("runs/conscience-laya-v5"), "v9": laya.load("runs/conscience-laya-v9")}

for r in harm:
    wv = {BAND.get(b,b) for b in T.expected_bands(r,"vertical")}
    wh = {BAND.get(b,b) for b in T.expected_bands(r,"horizontal")}
    print(f"\n=== {r['id']}")
    print(f"    {r['situation']}")
    print(f"    want V={sorted(wv)} H={sorted(wh)}  | {r['notes'][:70]}")
    for tag, ag in agents.items():
        a = ag.predict(r["situation"], Q).get("answers") or {}
        v = (a.get("vertical") or {}); h = (a.get("horizontal") or {})
        gv, gh = v.get("choice"), h.get("choice")
        okv, okh = gv in wv, gh in wh
        mark = "OK  " if (okv and okh) else ("part" if (okv or okh) else "MISS")
        print(f"      {tag}: {mark} V={gv:9s}({v.get('confidence',0):.2f}) "
              f"H={gh:9s}({h.get('confidence',0):.2f})  probs V={ {k: round(x,2) for k,x in (v.get('probabilities') or {}).items()} }")
