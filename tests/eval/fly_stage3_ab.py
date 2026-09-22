"""Stage 3 A/B: Fly OFF vs LIVE for teach → rank and GF interrupt.

Run:
  MEMORY_FLYMB_MODE=off MEMORY_FLYGF_MODE=off python -m tests.eval.fly_stage3_ab
  MEMORY_FLYMB_MODE=live MEMORY_FLYGF_MODE=live python -m tests.eval.fly_stage3_ab
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    from cognition.flymemory.teach_api import teach_preference
    from cognition.memory.fly_rank import adjust_recall_score
    from cognition.memory.preference_store import preference_delta, record_preference
    from cognition.fly_behavior.giant_fiber import assess_interrupt

    uid = "stage3-ab"
    modes = {
        k: os.getenv(k, "off")
        for k in ("MEMORY_FLYMB_MODE", "MEMORY_FLYGF_MODE")
    }
    report: dict = {"modes": modes, "ok": True, "checks": []}

    pref = teach_preference("fruit tarts", direction="avoid", user_id=uid)
    record_preference("fruit tarts", "avoid", user_id=uid)
    tart_score, tart_meta = adjust_recall_score(1.0, "we ate fruit tarts yesterday", user_id=uid, log_influence=False)
    other_score, _ = adjust_recall_score(1.0, "the garden was quiet", user_id=uid, log_influence=False)
    pdelta = preference_delta("we ate fruit tarts yesterday", user_id=uid)
    gf = assess_interrupt("STOP right now")

    live = modes.get("MEMORY_FLYMB_MODE") == "live"
    report["checks"].append({"name": "teach", "pref": pref})
    report["checks"].append({"name": "rank", "tart": tart_score, "other": other_score, "meta": tart_meta, "pdelta": pdelta})
    report["checks"].append({"name": "gf", "gf": gf})

    if live:
        if tart_score >= other_score and pdelta >= 0:
            report["ok"] = False
            report["reason"] = "live avoid did not suppress tart vs other"
        if not (gf.get("interrupt") or float(gf.get("urgency") or 0) >= 0.5):
            report["ok"] = False
            report["reason"] = "live GF did not flag STOP"
    print(json.dumps(report, default=str))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
