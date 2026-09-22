"""Stage 4 A/B: eligibility delayed credit + dopamine + reversal.

Run:
  MEMORY_FLYMB_MODE=live FLY_ELIGIBILITY=1 python -m tests.eval.fly_stage4_ab
"""
from __future__ import annotations

import json
import os


def main() -> int:
    os.environ.setdefault("MEMORY_FLYMB_MODE", "live")
    os.environ.setdefault("FLY_ELIGIBILITY", "1")

    from cognition.flymemory.eligibility import assign_credit, clear, record_step, stats
    from cognition.flymemory.dopamine import pulse, split_channels
    from cognition.flymemory.consolidate_mb import consolidate

    uid = "stage4-ab"
    clear(uid)
    report: dict = {"ok": True, "checks": []}

    ch = split_channels(0.8)
    report["checks"].append({"name": "split", "ch": ch})
    if ch["pam"] <= 0 or ch["ppl1"] != 0:
        report["ok"] = False

    for t in ("looked up fruit tart recipe", "mentioned birthday pastry", "user scowled"):
        record_step(uid, t)
    credit = assign_credit(uid, -0.6)
    report["checks"].append({"name": "delayed_credit", "credit": credit, "stats": stats(uid)})
    if not credit.get("taught") and credit.get("reason") not in ("mb_unavailable",):
        if credit.get("reason") not in ("disabled_or_off", "mb_unavailable"):
            report["ok"] = False
            report["reason"] = "credit_failed"

    dop = pulse(-0.5, user_id=uid, text="fruit tarts again", source="stage4_ab")
    report["checks"].append({"name": "dopamine", "dop": dop})

    cons = consolidate(uid, sleep_pressure=0.9, force=True)
    report["checks"].append({"name": "consolidate", "cons": cons})

    print(json.dumps(report, default=str))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
