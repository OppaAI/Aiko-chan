"""Stage 6 smoke: DN body drive + GF cancel.

Run:
  MEMORY_FLYDN_MODE=live MEMORY_FLYGF_MODE=live python -m tests.eval.fly_stage6_ab
"""
from __future__ import annotations

import json
import os


def main() -> int:
    os.environ.setdefault("MEMORY_FLYDN_MODE", "live")
    os.environ.setdefault("MEMORY_FLYGF_MODE", "live")
    report: dict = {"ok": True, "checks": []}

    from cognition.flysense.dn import FlyDN
    from cognition.fly_behavior.dn_body import body_drive, agent_step_budget
    from cognition.fly_behavior.gf_global import should_cancel_output, clear_interrupt

    drv = FlyDN().drive(energy=0.9, decisiveness=0.8, affect=0.3)
    report["checks"].append({"name": "dn_drive", "drv": drv})
    for k in ("expression_intensity", "gesture_intensity", "gaze_speed", "action_vigor"):
        if k not in drv:
            report["ok"] = False
            report["reason"] = f"missing_{k}"

    bd = body_drive(user_id="stage6-ab")
    report["checks"].append({"name": "body", "bd": bd})
    report["checks"].append({"name": "budget", "n": agent_step_budget(user_id="stage6-ab", base=8)})
    report["checks"].append({"name": "cancel", "v": should_cancel_output("stage6-ab")})
    clear_interrupt("stage6-ab")

    print(json.dumps(report, default=str))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
