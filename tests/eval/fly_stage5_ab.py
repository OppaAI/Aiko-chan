"""Stage 5 smoke: semantic features, CX topic, LH method.

Run:
  MEMORY_FLYCX_MODE=live MEMORY_FLYLH_MODE=live python -m tests.eval.fly_stage5_ab
"""
from __future__ import annotations

import json
import os


def main() -> int:
    os.environ.setdefault("MEMORY_FLYCX_MODE", "live")
    os.environ.setdefault("MEMORY_FLYLH_MODE", "live")
    report: dict = {"ok": True, "checks": []}

    from cognition.fly_behavior.semantic_features import (
        heading_from_features,
        semantic_features,
    )
    from cognition.fly_behavior.cx_topic import apply_topic_drive
    from cognition.fly_behavior.lateral_horn import context_prior
    from cognition.fly_behavior.cx_features import blend_cx_features

    uid = "stage5-ab"
    f1 = semantic_features("fruit tarts for my birthday", user_id=uid)
    f2 = semantic_features("birthday pastry dessert", user_id=uid)
    h1 = heading_from_features(f1)
    h2 = heading_from_features(f2)
    report["checks"].append({"name": "features", "h1": h1, "h2": h2, "n": len(f1)})

    topic = apply_topic_drive("tell me about the shogi game", user_id=uid)
    report["checks"].append({"name": "cx_topic", "topic": topic})
    if topic.get("reason") == "observe_only":
        report["ok"] = False
        report["reason"] = "still_observe_only"

    context_prior("garden roses in the morning", user_id=uid, record=True)
    p2 = context_prior("garden roses in the morning", user_id=uid, record=True)
    report["checks"].append({"name": "lh", "prior": p2})

    affect = [0.1] * 8
    feats, pen, fat = blend_cx_features(affect, 0.2, 0.1, "hello", uid)
    report["checks"].append({"name": "blend", "feats0": feats[0], "pen": pen})

    print(json.dumps(report, default=str))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
