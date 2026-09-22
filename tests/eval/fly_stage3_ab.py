"""Stage 3 A/B: Fly OFF vs LIVE for teach → rank and GF interrupt.

Run:
  MEMORY_FLYMB_MODE=off MEMORY_FLYGF_MODE=off python -m tests.eval.fly_stage3_ab
  MEMORY_FLYMB_MODE=live MEMORY_FLYGF_MODE=live python -m tests.eval.fly_stage3_ab
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid


def main() -> int:
    # Hermetic: redirect user state to a temp dir with a fresh uid so eval
    # runs never pollute the live ~/.aiko store (learned avoids would
    # otherwise leak into real recall ranking).
    tmp = tempfile.mkdtemp(prefix="fly-stage3-ab-")
    for key in ("USER_STATE_ROOT", "AIKO_USER_STATE_ROOT", "USER_SPACE_ROOT"):
        os.environ[key] = tmp
    try:
        from system import userspace

        def _tmp_state(user_id=None):
            import pathlib
            p = pathlib.Path(tmp) / (str(user_id or "default"))
            p.mkdir(parents=True, exist_ok=True)
            return str(p)

        userspace.user_state_dir = _tmp_state  # type: ignore[attr-defined]
    except Exception:
        pass

    from cognition.flymemory.teach_api import teach_preference
    from cognition.memory.fly_rank import adjust_recall_score
    from cognition.memory.preference_store import preference_delta, record_preference
    from cognition.fly_behavior.giant_fiber import assess_interrupt

    uid = f"stage3-ab-{uuid.uuid4().hex[:8]}"
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

    mb_live = modes.get("MEMORY_FLYMB_MODE") == "live"
    gf_live = modes.get("MEMORY_FLYGF_MODE") == "live"
    report["checks"].append({"name": "teach", "pref": pref})
    report["checks"].append({"name": "rank", "tart": tart_score, "other": other_score, "meta": tart_meta, "pdelta": pdelta})
    report["checks"].append({"name": "gf", "gf": gf})

    if mb_live:
        if tart_score >= other_score:
            report["ok"] = False
            report["reason"] = "live avoid did not suppress tart vs other"
        if pdelta >= 0:
            report["ok"] = False
            report["reason"] = "live avoid preference delta was not negative"
    if gf_live and not (gf.get("interrupt") or float(gf.get("urgency") or 0) >= 0.5):
        report["ok"] = False
        report["reason"] = "live GF did not flag STOP"
    print(json.dumps(report, default=str))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
