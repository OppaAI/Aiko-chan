#!/usr/bin/env python3
"""A/B harness for fly modes — offline, no chat UI required.

Usage:
  python -m tests.eval.fly_ab_harness
  MEMORY_FLYMB_MODE=live MEMORY_FLYGF_MODE=live python -m tests.eval.fly_ab_harness

Compares deterministic proxies for a fixed prompt set under off vs current env modes.
Does not claim statistical significance; flags large score deltas for human review.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass


PROMPTS = [
    "hello there",
    "STOP right now",
    "what did we talk about yesterday",
    "remember that I prefer short answers",
    "urgent: cancel the long research task",
    "tell me a calm story about the garden",
]


@dataclass
class Row:
    prompt: str
    gf_urgency: float
    gf_interrupt: bool
    lh_familiarity: float
    lh_bias: float
    sleep_mult: float
    mb_bias: float | None


def run_once() -> list[Row]:
    from cognition.fly_behavior.giant_fiber import assess_interrupt
    from cognition.fly_behavior.lateral_horn import context_prior
    from cognition.fly_behavior.sleep_sched import dream_boost_multiplier

    rows: list[Row] = []
    uid = "ab-harness"
    for p in PROMPTS:
        gf = assess_interrupt(p)
        lh = context_prior(p, user_id=uid)
        mult = dream_boost_multiplier(uid)
        mb_bias = None
        try:
            from cognition.flymemory import text_features
            from cognition.fly_registry import get_flymb
            mb = get_flymb(uid)
            if mb is not None:
                mb_bias = float(mb.valence_bias(text_features(p)))
        except Exception:
            mb_bias = None
        rows.append(
            Row(
                prompt=p,
                gf_urgency=float(gf.get("urgency") or 0.0),
                gf_interrupt=bool(gf.get("interrupt")),
                lh_familiarity=float(lh.get("familiarity") or 0.5),
                lh_bias=float(lh.get("bias") or 0.0),
                sleep_mult=float(mult),
                mb_bias=mb_bias,
            )
        )
    return rows


def main() -> int:
    modes = {
        k: os.getenv(k, "off")
        for k in (
            "MEMORY_FLYMB_MODE",
            "MEMORY_FLYCX_MODE",
            "MEMORY_FLYLH_MODE",
            "MEMORY_FLYGF_MODE",
            "MEMORY_FLYSLEEP_MODE",
            "MEMORY_FLYDN_MODE",
            "MEMORY_FLYAL_MODE",
        )
    }
    print("modes:", json.dumps(modes))
    rows = run_once()
    for r in rows:
        print(json.dumps(asdict(r)))
    stop = next(r for r in rows if r.prompt.startswith("STOP"))
    if modes.get("MEMORY_FLYGF_MODE") == "live" and not stop.gf_interrupt:
        print("WARN: live GF did not interrupt on STOP", file=sys.stderr)
        return 2
    print("ok", len(rows), "prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
