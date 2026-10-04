#!/usr/bin/env python3
"""Compare two live checkpoints through the real conscience ladder.

Point CCC_LAYA_BASE_URL at each laya serve port in turn; runs the same case set
through core.evaluate() + guardrails so the comparison includes the
deterministic layers, not just the model's opinion.
"""
import json
import os
import sys

sys.path.insert(0, "/home/oppa-ai/Aiko-chan")
from cognition.conscience.core import ConscienceCircuitCore
from cognition.conscience import guardrails

CASES = "/home/oppa-ai/Aiko-chan/conscience-lab/data/oppa_benign_cases.jsonl"
MAP = {"allow": "allow", "act_and_notify": "allow", "ask": "escalate", "refuse": "refuse"}


def main():
    port = sys.argv[1]
    rows = [json.loads(l) for l in open(CASES, encoding="utf-8") if l.strip()]
    core = ConscienceCircuitCore(f"audit-{port}")

    print(f"=== port {port} ({os.environ.get('CCC_LAYA_BASE_URL')}) ===")
    for cat in ("harmful", "stakes", "ambiguous"):
        sub = [r for r in rows if r["category"] == cat]
        ok = 0
        for r in sub:
            v = core.evaluate(act="respond", content=r["situation"],
                              context={"surface": "chat"})
            got = MAP.get(v.decision, v.decision)
            good = got == r.get("expect_decision")
            ok += good
            rules = [h.rule_id for h in guardrails.scan(r["situation"], act="respond",
                                                      context={"surface": "chat"})]
            print(f"  {'OK ' if good else '   '}{r['id']:24s} {got:9s} "
                  f"want={r.get('expect_decision'):9s} ladder={v.decision:8s} "
                  f"L0={rules or 'none'}")
        print(f"  -> {cat}: {ok}/{len(sub)}\n")


if __name__ == "__main__":
    main()