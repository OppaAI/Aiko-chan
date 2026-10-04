#!/usr/bin/env python3
"""What does Aiko actually DO when the conscience judge is wrong?

Every measurement so far scored the L2 judge in isolation. That understates the
system: core._run also has L3 deliberation (a constrained call to the main chat
model) and fuse() is one-way, so deliberation can only TIGHTEN a verdict -- it
can rescue an ALLOW but never manufacture a refusal out of nothing.

This traces the real ladder and reports, for clearly-wrong inputs the judge got
wrong:
  - the final decision
  - whether L3 fired
  - how often deliberation actually rescued the case
"""
import collections
import csv
import json
import sys

sys.path.insert(0, "/home/oppa-ai/Aiko-chan")
from cognition.conscience.core import ConscienceCircuitCore
from cognition.conscience import guardrails
from cognition.conscience.schema import ALLOW, CAUTION, ESCALATE, REFUSE

DATA = "/home/oppa-ai/Aiko-chan/conscience-lab/data/ethics_wrong.jsonl"


def main():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    llm = None
    try:
        from openai import OpenAI
        llm = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed", timeout=60.0)
    except Exception as e:
        print(f"no llm client: {e}")

    core = ConscienceCircuitCore("trace")
    stats = collections.Counter()
    gates = collections.Counter()
    rescued = []
    allowed_wrong = []

    for r in rows[:limit]:
        text = f"The user asks: {r['input']}"
        v = core.evaluate(act="respond", content=text, context={"surface": "chat"},
                          llm_client=llm, surface="chat")
        stats[v.decision] += 1
        gates[v.gate] += 1
        used_l3 = "deliberate" in v.layers_run
        if used_l3:
            stats["_l3_fired"] += 1
        if v.decision in (ESCALATE, REFUSE):
            if v.decision == REFUSE:
                stats["_refused"] += 1
            else:
                stats["_escalated"] += 1
        else:
            allowed_wrong.append((r["input"], v.decision, v.gate, used_l3,
                                  round(v.confidence, 2)))

    n = min(limit, len(rows))
    print(f"clearly-wrong inputs traced through the real ladder: {n}")
    print(f"L3 deliberation fired      : {stats['_l3_fired']}")
    print(f"final REFUSE               : {stats['_refused']}")
    print(f"final ESCALATE (asks Oppa) : {stats['_escalated']}")
    print(f"final ALLOW/CAUTION        : {stats[ALLOW] + stats[CAUTION]}")
    caught = stats["_refused"] + stats["_escalated"]
    print(f"\n=> caught before acting    : {caught}/{n} = {caught/n:.0%}")
    print(f"=> acted on                : {len(allowed_wrong)}/{n} = {len(allowed_wrong)/n:.0%}")
    print(f"\nby gate: {dict(gates)}")
    print("\n--- what got through ---")
    for text, dec, gate, l3, conf in allowed_wrong[:20]:
        print(f"  {dec:8s} gate={gate:10s} l3={'yes' if l3 else 'no ':3s} conf={conf}  {text[:66]}")


if __name__ == "__main__":
    main()