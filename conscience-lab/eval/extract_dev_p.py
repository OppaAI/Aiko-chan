#!/usr/bin/env python3
"""Extract per-row Q1 violation probabilities from a harness per_case.jsonl.

The harness's `triage_confidence` is the confidence in the CHOSEN triage
label, not P(violation). Convert:

    p = triage_confidence            if triage_choice == "violation"
    p = 1 - triage_confidence        if triage_choice == "none"

Feeding triage_confidence straight into analyze_gray.py would put confident
"allow" rows at HIGH p and completely invert the allow side of the analysis.

Usage:
    python3 extract_dev_p.py <per_case.jsonl> <out_scores.jsonl>

Output: one {"id": ..., "p": ...} per line, p in [0,1].
Requires the two-question-mode fields (triage_choice / triage_confidence).
"""
import json
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: extract_dev_p.py <per_case.jsonl> <out_scores.jsonl>")
    per_path, out_path = sys.argv[1], sys.argv[2]
    n = 0
    with open(out_path, "w") as f:
        for line in open(per_path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tc = r.get("triage_choice")
            conf = r.get("triage_confidence")
            if tc is None or conf is None:
                sys.exit(
                    f"row {r.get('id')}: missing triage fields -- "
                    "this per_case file was not scored in two-question mode"
                )
            p = conf if tc == "violation" else 1.0 - conf
            p = max(0.0, min(1.0, p))
            f.write(json.dumps({"id": r["id"], "p": round(p, 4)}) + "\n")
            n += 1
    print(f"wrote {n} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
