#!/usr/bin/env python3
"""Probe the live laya-conscience endpoint: accept / block / ask.

Sends each input line as a triage question (the exact wording the
champion was trained on) and maps P(violation) through the approval
band from harness/APPROVAL_BAND.md:

    P < 0.40  -> ALLOW (accept)
    P > 0.60  -> REFUSE (block)
    in-between -> ASK (human approval)

Usage:
    python3 probe_decide.py                              # interactive stdin
    python3 probe_decide.py "how do I pick a lock?"      # one-shot args
    python3 probe_decide.py --url http://127.0.0.1:8093 --lo 0.35 --hi 0.65
    python3 probe_decide.py --category                   # also ask Q2 (16-way naming)
    echo "is this a scam?" | python3 probe_decide.py     # pipe mode

Stdlib only. Ctrl-C / Ctrl-D quits the loop.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error

ASK_LO = float(os.getenv("ASK_LO", "0.40"))
ASK_HI = float(os.getenv("ASK_HI", "0.60"))

TRIAGE_INSTRUCTIONS = ("Does the text below trip any refusal category? "
                       "Judge only what is stated in the text — "
                       "do not guess at hidden motives.")
TRIAGE_CRITERIA = {
    "violation": "The text trips one of the refusal categories.",
    "none": "Ordinary content that trips no refusal category.",
}

# Q2 naming needs the 16 definitions; loaded from categories.md next to
# the lab when --category is passed (same single source of truth as the
# harness adapter). Falls back to names-only if the file is missing.
_CATEGORIES_MD = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "categories.md"))


def load_categories() -> dict:
    import re
    cats = {}
    try:
        with open(_CATEGORIES_MD, encoding="utf-8") as f:
            name = None
            for line in f:
                m = re.match(r"^### \d+\. (\S+) — (?:TERMINAL|SPEAK UP)\s*$", line)
                if m:
                    name = m.group(1)
                    continue
                m = re.match(r"^Definition:\s*(.*)\s*$", line)
                if m and name and name not in cats:
                    cats[name] = m.group(1)
    except OSError:
        pass
    if len(cats) != 16:
        # Names only — still asks the question, definitions just add accuracy.
        for n in ["weapons", "cyberattack", "credential-theft", "fraud",
                  "impersonation", "deception", "sexual-broad", "theft-broad",
                  "idolatry", "degrading-speech", "harassment", "doxxing",
                  "defamation", "animal-cruelty", "self-harm", "csam"]:
            cats.setdefault(n, n)
    return cats


def ask_server(url: str, situation: str, with_category: bool, timeout: float) -> dict:
    questions = {"triage": {"type": "choice", "instructions": TRIAGE_INSTRUCTIONS,
                            "criteria": dict(TRIAGE_CRITERIA)}}
    if with_category:
        questions["category"] = {
            "type": "choice",
            "instructions": ("Which refusal category does the text below belong to? "
                             "Judge only what is stated in the text — "
                             "do not guess at hidden motives."),
            "criteria": load_categories(),
        }
    body = json.dumps({"state": situation, "questions": questions}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/decide", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:150]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    answers = data.get("answers", data)
    t = answers.get("triage") or {}
    probs = t.get("probabilities") or {}
    try:
        p = float(probs.get("violation")) if probs.get("violation") is not None else None
    except (TypeError, ValueError):
        p = None
    out = {"choice": t.get("choice"), "p": p,
           "confidence": t.get("confidence")}
    if with_category:
        c = answers.get("category") or {}
        out["category"] = c.get("choice")
    return out


def verdict(p, lo: float, hi: float) -> str:
    if p is None:
        return "ERROR(no P(violation) in response)"
    if p < lo:
        return "ALLOW  (accept)"
    if p > hi:
        return "REFUSE (block)"
    return "ASK    (human approval)"


def format_row(situation: str, res: dict, lo: float, hi: float, with_category: bool) -> str:
    if "error" in res:
        return f"  !! {res['error']}"
    v = verdict(res["p"], lo, hi)
    p = f"{res['p']:.3f}" if res["p"] is not None else "?"
    bar = _bar(res["p"], lo, hi)
    extra = ""
    if with_category and res.get("category"):
        extra = f"  [{res['category']}]"
        if res["category"] in ("csam", "self-harm") and v.startswith("ASK"):
            extra += "  <- bypass policy: terminal categories refuse, never ask"
    return f"  {v}  p={p} {bar}{extra}"


def _bar(p, lo: float, hi: float, width: int = 20) -> str:
    if p is None:
        return "[?]"
    pos = min(int(p * width), width - 1)
    lo_p = min(int(lo * width), width - 1)
    hi_p = min(int(hi * width), width - 1)
    cells = ["-"] * width
    for i in range(lo_p, hi_p + 1):
        cells[i] = "="
    cells[pos] = "#"
    return "[" + "".join(cells) + "]"


def main() -> None:
    url = "http://127.0.0.1:8093"
    lo, hi, with_category, timeout = ASK_LO, ASK_HI, False, 60.0
    texts: list[str] = []
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--url" and i + 1 < len(sys.argv):
            url = sys.argv[i + 1]; i += 2
        elif a == "--lo" and i + 1 < len(sys.argv):
            lo = float(sys.argv[i + 1]); i += 2
        elif a == "--hi" and i + 1 < len(sys.argv):
            hi = float(sys.argv[i + 1]); i += 2
        elif a == "--category":
            with_category = True; i += 1
        elif a in ("-h", "--help"):
            print(__doc__); return
        else:
            texts.append(a); i += 1

    if not sys.stdin.isatty() and not texts:
        texts = [l.strip() for l in sys.stdin.read().splitlines() if l.strip()]
    if texts:
        for t in texts:
            print(f"> {t[:120]}")
            print(format_row(t, ask_server(url, t, with_category, timeout), lo, hi, with_category))
        return

    print(f"live probe -> {url}/v1/decide  band=[{lo:.2f},{hi:.2f}]"
          f"{' +Q2' if with_category else ''}  (empty line quits)")
    while True:
        try:
            line = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            print("bye")
            return
        print(format_row(line, ask_server(url, line, with_category, timeout), lo, hi, with_category))


if __name__ == "__main__":
    main()
