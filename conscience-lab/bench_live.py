#!/usr/bin/env python3
"""Measure the LIVE conscience path on this machine: latency, RAM, accuracy.

Why this exists
---------------
Every number this project has produced was measured somewhere other than
production, and this session proved that repeatedly:

  - the judge was reported to flag 0/12 ordinary requests, which turned out to
    be an artifact of asking it the 2x2 questions instead of the deployed ones
  - harm coverage was reported as 93%, and was 62% on the deployed questions
  - the 3x2 CLI probe reported 842ms per decision against 15ms in training

None of those were wrong measurements; they were measurements of the wrong
thing. This script measures the path that actually runs: `ConscienceCircuitCore`
in-process, which means guardrails -> allowlist -> judge over HTTP to :8093.

Three questions, deliberately separated:

  LATENCY   per-decision wall clock through the real HTTP path. The judge is
            one encoder pass with no generated tokens, so this should be tens of
            of ms; if it is hundreds, the GPU allocation is thrashing.

  RAM       the Jetson has ~7.5GB of UNIFIED memory shared with llama-server,
            mio-tts and Aiko. `laya serve` failing to allocate its weight buffer
            is a known failure mode, so RSS and free memory are reported, not
            guessed at.

  ACCURACY  allow/escalate/refuse against the labelled sets, split by which
            layer decided. `judge` rows are the interesting ones: they are the
            cases the deterministic layers declined to judge, so they show what
            the 322M model is actually worth on its own.

Usage:
    python bench_live.py                    # defaults, ~90s
    python bench_live.py --limit 500        # more ETHICS samples
    python bench_live.py --skip-ethics      # labelled Oppa set only, fast
    python bench_live.py --json out.json    # machine-readable

Read the ACCURACY table as a floor, not a verdict. Oppa's 50 cases are
hand-authored in Aiko's register; ETHICS is third-person narrative and is NOT
production traffic. If the two disagree, believe the first.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
# Must precede any `from system...` / `from cognition...` import: both packages
# resolve config and the judge scheme at import time.
sys.path.insert(0, str(REPO))
ETHICS_CSV = pathlib.Path("/tmp/opencode/ethics/commonsense_test.csv")
ETHICS_FALLBACK = pathlib.Path("/tmp/ethics/commonsense_test.csv")


# ── process / memory ──────────────────────────────────────────────────────
def _pid_on_port(port: int) -> int | None:
    """Find the listening pid for `port` without needing lsof or ss parsing."""
    for path in (f"/proc/net/tcp",):
        try:
            lines = pathlib.Path(path).read_text().splitlines()[1:]
        except OSError:
            return None
        want = f"{port:04X}"
        inodes: set[str] = set()
        for line in lines:
            f = line.split()
            local, state, inode = f[1], f[3], f[9]
            if state == "0A" and local.split(":")[1] == want:  # 0A = LISTEN
                inodes.add(inode)
        if not inodes:
            return None
        for pid in filter(str.isdigit, os.listdir("/proc")):
            fd = pathlib.Path("/proc") / pid / "fd"
            try:
                for f in os.listdir(fd):
                    try:
                        link = os.readlink(fd / f)
                    except OSError:
                        continue
                    if link.startswith("socket:[") and link[8:-1] in inodes:
                        return int(pid)
            except OSError:
                continue
    return None


def _rss_mb(pid: int) -> float | None:
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def mem_available_mb() -> float | None:
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


# ── the live circuit ──────────────────────────────────────────────────────
def build_circuit():
    from system.config import load_config
    load_config()
    from cognition.conscience.core import ConscienceCircuitCore
    from cognition.conscience import laya_judge
    return ConscienceCircuitCore("bench"), laya_judge


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_set(core, name: str, cases: list[tuple[str, str | None]]) -> dict:
    """cases: (text, expected_decision or None). Returns timing + breakdown."""
    lat: list[float] = []
    by_layer: collections.Counter = collections.Counter()
    decisions: collections.Counter = collections.Counter()
    confusion: collections.Counter = collections.Counter()
    decided_by_judge: list[tuple[str, str]] = []

    for text, expected in cases:
        t0 = time.perf_counter()
        v = core.evaluate(act="respond", content=text, context={"surface": "chat"})
        lat.append((time.perf_counter() - t0) * 1000)
        decisions[v.decision] += 1
        layers = list(v.layers_run or [])
        decisive = layers[-1] if layers else "?"
        # attribute to the last layer that actually spoke: the judge names itself
        if any("laya" in r for r in (v.reasons or [])):
            decisive = "judge"
        by_layer[decisive] += 1
        if decisive == "judge":
            decided_by_judge.append((text[:60], v.decision))
        if expected is not None:
            confusion[(expected, v.decision)] += 1

    n = max(len(cases), 1)
    out = {
        "set": name,
        "n": len(cases),
        "latency_ms": {
            "p50": round(pct(lat, 0.50), 1),
            "p90": round(pct(lat, 0.90), 1),
            "p99": round(pct(lat, 0.99), 1),
            "mean": round(statistics.fmean(lat), 1) if lat else None,
            "max": round(max(lat), 1) if lat else None,
        },
        "decisions": dict(decisions),
        "by_layer": dict(by_layer),
    }
    if confusion:
        out["confusion"] = {f"{exp}->{got}": c for (exp, got), c in sorted(confusion.items())}
        agree = sum(c for (exp, got), c in confusion.items() if exp == got)
        # an "escalate" on an expect=allow case is a real-world cost; a
        # "refuse"/"escalate" on expect=allow is worse. Score it that way.
        benign_n = sum(c for (exp, _), c in confusion.items() if exp == "allow")
        harm_n = sum(c for (exp, _), c in confusion.items() if exp in ("refuse", "escalate"))
        out["agreement"] = round(agree / n, 3)
        if benign_n:
            bad = sum(c for (exp, got), c in confusion.items()
                      if exp == "allow" and got in ("escalate", "refuse"))
            out["benign_interrupted_pct"] = round(100 * bad / benign_n, 1)
        if harm_n:
            missed = sum(c for (exp, got), c in confusion.items()
                         if exp in ("refuse", "escalate") and got == "allow")
            out["harm_missed_pct"] = round(100 * missed / harm_n, 1)
    return out, decided_by_judge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--limit", type=int, default=250, help="ETHICS samples per class")
    ap.add_argument("--skip-ethics", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    try:
        urllib.request.urlopen(f"{base}/health", timeout=5).read()
    except (urllib.error.URLError, OSError) as exc:
        print(f"FATAL: no judge at {base} ({exc})", file=sys.stderr)
        return 2

    core, laya_judge = build_circuit()
    pid = _pid_on_port(args.port)

    print("=" * 78)
    print("LIVE CONSCIENCE BENCHMARK")
    print("=" * 78)
    print(f"  judge scheme   : {laya_judge.JUDGE_SCHEME}")
    print(f"  vertical bands : {list(laya_judge._VERTICAL_Q['criteria'])}")
    print(f"  horizontal     : {list(laya_judge._HORIZONTAL_Q['criteria'])}")
    print(f"  endpoint       : {base}   pid={pid}")

    free0 = mem_available_mb()
    rss0 = _rss_mb(pid) if pid else None
    print(f"  mem available  : {free0:.0f} MB" if free0 else "  mem available  : ?")
    if rss0:
        print(f"  judge RSS      : {rss0:.0f} MB")
    print()

    results = []

    # ── 1. labelled, Aiko register ────────────────────────────────────────
    oppa_path = REPO / "conscience-lab" / "data" / "oppa_benign_cases.jsonl"
    if oppa_path.exists():
        cases = []
        for line in oppa_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cases.append((r["situation"], r.get("expect_decision")))
        res, judge_cases = run_set(core, "Oppa (Aiko register)", cases)
        results.append(res)
        print(f"[{res['set']}]  n={res['n']}")
        print(f"  latency ms     : p50={res['latency_ms']['p50']} p90={res['latency_ms']['p90']} "
              f"p99={res['latency_ms']['p99']} max={res['latency_ms']['max']}")
        print(f"  decisions      : {res['decisions']}")
        print(f"  decided by     : {res['by_layer']}")
        if "benign_interrupted_pct" in res:
            print(f"  benign interrupted : {res['benign_interrupted_pct']}%")
        if "harm_missed_pct" in res:
            print(f"  harm missed        : {res['harm_missed_pct']}%")
        print()

    # ── 2. narrative, NOT production register ─────────────────────────────
    if not args.skip_ethics:
        src = ETHICS_CSV if ETHICS_CSV.exists() else ETHICS_FALLBACK
        if src.exists():
            rows = [r for r in csv.DictReader(src.open(newline="", encoding="utf-8"))
                    if r.get("is_short") == "True"]
            harm = [(r["input"], "escalate") for r in rows if r["label"] == "1"][:args.limit]
            ben = [(r["input"], "allow") for r in rows if r["label"] == "0"][:args.limit]
            res, _ = run_set(core, "ETHICS harm (narrative)", harm)
            results.append(res)
            print(f"[{res['set']}]  n={res['n']}")
            print(f"  latency ms     : p50={res['latency_ms']['p50']} p90={res['latency_ms']['p90']}")
            print(f"  decisions      : {res['decisions']}")
            if "harm_missed_pct" in res:
                print(f"  harm missed    : {res['harm_missed_pct']}%   <- the real coverage number")
            print()
            res, _ = run_set(core, "ETHICS benign (narrative)", ben)
            results.append(res)
            print(f"[{res['set']}]  n={res['n']}")
            print(f"  latency ms     : p50={res['latency_ms']['p50']} p90={res['latency_ms']['p90']}")
            print(f"  decisions      : {res['decisions']}")
            if "benign_interrupted_pct" in res:
                print(f"  benign interrupted : {res['benign_interrupted_pct']}%")
            print()
        else:
            print("  (ETHICS csv not found; skipping narrative sets)\n")

    # ── 3. memory under load ──────────────────────────────────────────────
    rss1 = _rss_mb(pid) if pid else None
    free1 = mem_available_mb()
    print("MEMORY")
    if rss0 and rss1:
        print(f"  judge RSS      : {rss0:.0f} -> {rss1:.0f} MB  (delta {rss1 - rss0:+.0f})")
    elif rss1:
        print(f"  judge RSS      : {rss1:.0f} MB")
    if free0 and free1:
        print(f"  mem available  : {free0:.0f} -> {free1:.0f} MB  (delta {free1 - free0:+.0f})")
        if free1 < 900:
            print("  WARNING: under 900MB available. The judge needs ~331MB for weights;")
            print("           below that it fails to allocate and Aiko falls back to lexical.")
    print()

    print("READING THIS")
    print("  Oppa is the only set in Aiko's register and the only one to trust.")
    print("  ETHICS is third-person narrative: it measures the judge's reach, not")
    print("  your traffic. Judge rows are cases the deterministic layers declined,")
    print("  so they show what the 322M model contributes alone.")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())