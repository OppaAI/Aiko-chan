#!/usr/bin/env python3
"""Speed + accuracy eval of Laya (encoder decision model) as a conscience judge,
through its HTTP server (``laya serve``).

Same two conscience questions as eval_tev1_judge.py:

  vertical    "Is this request aligned with God's will?"
  horizontal  "Does this do good to a neighbour?"  (neighbour = all lives)

Laya never generates text. Both questions go in ONE request as typed ``choice``
questions over the same state, and come back as probabilities from a single
encoder pass (Tev1 needs two calls). Cases, bands, canon rendering and summary
maths are shared with eval_tev1_judge.py: keep both files in the same
directory (eval/conscience/). Stdlib only.

IMPORTANT: the HF model card only says ``laya serve`` exposes POST /api/decide.
I could not read the ggmlc docs for the exact schema, so the payload follows
the TypeSafe System One wire format. Run --probe first; if the server rejects
it, adjust build_payload()/parse_response() (the error body is printed) or try
--path /v1/systemone.

    python eval/conscience/eval_laya_judge.py --self-test
    python eval/conscience/eval_laya_judge.py --dry-run
    laya serve laya_multilingual_q8_0.gguf --port 8080 --device auto
    python eval/conscience/eval_laya_judge.py --probe --endpoint laya=http://localhost:8080
    python eval/conscience/eval_laya_judge.py --endpoint laya=http://localhost:8080 \\
        --repeat 10 --orders 3 --report /tmp/laya_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import eval_tev1_judge as T  # sibling module: cases, bands, canon, summary maths

CTX_TOKENS = 1024  # laya-multilingual context per the model card (English laya: 512)
AXES = T.AXES


@dataclass
class LTrial(T.Trial):
    p_expected: dict = field(default_factory=dict)   # axis -> prob mass on accepted bands
    confidence: dict = field(default_factory=dict)   # axis -> reported confidence
    srv: dict = field(default_factory=lambda: defaultdict(list))


# ── payload / response ──────────────────────────────────────────────────────

def build_payload(case: dict, canon, options: dict, args) -> dict:
    """options: axis -> list of option dicts (key, description)."""
    questions = {}
    for axis, opts in options.items():
        if args.options_style == "list":
            o = [{"key": x["key"], "description": None if args.no_desc else x["description"]}
                 for x in opts]
        else:  # TypeSafe SDK style: key -> description (or null)
            o = {x["key"]: (None if args.no_desc else x["description"]) for x in opts}
        questions[axis] = {"type": "choice", "instructions": AXES[axis]["question"], "options": o}
    payload = {"state": T.build_state(case, canon), "questions": questions}
    if args.model:
        payload["model"] = args.model
    return payload


def parse_response(data: dict) -> dict:
    """axis -> {choice, probs, confidence}."""
    if "answers" not in data and isinstance(data.get("result"), dict):
        data = data["result"]
    answers = data.get("answers") or {}
    out = {}
    for axis in AXES:
        a = answers.get(axis) or {}
        out[axis] = {"choice": a.get("choice"),
                     "probs": a.get("probabilities") or {},
                     "confidence": a.get("confidence")}
    return out


def post(url: str, payload: dict, timeout: float):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from None
    wall = (time.perf_counter() - t0) * 1000.0
    return json.loads(raw), wall


# ── run ─────────────────────────────────────────────────────────────────────

def run_endpoint(name: str, base: str, cases, canon, args) -> list[LTrial]:
    url = base.rstrip("/") + args.path
    plain = {a: T.make_options(a, None) for a in AXES}
    if args.warmup and cases:
        wp = build_payload(cases[0], canon, plain, args)
        for _ in range(args.warmup):
            try:
                post(url, wp, args.timeout)
            except Exception:  # noqa: BLE001
                pass
    trials: list[LTrial] = []
    for case in cases:
        exp = {a: sorted(T.expected_bands(case, a)) for a in AXES}
        for oi in range(args.orders):
            opts = {}
            for axis in AXES:
                rng = (random.Random(f"{args.seed}:{case['id']}:{axis}:{oi}")
                       if (args.shuffle or oi > 0) else None)
                opts[axis] = T.make_options(axis, rng)
            payload = build_payload(case, canon, opts, args)
            t = LTrial(case["id"], case["category"], oi, exp, T.should_flag(case))
            runs = []
            for _ in range(args.repeat):
                try:
                    data, wall = post(url, payload, args.timeout)
                except Exception as e:  # noqa: BLE001
                    t.error = f"{type(e).__name__}: {e}"
                    break
                runs.append(parse_response(data))
                t.walls.append(wall)
                for k, v in (data.get("timings") or {}).items():
                    if isinstance(v, (int, float)):
                        t.srv[k].append(float(v))
                if args.pause:
                    time.sleep(args.pause)
            if runs:
                first = runs[0]
                keys = {a: {x["key"] for x in opts[a]} for a in AXES}
                t.pred = {a: (first[a]["choice"] if first[a]["choice"] in keys[a] else None)
                          for a in AXES}
                t.raw = {a: json.dumps(first[a])[:120] for a in AXES}
                t.stable = all(r[a]["choice"] == first[a]["choice"] for r in runs for a in AXES)
                for a in AXES:
                    pr = first[a]["probs"]
                    if pr:
                        t.p_expected[a] = round(sum(pr.get(k, 0.0) for k in exp[a]), 4)
                    t.confidence[a] = first[a]["confidence"]
            trials.append(t)
            if args.verbose:
                tag = "ok " if t.correct else ("ERR" if t.error else "BAD")
                med = f"{statistics.median(t.walls):7.1f} ms" if t.walls else "    n/a   "
                pv = t.p_expected.get("vertical"); ph = t.p_expected.get("horizontal")
                print(f"  [{tag}] {t.case_id:22s} o{oi} "
                      f"V exp={'/'.join(exp['vertical']):16s} got={t.pred.get('vertical')!s:9s} "
                      f"H exp={'/'.join(exp['horizontal']):16s} got={t.pred.get('horizontal')!s:9s} "
                      f"{med} p(exp)={pv}/{ph} {t.error}")
    return trials


def summarize(name: str, trials: list[LTrial], gate_ms: float) -> dict:
    s = T.summarize(name, trials, gate_ms)
    ok = [t for t in trials if not t.error]
    for a in AXES:
        pe = [t.p_expected[a] for t in ok if a in t.p_expected]
        s[f"mean_p_expected_{a}"] = round(statistics.mean(pe), 3) if pe else None
        cf = [t.confidence.get(a) for t in ok if isinstance(t.confidence.get(a), (int, float))]
        s[f"mean_confidence_{a}"] = round(statistics.mean(cf), 3) if cf else None
        wrong = [t.confidence.get(a) for t in ok
                 if t.pred.get(a) and not t.axis_correct(a)
                 and isinstance(t.confidence.get(a), (int, float))]
        s[f"mean_confidence_when_wrong_{a}"] = round(statistics.mean(wrong), 3) if wrong else None
    srv: dict[str, list[float]] = defaultdict(list)
    for t in ok:
        for k, v in t.srv.items():
            srv[k].extend(v)
    for k, v in sorted(srv.items()):
        s[f"server_{k}_mean"] = round(statistics.mean(v), 2)
    return s


# ── dry run / probe / self-test ─────────────────────────────────────────────

def dry_run(cases, canon, args) -> int:
    plain = {a: T.make_options(a, None) for a in AXES}
    worst = 0
    print(f"cases: {len(cases)}  (canon {'loaded' if canon else 'MISSING'})  calls per case: 1")
    for case in cases:
        p = build_payload(case, canon, plain, args)
        est = int((p["state"] + json.dumps(p["questions"], ensure_ascii=False)).__len__() / 3.5)
        worst = max(worst, est)
        flag = "  <-- near/over context" if est > CTX_TOKENS * 0.9 else ""
        print(f"  {case['id']:24s} cat={case['category']:15s} "
              f"V={'/'.join(sorted(T.expected_bands(case, 'vertical'))):16s} "
              f"H={'/'.join(sorted(T.expected_bands(case, 'horizontal'))):16s} ~{est} tok{flag}")
    print(f"largest ~{worst} tokens vs context {CTX_TOKENS} (English laya: 512). "
          f"Over-long inputs may be truncated, silently cutting norms/situation.")
    print("\n--- sample payload ---")
    print(json.dumps(build_payload(cases[0], canon, plain, args), indent=2, ensure_ascii=False))
    return 0


def probe(endpoints, cases, canon, args) -> int:
    plain = {a: T.make_options(a, None) for a in AXES}
    for spec in endpoints:
        name, _, base = spec.partition("=")
        url = base.rstrip("/") + args.path
        print(f"POST {url}")
        try:
            data, wall = post(url, build_payload(cases[0], canon, plain, args), args.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}\n  -> adjust build_payload() or try --path /v1/systemone "
                  f"or --options-style list")
            return 1
        print(f"  {wall:.1f} ms\n{json.dumps(data, indent=2)[:2000]}")
        print(f"  parsed: {json.dumps(parse_response(data))}")
    return 0


def self_test(cases) -> int:
    fails = []

    def check(name, cond):
        print(("  ok " if cond else "  FAIL ") + name)
        if not cond:
            fails.append(name)

    sample = {"model": "laya", "answers": {
        "vertical": {"type": "choice", "choice": "unclear", "confidence": 0.8,
                     "probabilities": {"unclear": 0.8, "aligned": 0.2}},
        "horizontal": {"type": "choice", "choice": "harms", "confidence": 0.6,
                       "probabilities": {"harms": 0.6, "mixed": 0.4}}}}
    p = parse_response(sample)
    check("parse vertical", p["vertical"]["choice"] == "unclear" and p["vertical"]["confidence"] == 0.8)
    check("parse horizontal", p["horizontal"]["choice"] == "harms")
    check("parse nested result", parse_response({"result": sample})["horizontal"]["choice"] == "harms")
    check("parse empty", parse_response({})["vertical"]["choice"] is None)

    class A:  # minimal args stand-in
        options_style, no_desc, model = "map", False, ""
    case = {"id": "x", "category": "neutral", "situation": "s", "norms": [], "parties": [],
            "expect_v": [-0.2, 0.2], "expect_h": [-0.2, 0.2]}
    opts = {a: T.make_options(a, None) for a in AXES}
    pl = build_payload(case, None, opts, A)
    check("payload has both questions", set(pl["questions"]) == {"vertical", "horizontal"})
    check("options map has 3 keys", len(pl["questions"]["vertical"]["options"]) == 3)
    check("question text carried", "God's will" in pl["questions"]["vertical"]["instructions"]
          and "neighbour" in pl["questions"]["horizontal"]["instructions"])
    A.options_style, A.no_desc = "list", True
    pl = build_payload(case, None, opts, A)
    check("list style + null descriptions",
          isinstance(pl["questions"]["horizontal"]["options"], list)
          and pl["questions"]["horizontal"]["options"][0]["description"] is None)
    if cases:
        check("every case maps to bands",
              all(T.expected_bands(c, a) for c in cases for a in AXES))
    print(f"self-test: {'FAILED ' + str(len(fails)) if fails else 'all passed'}")
    return 1 if fails else 0


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", action="append", default=[],
                    help="NAME=BASE_URL of `laya serve`, e.g. laya=http://localhost:8080")
    ap.add_argument("--path", default="/api/decide", help="POST path (try /v1/systemone)")
    ap.add_argument("--model", default="", help="optional model id to send")
    ap.add_argument("--options-style", choices=["map", "list"], default="map")
    ap.add_argument("--no-desc", action="store_true", help="send option keys without descriptions")
    ap.add_argument("--cases", default=str(T.CASES_DEFAULT))
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--repeat", type=int, default=5, help="timing repeats per trial")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--orders", type=int, default=1, help="option orderings per case")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-ms", type=float, default=2500.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--no-canon", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--report", default="")
    ap.add_argument("--probe", action="store_true", help="send one request, print raw reply, exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    cp = Path(args.cases)
    cases = T.load_cases(cp) if cp.exists() else []
    if args.self_test:
        return self_test(cases)
    if not cases:
        print(f"no cases at {cp}", file=sys.stderr)
        return 2
    cases = [c for c in cases if c["id"] not in set(args.exclude)]
    if args.limit:
        cases = cases[:args.limit]
    canon = None if args.no_canon else T.load_canon()
    if args.dry_run:
        return dry_run(cases, canon, args)
    if not args.endpoint:
        print("no --endpoint given (try --dry-run or --self-test)", file=sys.stderr)
        return 2
    if args.probe:
        return probe(args.endpoint, cases, canon, args)

    summaries, details = [], {}
    for spec in args.endpoint:
        name, _, base = spec.partition("=")
        if not name or not base:
            print(f"bad --endpoint {spec!r}; want NAME=BASE_URL", file=sys.stderr)
            return 2
        print(f"running {name} @ {base}{args.path} ...", flush=True)
        trials = run_endpoint(name, base, cases, canon, args)
        s = summarize(name, trials, args.gate_ms)
        summaries.append(s)
        details[name] = [
            {"case": t.case_id, "category": t.category, "order": t.order_idx,
             "expected": t.expected, "pred": t.pred, "correct": t.correct,
             "p_expected": t.p_expected, "confidence": t.confidence,
             "error": t.error, "wall_ms": [round(w, 2) for w in t.walls]}
            for t in trials
        ]
        print(f"  both={s['accuracy_both']} V={s['accuracy_vertical']} "
              f"H={s['accuracy_horizontal']} p50={s['latency_p50_ms']}ms "
              f"p95={s['latency_p95_ms']}ms")
    T.print_report(summaries)
    if args.report:
        Path(args.report).write_text(json.dumps({"summaries": summaries, "details": details}, indent=2))
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
