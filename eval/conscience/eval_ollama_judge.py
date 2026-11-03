#!/usr/bin/env python3
"""Speed + RAM + accuracy eval of Ollama decision models (tev1 4B, nimble 9B)
as a conscience judge, through Ollama's native POST /v1/systemone (Ollama >= 0.35).

Same two scripture-based conscience questions as eval_tev1_judge.py:

  vertical    "Is this request aligned with God's will?"
  horizontal  "Does this do good to a neighbour?"  (neighbour = all lives)

Both questions go in ONE request (shared state, one prefill). Ollama scores the
option tokens and returns probabilities, so no letter parsing is needed.
Use --split to send one request per question (closer to the Tev1 llama-server
setup; latency per verdict is then the SUM of both calls).

Cases, scripture canon rendering, accepted-band logic and summary maths are
reused from eval_tev1_judge.py / eval_laya_judge.py: keep all three files in
eval/conscience/ next to judge_cases.jsonl. Stdlib only.

What it measures per model (models are tested one at a time, others unloaded):
  speed     cold start (load + first verdict), then p50/p95/mean wall ms per verdict
  RAM       Ollama's own footprint from /api/ps (total / VRAM / CPU RAM), plus peak
            RSS of all `ollama` processes sampled from /proc during the run (Linux)
  GPU       peak/baseline memory.used via nvidia-smi, if present
  accuracy  both-axes / per-axis accuracy, gate accuracy, false-block and miss
            rate, per-category accuracy, option-order consistency (--orders)

    ollama pull tev1 && ollama pull nimble
    python eval/conscience/eval_ollama_judge.py --probe
    python eval/conscience/eval_ollama_judge.py --repeat 5 --orders 3 \\
        --report /tmp/ollama_eval.json
    python eval/conscience/eval_ollama_judge.py --models tev1 nimble tev1:0.8b --verbose

Notes
  * Ollama's /v1/systemone returns no server-side timings, only token usage, so
    latency is client wall time on localhost (HTTP overhead is negligible).
  * The whole prompt must fit the loaded context window and is never truncated;
    an over-long case comes back as HTTP 400. Raise it with the server env var
    OLLAMA_CONTEXT_LENGTH (e.g. 8192) and restart `ollama serve`.
  * RSS includes file-backed mmap pages of the weights that were actually
    touched, so it is the honest "how much RAM is this holding" number on CPU.
    With GPU offload most of the model shows up under VRAM instead.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import eval_tev1_judge as T          # cases, bands, canon, state rendering, summary maths
import eval_laya_judge as L          # LTrial, parse_response, summarize, post

AXES = T.AXES


# ── small HTTP helpers ──────────────────────────────────────────────────────

def http_get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def ps_models(host: str) -> list[dict]:
    try:
        return http_get(host + "/api/ps").get("models") or []
    except Exception:  # noqa: BLE001
        return []


def ps_entry(host: str, model: str) -> dict | None:
    """Find a loaded model in /api/ps; 'tev1' matches 'tev1:latest'."""
    base = model.split(":")[0]
    for m in ps_models(host):
        name = m.get("name") or m.get("model") or ""
        if name == model or (":" not in model and name.split(":")[0] == base and
                             name.endswith(":latest")) or name == model + ":latest":
            return m
    return None


def check_version(host: str) -> None:
    try:
        v = http_get(host + "/api/version")["version"]
    except Exception as e:  # noqa: BLE001
        print(f"warning: cannot reach {host} ({e}). Is `ollama serve` running?", file=sys.stderr)
        return
    m = re.match(r"(\d+)\.(\d+)", v)
    if m and (int(m.group(1)), int(m.group(2))) < (0, 35):
        print(f"warning: Ollama {v} is older than 0.35; /v1/systemone will 404.", file=sys.stderr)
    else:
        print(f"Ollama {v} at {host}")


# ── memory sampling (Linux /proc + optional nvidia-smi) ─────────────────────

def ollama_rss_mb() -> float | None:
    """Sum VmRSS over every process whose comm is 'ollama' (server + runners)."""
    total_kb, n = 0, 0
    proc = Path("/proc")
    if not proc.exists():
        return None
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            if (p / "comm").read_text().strip() != "ollama":
                continue
            for line in (p / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kb += int(line.split()[1])
                    n += 1
                    break
        except OSError:
            continue
    return round(total_kb / 1024.0, 1) if n else None


_HAVE_SMI = shutil.which("nvidia-smi") is not None


def gpu_used_mb() -> float | None:
    if not _HAVE_SMI:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
        vals = [float(x) for x in out.split() if x.strip()]
        return round(sum(vals), 1) if vals else None
    except Exception:  # noqa: BLE001
        return None


class MemSampler(threading.Thread):
    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_evt = threading.Event()
        self.rss_peak: float | None = None
        self.gpu_peak: float | None = None

    def _sample(self):
        r, g = ollama_rss_mb(), gpu_used_mb()
        if r is not None:
            self.rss_peak = r if self.rss_peak is None else max(self.rss_peak, r)
        if g is not None:
            self.gpu_peak = g if self.gpu_peak is None else max(self.gpu_peak, g)

    def run(self):
        while not self.stop_evt.is_set():
            self._sample()
            self.stop_evt.wait(self.interval)
        self._sample()

    def finish(self):
        self.stop_evt.set()
        self.join(timeout=5)


# ── payload / transport ─────────────────────────────────────────────────────

def build_payload(model: str, case: dict, canon, opts: dict, axes, keep_alive) -> dict:
    """Ollama /v1/systemone: criteria is {option_key: description}; dict order is
    the option order the model sees (shuffled orders test position bias)."""
    questions = {
        a: {"type": "choice",
            "instructions": AXES[a]["question"],
            "criteria": {o["key"]: o["description"] for o in opts[a]}}
        for a in axes
    }
    return {"model": model, "state": T.build_state(case, canon),
            "questions": questions, "keep_alive": keep_alive}


def ask(host: str, model: str, case: dict, canon, opts: dict, args, keep_alive):
    """One verdict = one request (default) or one per question (--split).
    Returns (parsed answers per axis, summed wall ms, (input_tokens, output_tokens))."""
    groups = [[a] for a in AXES] if args.split else [list(AXES)]
    answers, wall, itok, otok = {}, 0.0, 0, 0
    for g in groups:
        data, w = L.post(host + "/v1/systemone",
                         build_payload(model, case, canon, opts, g, keep_alive), args.timeout)
        answers.update(data.get("answers") or {})
        wall += w
        u = data.get("usage") or {}
        itok += int(u.get("input_tokens") or 0)
        otok += int(u.get("output_tokens") or 0)
    return L.parse_response({"answers": answers}), wall, (itok, otok)


def unload_all(host: str, timeout: float) -> None:
    """Unload everything so each model is measured from a cold, empty server."""
    for m in ps_models(host):
        name = m.get("name") or m.get("model")
        done = False
        for url, body in (
            (host + "/api/generate", {"model": name, "keep_alive": 0}),
            (host + "/v1/systemone",
             {"model": name, "state": "x", "keep_alive": 0,
              "questions": {"q": {"type": "choice", "instructions": "x",
                                  "criteria": {"a": "A", "b": "B"}}}}),
        ):
            try:
                L.post(url, body, timeout)
                done = True
                break
            except Exception:  # noqa: BLE001
                continue
        if not done:
            print(f"  warning: could not unload {name}; run `ollama stop {name}`", file=sys.stderr)
    for _ in range(40):
        if not ps_models(host):
            break
        time.sleep(0.5)
    time.sleep(1.5)  # let RSS / VRAM settle


# ── per-model run ───────────────────────────────────────────────────────────

def run_model(host: str, model: str, cases, canon, args) -> tuple[list, dict]:
    keep = args.keep_alive
    plain = {a: T.make_options(a, None) for a in AXES}

    unload_all(host, args.timeout)
    base_rss, base_gpu = ollama_rss_mb(), gpu_used_mb()

    # cold start: first verdict includes loading the weights
    t0 = time.perf_counter()
    try:
        ask(host, model, cases[0], canon, plain, args, keep)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{model}: cold request failed: {e}") from None
    cold_ms = (time.perf_counter() - t0) * 1000.0
    time.sleep(1.0)
    ps = ps_entry(host, model) or {}

    for _ in range(args.warmup):
        try:
            ask(host, model, cases[0], canon, plain, args, keep)
        except Exception:  # noqa: BLE001
            pass

    sampler = MemSampler()
    sampler.start()
    trials: list[L.LTrial] = []
    try:
        for case in cases:
            exp = {a: sorted(T.expected_bands(case, a)) for a in AXES}
            for oi in range(args.orders):
                opts = {}
                for axis in AXES:
                    rng = (random.Random(f"{args.seed}:{case['id']}:{axis}:{oi}")
                           if (args.shuffle or oi > 0) else None)
                    opts[axis] = T.make_options(axis, rng)
                t = L.LTrial(case["id"], case["category"], oi, exp, T.should_flag(case))
                runs = []
                for rep in range(args.repeat):
                    try:
                        parsed, wall, (itok, otok) = ask(host, model, case, canon, opts, args, keep)
                    except Exception as e:  # noqa: BLE001
                        t.error = f"{type(e).__name__}: {e}"
                        break
                    runs.append(parsed)
                    t.walls.append(wall)
                    if rep == 0:
                        t.ptok, t.ctok = itok or None, otok or None
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
                    print(f"  [{tag}] {t.case_id:22s} o{oi} "
                          f"V exp={'/'.join(exp['vertical']):16s} got={t.pred.get('vertical')!s:9s} "
                          f"H exp={'/'.join(exp['horizontal']):16s} got={t.pred.get('horizontal')!s:9s} "
                          f"{med} p(exp)={t.p_expected.get('vertical')}/{t.p_expected.get('horizontal')} "
                          f"{t.error}")
    finally:
        sampler.finish()

    ps = ps_entry(host, model) or ps
    mb = lambda b: round(b / 1048576.0, 1) if isinstance(b, (int, float)) else None  # noqa: E731
    size, vram = ps.get("size"), ps.get("size_vram")
    mem = {
        "cold_start_ms": round(cold_ms, 1),
        "ollama_total_mb": mb(size),
        "ollama_vram_mb": mb(vram),
        "ollama_cpu_ram_mb": mb(size - vram) if isinstance(size, (int, float)) and isinstance(vram, (int, float)) else None,
        "ollama_context_length": ps.get("context_length"),
        "rss_baseline_mb": base_rss,
        "rss_peak_mb": sampler.rss_peak,
        "rss_delta_mb": (round(sampler.rss_peak - base_rss, 1)
                         if sampler.rss_peak is not None and base_rss is not None else None),
        "gpu_baseline_mb": base_gpu,
        "gpu_peak_mb": sampler.gpu_peak,
        "gpu_delta_mb": (round(sampler.gpu_peak - base_gpu, 1)
                         if sampler.gpu_peak is not None and base_gpu is not None else None),
    }
    return trials, mem


def summarize(model: str, trials, mem: dict, args) -> dict:
    s = L.summarize(model, trials, args.gate_ms)
    ok = [t for t in trials if not t.error]
    walls = [w for t in ok for w in t.walls]
    ptok = [t.ptok for t in ok if t.ptok]
    if walls and ptok:  # rough prefill throughput, input tokens per second of wall time
        s["input_tok_per_s"] = round(statistics.mean(ptok) / (statistics.mean(walls) / 1000.0), 1)
    s["calls_per_verdict"] = 2 if args.split else 1
    s.update(mem)
    return s


# ── probe ───────────────────────────────────────────────────────────────────

def probe(host: str, models, cases, canon, args) -> int:
    plain = {a: T.make_options(a, None) for a in AXES}
    for m in models:
        print(f"POST {host}/v1/systemone  model={m}")
        try:
            data, wall = L.post(host + "/v1/systemone",
                                build_payload(m, cases[0], canon, plain, list(AXES), args.keep_alive),
                                args.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}\n  -> `ollama pull {m}`? Ollama >= 0.35? "
                  f"prompt too long for the context (OLLAMA_CONTEXT_LENGTH)?")
            return 1
        print(f"  {wall:.0f} ms (includes model load if cold)\n{json.dumps(data, indent=2)[:1500]}")
        print(f"  parsed: {json.dumps(L.parse_response(data))}")
        e = ps_entry(host, m)
        print(f"  /api/ps: {json.dumps(e)[:400] if e else 'model not listed'}")
    return 0


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--models", nargs="+", default=["tev1", "nimble"],
                    help="Ollama model names (default: tev1 nimble)")
    ap.add_argument("--cases", default=str(T.CASES_DEFAULT))
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per request; generous because the cold request loads the model")
    ap.add_argument("--keep-alive", default="30m", help="keep_alive sent with each request")
    ap.add_argument("--repeat", type=int, default=5, help="timing repeats per trial")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--orders", type=int, default=1, help="option orderings per case (>1 tests position bias)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", action="store_true", help="one request per question instead of both in one")
    ap.add_argument("--gate-ms", type=float, default=2500.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--no-canon", action="store_true", help="norm ids only, skip scripture rendering")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--report", default="")
    ap.add_argument("--probe", action="store_true", help="one request per model, print raw reply, exit")
    args = ap.parse_args()
    args.host = args.host.rstrip("/")
    ka = args.keep_alive
    args.keep_alive = float(ka) if re.fullmatch(r"-?\d+(\.\d+)?", ka) else ka

    cp = Path(args.cases)
    cases = T.load_cases(cp) if cp.exists() else []
    if not cases:
        print(f"no cases at {cp}", file=sys.stderr)
        return 2
    cases = [c for c in cases if c["id"] not in set(args.exclude)]
    if args.limit:
        cases = cases[:args.limit]
    canon = None if args.no_canon else T.load_canon()
    check_version(args.host)
    if ollama_rss_mb() is None:
        print("note: no `ollama` process visible in /proc (not Linux, or running in a "
              "container/other namespace): RSS numbers will be empty; /api/ps still works.",
              file=sys.stderr)
    if not _HAVE_SMI:
        print("note: nvidia-smi not found, GPU numbers skipped (ollama_vram_mb from /api/ps "
              "still shows VRAM use).", file=sys.stderr)
    if args.probe:
        return probe(args.host, args.models, cases, canon, args)

    summaries, details = [], {}
    for model in args.models:
        print(f"\nrunning {model}: {len(cases)} cases x {args.orders} order(s) x "
              f"{args.repeat} repeat(s) ...", flush=True)
        try:
            trials, mem = run_model(args.host, model, cases, canon, args)
        except RuntimeError as e:
            print(f"  skipped: {e}", file=sys.stderr)
            continue
        s = summarize(model, trials, mem, args)
        summaries.append(s)
        details[model] = [
            {"case": t.case_id, "category": t.category, "order": t.order_idx,
             "expected": t.expected, "pred": t.pred, "correct": t.correct,
             "p_expected": t.p_expected, "confidence": t.confidence,
             "error": t.error, "wall_ms": [round(w, 2) for w in t.walls]}
            for t in trials
        ]
        print(f"  both={s['accuracy_both']} V={s['accuracy_vertical']} H={s['accuracy_horizontal']} "
              f"p50={s['latency_p50_ms']}ms p95={s['latency_p95_ms']}ms cold={s['cold_start_ms']}ms "
              f"ollama={s['ollama_total_mb']}MB (vram {s['ollama_vram_mb']}MB) "
              f"rss_peak={s['rss_peak_mb']}MB")
    unload_all(args.host, args.timeout)
    if not summaries:
        return 1
    T.print_report(summaries)
    if args.report:
        Path(args.report).write_text(json.dumps({"summaries": summaries, "details": details}, indent=2))
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
