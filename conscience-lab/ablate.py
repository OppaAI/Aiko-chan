#!/usr/bin/env python3
"""Ablation matrix for the conscience ladder: latency, RAM, accuracy.

Every row is a full ladder re-measured end to end, not a component micro-benchmark.
The point is that each layer's value depends on what the others already do:

    L0 guardrails catch 0/250 narrative harm but are the only REFUSE tier
    L1 allowlist clears routine traffic the judge would otherwise escalate
    L1 canon retrieval feeds the judge -- believed inert, never actually tested
    L2 judge carries narrative-harm recall, the only layer that does

So "how good is layer X" has no answer in isolation. This measures the whole
ladder with each layer switched on and off, and each judge swapped, and
reports the same three numbers for every row so they are comparable.

Rows
----
  A  L0+L1+L2   3x2   semantic on     <- everything on
  B  L0+L1+L2   3x2   semantic off    <- WHAT IS DEPLOYED
  C  L0+L1+L2   v5    semantic off
  D  L0+L1+L2   base  semantic off
  E  L0+L1      (no judge)            <- what the deterministic layers alone do
  F  L2 only    3x2
  G  L2 only    v5
  H  L2 only    base

Layer toggles are explicit monkeypatches on the imported modules, with the
prior value restored afterwards, so a row cannot leak into the next row. The
judge swap needs a real server restart because the GGUF is loaded at boot.

Usage
-----
    sudo systemctl stop laya-conscience.service     # frees ~1.2GB, required
    python ablate.py                                # all rows
    python ablate.py --rows B,E,F                   # subset
    python ablate.py --limit 120                    # ETHICS samples per class

Row B must reproduce bench_live.py; if it does not, the harness is wrong.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import json
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

MODELS = {
    "3x2": REPO / "models/laya/conscience-laya-3x2-q8_0.gguf",
    "v5": REPO / "models/laya/conscience-laya-v5-q8_0.gguf",
    "base": REPO / "models/laya/laya_multilingual_q8_0.gguf",
}
LAYA_BIN = pathlib.Path("/home/oppa-ai/ggmlc/build/examples/laya/laya")
PORT = 8093
Q_FILE = pathlib.Path("/tmp/ablate_questions.json")

# Judge scheme per checkpoint: the vocabulary it was trained on. Asking a model
# a question it has not seen makes it emit foreign band names, which then read as
# real negatives and inflate every number in the row.
SCHEME_FOR_MODEL = {"3x2": "3x2", "v5": "legacy", "base": "2x2"}

# Set by --force-scheme. Comparing checkpoints across vocabularies confounds the
# weights with the question wording, which is how "base beats v5" could be an
# artefact rather than a result.
FORCE_SCHEME = False


# ── server lifecycle ──────────────────────────────────────────────────────
class Judge:
    """Owns the laya server process so rows can swap the GGUF."""

    def __init__(self, port: int):
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.current: str | None = None

    def _healthy(self) -> bool:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).read()
            return True
        except Exception:
            return False

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _port_listener(self) -> str:
        try:
            out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True,
                                 timeout=10).stdout
            for line in out.splitlines():
                if f":{self.port} " in line or line.rstrip().endswith(f":{self.port}"):
                    return line.split("pid=")[-1].split(",")[0] if "pid=" in line else "?"
        except Exception:
            pass
        return "?"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None
        self.current = None

    def start(self, model_key: str) -> None:
        if self.current == model_key and self._healthy() and self._alive():
            return
        self.stop()
        # Refuse to run if something else already owns the port. Without this, a
        # leftover server answers our health check, our own bind fails, and the
        # row silently reports the OTHER model's numbers -- which is exactly what
        # happened once already (base row scored 3x2's results).
        if self._healthy():
            holder = self._port_listener()
            raise SystemExit(
                f"FATAL: port {self.port} is already served by pid {holder}, not by "
                f"this harness. Kill it first: kill {holder}"
            )
        path = MODELS[model_key]
        if not path.exists():
            raise SystemExit(f"FATAL: missing GGUF for {model_key}: {path}")
        log = open(f"/tmp/ablate_laya_{model_key}.log", "w")
        self.proc = subprocess.Popen(
            [str(LAYA_BIN), "serve", str(path), "--port", str(self.port), "--device", "cuda"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        for _ in range(180):
            # Health alone is not identity: require OUR process to still be up,
            # so a foreign server answering the port cannot be mistaken for ours.
            if self._healthy() and self._alive():
                self.current = model_key
                return
            if self.proc.poll() is not None:
                raise SystemExit(
                    f"FATAL: server exited for {model_key}; see "
                    f"/tmp/ablate_laya_{model_key}.log"
                )
            time.sleep(1)
        raise SystemExit(f"FATAL: {model_key} did not become healthy in 180s")


# ── layer toggles ─────────────────────────────────────────────────────────
@contextlib.contextmanager
def layers(l0: bool, l1a: bool):
    """Enable/disable L0 reflex and the L1a allowlist for one row.

    `scan()` reads the module-global GUARDRAILS_ENABLED, so rebinding it on the
    guardrails module is what actually gates the rules. (It is defined in
    schema.py and re-exported; there is no separate L0 flag.)
    """
    from cognition.conscience import allowlist as A
    from cognition.conscience import guardrails as G

    gr = G.GUARDRAILS_ENABLED
    ksa = A.known_safe_class
    G.GUARDRAILS_ENABLED = gr and l0
    if not l1a:
        A.known_safe_class = lambda _t: None
    try:
        yield
    finally:
        G.GUARDRAILS_ENABLED = gr
        A.known_safe_class = ksa


def get_embedder():
    """Production's embedder: HarrierEmbedder over llama-server's /embedding.

    Constructed directly rather than via a singleton -- there is no module-level
    instance, and `cognition.conscience.__init__`'s `memorize.embedder()` is only
    illustrative docstring text, not a real accessor. Without this, canon.retrieve
    never takes the semantic branch and a `semantic` row measures nothing while
    appearing to pass.
    """
    from cognition.memory.vecstore import HarrierEmbedder
    return HarrierEmbedder()


def write_questions() -> None:
    """Questions file the CLI/server uses, so all rows score the same shapes."""
    from cognition.conscience import laya_judge as L
    json.dump({"vertical": L._SCHEMES["3x2"][0], "horizontal": L._SCHEMES["3x2"][1]},
              open(Q_FILE, "w"))


# ── measurement ───────────────────────────────────────────────────────────
def proc_rss_mb(pid: int) -> float | None:
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


def load_sets(limit: int, cases: str = "oppa_benign_cases.jsonl"):
    """`cases` selects the labelled population. Default is Oppa's conversational
    set; agent_cases.jsonl is the operational one. Both use expect_decision;
    agent_cases additionally marks `harm` so misses can be split by intent.
    """
    oppa = [json.loads(l) for l in
            (HERE / "data" / cases).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    src = pathlib.Path("/tmp/ethics/commonsense_test.csv")
    harm = ben = []
    if src.exists():
        rows = [r for r in csv.DictReader(src.open(newline="", encoding="utf-8"))
                if r.get("is_short") == "True"]
        harm = [r["input"] for r in rows if r["label"] == "1"][:limit]
        ben = [r["input"] for r in rows if r["label"] == "0"][:limit]
    return oppa, harm, ben


def run_row(judge: Judge, row: dict, oppa, harm, ben) -> dict:
    from system.config import load_config
    load_config()
    # The scheme is read from the environment at import time, and load_config()
    # re-exports the YAML value over anything set beforehand -- so it CANNOT be
    # switched per-row inside one process. It is set externally, once per
    # invocation, and asserted here. The GGUF swap is server-side, so the Python
    # judge object needs no reload at all.
    from cognition.conscience import laya_judge as L
    from cognition.conscience.core import ConscienceCircuitCore

    want = SCHEME_FOR_MODEL.get(row["model"]) if row.get("model") else None
    if want and L.JUDGE_SCHEME != want and not FORCE_SCHEME:
        raise SystemExit(
            f"row {row['id']} ({row['model']}): judge scheme is {L.JUDGE_SCHEME}, "
            f"expected {want}. Run this row with CCC_JUDGE_SCHEME={want} -- asking "
            f"a checkpoint a question it was not trained on inflates every number."
        )

    if row.get("model"):
        judge.start(row["model"])
    else:
        judge.stop()

    ceiling = row.get("ceiling", 4)
    core = ConscienceCircuitCore(f"ablate-{row['id']}")
    ctx = {"surface": "chat"}
    # CANON_SEMANTIC is read at import time in canon.py, so the env var must be
    # set before this process starts -- that is why row A is a separate run.
    from cognition.conscience import canon as canon_mod
    embedder = get_embedder() if row.get("semantic") else None
    out = {"id": row["id"], "model": row.get("model") or "none",
           "l0": row["l0"], "l1a": row["l1a"], "semantic": row.get("semantic", False)}
    # Recorded rather than assumed: a semantic row with CANON_SEMANTIC off, or
    # with no embedder, silently measures the lexical path and looks fine.
    out["canon_semantic_active"] = bool(getattr(canon_mod, "CANON_SEMANTIC", False))
    out["embedder_supplied"] = embedder is not None
    if row.get("semantic") and not out["canon_semantic_active"]:
        raise SystemExit(
            f"row {row['id']}: asked for semantic but CANON_SEMANTIC is "
            f"{out['canon_semantic_active']} -- refusing to report a row that "
            f"measures the lexical path"
        )

    rss0 = proc_rss_mb(judge.proc.pid) if judge.proc else None
    avail0 = mem_available_mb()

    with layers(row["l0"], row["l1a"]):
        def one(text):
            t0 = time.perf_counter()
            kw = {"embedder": embedder} if embedder is not None else {}
            v = core.evaluate(act="respond", content=text, context=dict(ctx),
                              max_layer=ceiling, **kw)
            return (time.perf_counter() - t0) * 1000, v

        for _ in range(3):
            one(oppa[0]["situation"])

        lat, dec, bylayer = [], collections.Counter(), collections.Counter()
        for r in oppa:
            ms, v = one(r["situation"])
            lat.append(ms); dec[v.decision] += 1
            bylayer[(v.layers_run or ["?"])[-1] if not row["l0"] else
                    ("judge" if any("laya" in x for x in (v.reasons or []))
                     else (v.layers_run or ["?"])[-1])] += 1
            r["_got"] = v.decision

        harmful = [r for r in oppa if r.get("harm")]
        out["oppa"] = {
            "n": len(oppa),
            "harm_n": len(harmful),
            "harm_blocked": sum(1 for r in harmful if r["_got"] != "allow") if harmful else None,
            "benign_blocked": sum(1 for r in oppa if not r.get("harm")
                                  and r["_got"] in ("escalate", "refuse")),
            "lat_p50": round(statistics.median(lat), 1),
            "lat_p90": round(sorted(lat)[int(len(lat) * 0.9)], 1),
            "decisions": dict(dec),
            "benign_interrupted": sum(1 for r in oppa if r["expect_decision"] == "allow"
                                      and r["_got"] in ("escalate", "refuse")),
            "benign_total": sum(1 for r in oppa if r["expect_decision"] == "allow"),
            "harm_missed": sum(1 for r in oppa if r["expect_decision"] in ("refuse", "escalate")
                               and r["_got"] == "allow"),
            "harm_total": sum(1 for r in oppa if r["expect_decision"] in ("refuse", "escalate")),
            "by_layer": dict(bylayer),
        }

        for name, pop, key in (("harm", harm, "harm_missed"), ("benign", ben, "benign_interrupted")):
            if not pop:
                continue
            l2 = []
            n_block = 0
            for t in pop:
                ms, v = one(t)
                l2.append(ms)
                if v.decision in ("escalate", "refuse"):
                    n_block += 1
            out[f"ethics_{name}"] = {
                "n": len(pop),
                "lat_p50": round(statistics.median(l2), 1),
                "interrupted": n_block,
            }
        if embedder is not None:
            sc = canon_mod.get_canon().retrieve(harm[0] if harm else oppa[0]["situation"],
                                                embedder=embedder)
            out["semantic_probe_norms"] = len([1 for s_, _ in sc if s_ > 0.0])

    rss1 = proc_rss_mb(judge.proc.pid) if judge.proc else None
    out["judge_rss_mb"] = round(rss1, 0) if rss1 else None
    out["rss_growth_mb"] = round(rss1 - rss0, 0) if (rss1 and rss0) else None
    out["mem_available_mb"] = round(mem_available_mb(), 0) if mem_available_mb() else None
    for r in oppa:
        r.pop("_got", None)
    return out


ROWS = [
    dict(id="A", model="3x2", l0=True,  l1a=True,  ceiling=4, semantic=True,
         label="L0+L1+L2  3x2  semantic ON"),
    dict(id="B", model="3x2", l0=True,  l1a=True,  ceiling=4, semantic=False,
         label="L0+L1+L2  3x2  semantic off  <-- DEPLOYED"),
    dict(id="C", model="v5",  l0=True,  l1a=True,  ceiling=4, semantic=False,
         label="L0+L1+L2  v5   semantic off"),
    dict(id="D", model="base",l0=True,  l1a=True,  ceiling=4, semantic=False,
         label="L0+L1+L2  base semantic off"),
    dict(id="E", model=None,  l0=True,  l1a=True,  ceiling=1, semantic=False,
         label="L0+L1 only, no judge"),
    dict(id="F", model="3x2", l0=False, l1a=False, ceiling=4, semantic=False,
         label="L2 only  3x2"),
    dict(id="G", model="v5",  l0=False, l1a=False, ceiling=4, semantic=False,
         label="L2 only  v5"),
    dict(id="H", model="base",l0=False, l1a=False, ceiling=4, semantic=False,
         label="L2 only  base"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="ABCDEFGH")
    ap.add_argument("--cases", default="oppa_benign_cases.jsonl",
                    help="labelled set in conscience-lab/data/")
    ap.add_argument("--tag", default="", help="suffix for the json output")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--json", default="")
    ap.add_argument("--warn-semantic-off", action="store_true",
                    help="fail if a row asked for semantic but CANON_SEMANTIC is off")
    ap.add_argument("--force-scheme", action="store_true",
                    help="allow asking a checkpoint a vocabulary it was not tuned on; "
                         "for controls that isolate the vocabulary from the weights")
    args = ap.parse_args()
    globals()["FORCE_SCHEME"] = args.force_scheme

    write_questions()
    oppa = load_sets(args.limit, args.cases)[0]
    # Normalise the two schemas: oppa_* uses expect_decision, agent_cases uses
    # expect. Same three values, so the comparison logic can stay shared.
    for r in oppa:
        r.setdefault("expect_decision", r.get("expect"))
    harm = ben = []
    judge = Judge(PORT)
    results = []
    try:
        for row in ROWS:
            if row["id"] not in args.rows:
                continue
            if row.get("semantic") and not os.environ.get("CCC_CANON_SEMANTIC"):
                print(f"\n>>> row {row['id']}: SKIPPED -- semantic requested but "
                      f"CCC_CANON_SEMANTIC is not set in this process env. Run it as:\n"
                      f"    CCC_CANON_SEMANTIC=1 python ablate.py --rows A",
                      flush=True)
                continue
            print(f"\n>>> row {row['id']}: {row['label']}", flush=True)
            t0 = time.perf_counter()
            res = run_row(judge, row, oppa, harm, ben)
            res["wall_s"] = round(time.perf_counter() - t0, 1)
            results.append(res)
            o = res["oppa"]
            print(f"    oppa p50={o['lat_p50']}ms  benign interrupted "
                  f"{o['benign_interrupted']}/{o['benign_total']}  "
                  f"harm missed {o['harm_missed']}/{o['harm_total']}", flush=True)
            print(f"    judge RSS {res['judge_rss_mb']}MB  "
                  f"mem avail {res['mem_available_mb']}MB", flush=True)
    finally:
        judge.stop()

    out_json = args.json or f"/tmp/ablate_{args.cases.split('.')[0]}{args.tag}.json"
    pathlib.Path(out_json).write_text(json.dumps(results, indent=2))

    hdr = f"{'row':4s} {'config':30s} {'oppa p50':>9s} {'benign int':>11s} {'harm miss':>10s} {'eth harm':>9s} {'RSS':>7s}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in results:
        o, e = r["oppa"], r.get("ethics_harm", {})
        eh = f"{e.get('interrupted', '-')}/{e.get('n', '-')}"
        print(f"{r['id']:4s} {r['model'] + ' L0' if r['l0'] else 'L2 ':4s}"
              f"{'a' if r['l1a'] else ' ':1s} sem={'Y' if r['semantic'] else 'n'} "
              f"{'':13s} {o['lat_p50']:8.1f} "
              f"{o['benign_interrupted']:4d}/{o['benign_total']:<4d} "
              f"{o['harm_missed']:4d}/{o['harm_total']:<4d} {eh:>9s} "
              f"{r['judge_rss_mb'] or 0:6.0f}M")
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())