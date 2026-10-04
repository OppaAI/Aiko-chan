#!/usr/bin/env python3
"""Weight-space interpolation between two fine-tunes of the same base.

v5 wins the 24 hand-authored cases, v8 wins the 263-row set. Both are full
fine-tunes from the same initialisation (the base checkpoint), differing only in
their training data. That is precisely the setting where linear interpolation
of the weights behaves well: the two solutions sit in the same basin, so a
midpoint is usually a valid model rather than noise.

Sweeping alpha lets us ask directly whether some mixture keeps v8's accuracy on
the large set without giving up v5's benign-band coverage on the small one. An
alpha of 0 is v5, 1 is v8.

    ../.venv/bin/python soup_sweep.py --a "0 0.25 0.5 0.75 1"
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/oppa-ai/jetson/Aiko-chan/eval/conscience")

HERE = Path(__file__).resolve().parent


def soup(a_dir: Path, b_dir: Path, alpha: float, out: Path) -> Path:
    """w = (1-alpha)*a + alpha*b, written as a loadable checkpoint dir."""
    from safetensors.torch import load_file, save_file

    A = load_file(str(a_dir / "model.safetensors"))
    B = load_file(str(b_dir / "model.safetensors"))
    if set(A) != set(B):
        raise SystemExit(f"key mismatch: {sorted(set(A) ^ set(B))[:5]}")

    merged = {}
    for k in A:
        ta, tb = A[k], B[k]
        if ta.is_floating_point():
            merged[k] = ((1.0 - alpha) * ta.float() + alpha * tb.float()).to(ta.dtype)
        else:
            merged[k] = ta  # integer buffers (ids, shapes) must not interpolate

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(a_dir, out)
    save_file(merged, str(out / "model.safetensors"))
    return out


def evaluate(ckpt: Path, test: Path) -> dict:
    """Score a checkpoint on both sets in a fresh process (keeps VRAM clean)."""
    code = f'''
import json, sys
from pathlib import Path
sys.path.insert(0, {str(REPO)!r}); sys.path.insert(0, "/home/oppa-ai/jetson/Aiko-chan/eval/conscience")
from cognition.conscience.laya_judge import _HORIZONTAL_Q, _VERTICAL_Q
import laya, eval_tev1_judge as T
Q = {{"vertical": _VERTICAL_Q, "horizontal": _HORIZONTAL_Q}}
BAND = {{"unclear": "unknown", "mixed": "unknown"}}
ag = laya.load({str(ckpt)!r})
def ask(s):
    a = ag.predict(s, Q).get("answers") or {{}}
    return ((a.get("vertical") or {{}}).get("choice"), (a.get("horizontal") or {{}}).get("choice"))
rows = [json.loads(l) for l in Path({str(test)!r}).read_text(encoding="utf-8").splitlines() if l.strip()]
v=h=b=0
for r in rows:
    g = ask(r["fields"]["scenario"])
    okv = g[0]==r["answers"]["vertical"]; okh = g[1]==r["answers"]["horizontal"]
    v+=okv; h+=okh; b+=okv and okh
n=len(rows); big = {{"n":n,"V":round(v/n,3),"H":round(h/n,3),"both":round(b/n,3)}}
cases=[json.loads(l) for l in Path(T.CASES_DEFAULT).read_text(encoding="utf-8").splitlines() if l.strip()]
canon=T.load_canon(); v=h=b=0
for c in cases:
    g=ask(T.build_state(c,canon))
    okv=g[0] in {{BAND.get(x,x) for x in T.expected_bands(c,"vertical")}}
    okh=g[1] in {{BAND.get(x,x) for x in T.expected_bands(c,"horizontal")}}
    v+=okv; h+=okh; b+=okv and okh
n=len(cases); small={{"n":n,"V":round(v/n,3),"H":round(h/n,3),"both":round(b/n,3)}}
print("@@"+json.dumps({{"big":big,"small":small}}))
'''
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    print(res.stdout[-800:], res.stderr[-800:])
    raise SystemExit(f"scoring failed for {ckpt}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="runs/conscience-laya-v5")
    ap.add_argument("--b", default="runs/conscience-laya-v8")
    ap.add_argument("--alphas", default="0 0.25 0.5 0.75 1")
    ap.add_argument("--test", default="data/conscience_test_v8.jsonl")
    ap.add_argument("--outdir", default="runs/soup")
    args = ap.parse_args()

    a_dir, b_dir = HERE / args.a, HERE / args.b
    alphas = [float(x) for x in args.alphas.split()]
    outroot = HERE / args.outdir
    outroot.mkdir(parents=True, exist_ok=True)

    print(f"a = {a_dir.name}\nb = {b_dir.name}\n")
    print(f"{'alpha':>6s} | {'263-row set  V/H/both':>26s} | {'24-case set  V/H/both':>26s}")
    print("-" * 66)
    table = []
    for al in alphas:
        tag = f"alpha{al:g}".replace(".", "")
        ck = soup(a_dir, b_dir, al, outroot / tag) if al not in (0.0, 1.0) else (a_dir if al == 0 else b_dir)
        r = evaluate(ck, HERE / args.test)
        big, small = r["big"], r["small"]
        print(f"{al:6.2f} | {big['V']:8.3f} {big['H']:6.3f} {big['both']:6.3f}          "
              f"| {small['V']:8.3f} {small['H']:6.3f} {small['both']:6.3f}")
        table.append((al, big, small))

    # bar: beat v5 (alpha=0) on the big set AND match it on the small set
    base_big = next(t[1] for t in table if t[0] == 0.0)
    base_small = next(t[2] for t in table if t[0] == 0.0)
    print("\ncandidates that beat v5 on 263 rows without losing on the 24:")
    found = False
    for al, big, small in table:
        gb = big["both"] > base_big["both"] and big["V"] >= base_big["V"] and big["H"] >= base_big["H"]
        gs = small["both"] >= base_small["both"]
        if gb and gs:
            found = True
            print(f"   alpha={al:g}  big both={big['both']}  small both={small['both']}")
    if not found:
        print("   none -- report above shows the trade-off curve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())