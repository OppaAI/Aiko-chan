"""Build the compact full-connectome CSR for Phase 4 (MaleCNS v1.0).

Downloads (first run):
  body-annotations-male-cns-v1.0-minconf-0.5.feather  (~14MB)
  body-neurotransmitters-male-cns-v1.0.feather        (~42MB)
  connectome-weights-male-cns-v1.0-minconf-0.5.feather (~1.05GB)

Pipeline (CC-BY, Berg et al. 2025, https://male-cns.janelia.org/download/):
  Follows the paper's own quantify-neuron-connections.ipynb exactly:
  1. keep bodies whose superclass is assigned and not 'tbc' (~166.4k neurons)
  2. keep edges with both ends valid-superclass, NO weight threshold
     (~25.6M directed connections)
  3. sign(pre) from neurotransmitter predictions: GABA -> -1 (inhibitory),
     acetylcholine / glutamate -> +1 (excitatory), unknown -> +1
  4. per-pre normalisation: w /= sum(|w_out(pre)|) so a sparse KC seed
     neither vanishes nor explodes through the hops
  5. compact CSR, float32 weights, int32 indices  (the Phase 4 four rules)

Output: malecns_full.npz (~300MB) with
  indptr (n+1 int64), indices (m int32), data (m float32),
  pre_sorted (m int32, == repeat(arange(n), diff(indptr)), cached for the
    vectorised bincount matvec),
  node_ids (n int64 bodyId), node_types (n <U32), node_sign (n float32),
  meta: provenance JSON string.

The 1.05GB weights file is deleted after slicing (same as extract_mb.py).
Run: python tools/extract_full.py [--workdir DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "annot": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "nt": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
WEIGHT_THRESHOLD = None  # None = keep the paper's full 25.6M (no threshold);
                       # the >=5 figure in the notebook is a separate derived count.

# Neurotransmitter -> sign. Fly fast transmitters: GABA inhibitory;
# acetylcholine + glutamate excitatory (Dempsey et al. conventions).
def _nt_sign(nt: str) -> float:
    s = (nt or "").strip().lower()
    if not s or s in {"unknown", "none", "unclear"}:
        return 1.0
    if "gaba" in s:
        return -1.0
    if "acetylcholine" in s or "glutamate" in s:
        return 1.0
    # monoamines / peptides: treat as modulatory-positive, weak
    return 0.5


def _dl(url: str, dest: str) -> None:
    import urllib.request
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        # verify completeness for the big weights file via HEAD size
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as r:
                want = int(r.headers.get("Content-Length") or 0)
            if want and os.path.getsize(dest) == want:
                print(f"  have {dest}")
                return
            print(f"  incomplete {dest} "
                  f"({os.path.getsize(dest)}/{want}), re-downloading")
        except Exception:
            print(f"  have {dest}")
            return
    print(f"  downloading {url} ...")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  saved {dest} ({os.path.getsize(dest) / 1e9:.2f} GB)")
            return
        except Exception as exc:
            print(f"  attempt {attempt + 1} failed: {exc}")
            try:
                os.remove(dest)
            except OSError:
                pass
    raise RuntimeError(f"download failed after 3 attempts: {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.expanduser("~/workspace/malecns"))
    ap.add_argument("--out", default=None,
                    help="output npz path (default: <workdir>/malecns_full.npz)")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    out = args.out or os.path.join(args.workdir, "malecns_full.npz")
    t0 = time.time()

    for key, name in FILES.items():
        _dl(f"{BASE}/{name}", os.path.join(args.workdir, name))

    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    print("loading annotations ...")
    a = pd.read_feather(os.path.join(args.workdir, FILES["annot"]))
    print("  neurons:", len(a), "columns:", list(a.columns))
    # Paper's validity filter (quantify-neuron-connections.ipynb): keep bodies
    # whose superclass is assigned and not 'tbc' (to-be-confirmed). This is
    # what yields the published ~166.4k neurons / ~25.6M connections.
    valid = a["superclass"].fillna("").to_numpy()
    valid_mask = np.array([bool(s) and "tbc" not in s for s in valid])
    a = a[valid_mask].reset_index(drop=True)
    print(f"  valid-superclass neurons: {len(a)}")
    body_ids = a["bodyId"].to_numpy()
    id2pos = {int(b): i for i, b in enumerate(body_ids)}
    n = len(body_ids)

    print("loading neurotransmitter predictions ...")
    nt_path = os.path.join(args.workdir, FILES["nt"])
    nsign = np.ones(n, dtype=np.float32)
    if os.path.exists(nt_path):
        nt = pd.read_feather(nt_path)
        print("  nt columns:", list(nt.columns))
        # find the prediction column (prediction with max probability)
        pred_col = None
        for cand in ("predicted_nt", "consensus_nt", "celltype_predicted_nt"):
            if cand in nt.columns:
                pred_col = cand
                break
        if pred_col is None:
            for c in nt.columns:
                if "transmitter" in c.lower() and "prob" not in c.lower() and "top" in c.lower():
                    pred_col = c
                    break
        if pred_col is None:
            # fall back: any object column that is not bodyId
            for c in nt.columns:
                if c != "bodyId" and nt[c].dtype == object:
                    pred_col = c
                    break
        print("  nt prediction column:", pred_col)
        if pred_col is not None:
            idcol = "body" if "body" in nt.columns else (
                "bodyId" if "bodyId" in nt.columns else nt.columns[0])
            seen = set()
            for b, p in zip(nt[idcol].to_numpy(), nt[pred_col].to_numpy()):
                bi = int(b)
                if bi in seen:
                    continue  # first prediction per body wins
                seen.add(bi)
                i = id2pos.get(bi)
                if i is not None:
                    nsign[i] = _nt_sign(str(p))
            inh = int((nsign < 0).sum())
            print(f"  inhibitory neurons: {inh} ({inh / n:.1%})")
    else:
        print("  WARNING: nt file missing, all signs +1")

    print("selecting edges (paper rule: both ends valid-superclass%s) ..." % (
        f", weight >= {WEIGHT_THRESHOLD}" if WEIGHT_THRESHOLD else ", no weight threshold"))
    wpath = os.path.join(args.workdir, FILES["weights"])
    # Stream in 4M-row batches (the full 152M-row table does not fit in RAM
    # next to the mapped arrays). Hash-join each batch against the valid
    # bodyId->pos map, keeping only edges with both ends valid.
    idmap = pd.DataFrame({"bodyId": body_ids,
                          "pos": np.arange(n, dtype=np.int32)})
    pre_chunks, post_chunks, w_chunks = [], [], []
    total = 0
    for batch in ds.dataset(wpath, format="ipc").to_batches(
            columns=["body_pre", "body_post", "weight"], batch_size=4_000_000):
        df = batch.to_pandas()
        if WEIGHT_THRESHOLD:
            df = df[df["weight"] >= WEIGHT_THRESHOLD]
        if len(df) == 0:
            continue
        df = df.merge(idmap, left_on="body_pre", right_on="bodyId", how="inner")
        if len(df) == 0:
            continue
        df = df.merge(idmap, left_on="body_post", right_on="bodyId",
                      how="inner", suffixes=("", "_post"))
        if len(df) == 0:
            continue
        pre_chunks.append(df["pos"].to_numpy(dtype=np.int32))
        post_chunks.append(df["pos_post"].to_numpy(dtype=np.int32))
        w_chunks.append(df["weight"].to_numpy(dtype=np.float64))
        total += len(df)
        del df
    print(f"  selected edges: {total}")
    pre = np.concatenate(pre_chunks).astype(np.int64)
    post = np.concatenate(post_chunks).astype(np.int64)
    wgt = np.concatenate(w_chunks)
    del pre_chunks, post_chunks, w_chunks

    # signed weights: sign(pre) * count
    w = (wgt * nsign[pre]).astype(np.float32)
    del wgt

    # per-PRE normalisation: w /= sum(|w_out(pre)|). Each neuron's total
    # output magnitude is preserved through the hop (random-walk style), so a
    # sparse KC seed neither vanishes (per-post norm) nor explodes (raw counts).
    print("  normalising per-pre ...")
    pre_sum = np.bincount(pre, weights=np.abs(w).astype(np.float64), minlength=n)
    w = (w / np.maximum(pre_sum[pre], 1e-9)).astype(np.float32)

    # CSR, pre-major, float32
    print("  building CSR ...")
    order = np.argsort(pre, kind="stable")
    pre_s = pre[order].astype(np.int32)
    post_s = post[order].astype(np.int32)
    w_s = w[order]
    del pre, post, w, order
    counts = np.bincount(pre_s, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    del counts

    types = a["type"].fillna("").to_numpy().astype("<U32")

    meta = {
        "dataset": "male-cns:v1.0 (minconf-0.5)",
        "paper": "Berg et al. 2025", "license": "CC-BY",
        "edge_rule": ("paper quantify-neuron-connections.ipynb: both ends "
                      "valid-superclass (superclass assigned, no 'tbc'); "
                      "no weight threshold"),
        "sign_rule": "GABA->-1, ACh/Glu->+1, unknown->+1, other->+0.5; per-pre normalised",
        "n_neurons": int(n), "n_edges": int(len(pre_s)),
        "built_by": "cognition/flymemory/tools/extract_full.py (Phase 4)",
    }
    print("  saving", out)
    np.savez_compressed(
        out,
        indptr=indptr, indices=post_s, data=w_s, pre_sorted=pre_s,
        node_ids=body_ids.astype(np.int64),
        node_types=types, node_sign=nsign,
        meta_json=np.array(json.dumps(meta)),
    )
    size_mb = os.path.getsize(out) / 1e6
    print(f"  done: {n} neurons, {len(pre_s)} edges, {size_mb:.0f} MB "
          f"in {time.time() - t0:.0f}s")

    # delete the 1.05GB weights (same convention as extract_mb.py)
    try:
        os.remove(wpath)
        print("  deleted weights.feather (kept annotations + npz)")
    except OSError as e:
        print("  could not delete weights:", e)


if __name__ == "__main__":
    sys.exit(main())
