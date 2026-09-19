"""Build the verified full-brain MaleCNS catalog JSON for the bounded runtime.

Reads the official flat-connectome release over streaming batches (never the
whole graph on the heap), thresholds at weight>=5 (paper standard, Berg et
al. 2025), and writes canonical JSON while hashing the exact bytes, so the
embedded SHA-256 always matches what ConnectomeCatalog.from_path verifies.

Honesty rules (runtime contract):
  - region: "unknown" for every node (annotations carry no neuropil column).
  - sign: "unknown" for every edge (flat weights carry no sign; joining the
    separate neurotransmitter predictions is future work).
  - No token needed: public HTTPS release buckets (CC-BY).

Output: data/fly_catalog/male-cns-v1.0-w5.json (~400MB, gitignored).

Usage:
  python3 cognition/fly_runtime/tools/build_catalog.py \\
      --annotations /tmp/annot.feather --weights /tmp/weights.feather \\
      --out data/fly_catalog/male-cns-v1.0-w5.json [--min-weight 5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Make the repo importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc


def _canon(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-weight", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1_000_000)
    args = ap.parse_args()

    print("reading annotations…", flush=True)
    a = pd.read_feather(args.annotations)
    bodies = a["bodyId"].tolist()
    types = a["type"].fillna("").tolist()
    seen: dict[int, str] = {}
    for b, t in zip(bodies, types):
        if int(b) not in seen:
            seen[int(b)] = t if t else "unknown"
    print(f"nodes: {len(seen)}", flush=True)

    tmp = args.out + ".payload.tmp"
    h = hashlib.sha256()
    n_edges = 0

    def emit(raw: bytes) -> None:
        h.update(raw)
        fout.write(raw)

    dataset = ds.dataset(args.weights, format="ipc")
    filt = pc.field("weight") >= args.min_weight
    print("streaming edges…", flush=True)
    with open(tmp, "wb") as fout:
        emit(b'{"edges":[')
        first = True
        scanner = dataset.scanner(columns=["body_pre", "body_post", "weight"], filter=filt,
                                  batch_size=args.batch_size)
        for batch in scanner.to_batches():
            d = batch.to_pydict()
            for pre, post, w in zip(d["body_pre"], d["body_post"], d["weight"]):
                if not first:
                    emit(b",")
                first = False
                # Preserve numeric type exactly: from_path re-serializes the
                # parsed value, so 7 must stay 7 (not 7.0) and vice versa.
                wv = int(w) if float(w).is_integer() else float(w)
                emit(_canon({"source": str(int(pre)), "target": str(int(post)),
                             "weight": wv, "sign": "unknown"}))
                n_edges += 1
                if n_edges % 1_000_000 == 0:
                    print(f"  {n_edges} edges…", flush=True)
        emit(b'],"nodes":[')
        first = True
        for b in sorted(seen):
            if not first:
                emit(b",")
            first = False
            emit(_canon({"id": str(b), "region": "unknown", "type": seen[b]}))
        emit(b'],"source":')
        emit(_canon("male-cns:v1.0 minconf-0.5 weight>=5 (CC-BY, Berg et al. 2025, https://male-cns.janelia.org)"))
        emit(b',"version":')
        emit(_canon("v1.0-w5"))
        emit(b"}")
    digest = h.hexdigest()
    print(f"edges: {n_edges}, sha256: {digest}", flush=True)

    print("assembling final file…", flush=True)
    with open(tmp, "rb") as fin, open(args.out, "wb") as fout:
        first_byte = fin.read(1)
        assert first_byte == b"{", first_byte
        fout.write(b'{"checksum":"' + digest.encode() + b'",')
        while True:
            chunk = fin.read(8 << 20)
            if not chunk:
                break
            fout.write(chunk)
    import os
    os.remove(tmp)

    # Self-verify with the real loader before declaring success.
    from cognition.fly_runtime.catalog import ConnectomeCatalog
    cat = ConnectomeCatalog.from_path(args.out)
    print("verified:", cat.summary(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
