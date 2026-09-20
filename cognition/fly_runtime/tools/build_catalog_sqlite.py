#!/usr/bin/env python3
"""Convert the MaleCNS catalog JSON into the indexed SQLite form read by
cognition.fly_runtime.sqlite_catalog.SqliteCatalog.

Run ONCE on a build machine with ample RAM (the 489MB JSON parses to
gigabytes transiently) — never on the Jetson. Output lands next to the
JSON so service._configured_catalog() picks it up automatically:

    python3 cognition/fly_runtime/tools/build_catalog_sqlite.py \\
        --json data/fly_catalog/male-cns-v1.0-w5.json

Verifies the embedded SHA-256 exactly like ConnectomeCatalog.from_path,
then writes nodes / edges / meta tables plus the source index.
Also importable: build_sqlite(nodes, edges, meta, dest) for tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cognition.fly_runtime.sqlite_catalog import _SCHEMA_VERSION

_BATCH = 50_000


def build_sqlite(nodes: list[dict], edges: list[dict], meta: dict, dest: str | Path) -> dict:
    """Write node/edge dicts + meta into an indexed SQLite file. Returns stats."""
    dest = Path(dest)
    if dest.exists():
        dest.unlink()
    conn = sqlite3.connect(str(dest))
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE nodes(id TEXT PRIMARY KEY, type TEXT, region TEXT)")
        conn.execute("CREATE TABLE edges(source TEXT, target TEXT, weight REAL, sign TEXT)")
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?)",
            [("schema", str(_SCHEMA_VERSION)), ("source", str(meta.get("source", "unknown"))),
             ("version", str(meta.get("version", "unknown"))),
             ("checksum", str(meta.get("checksum", "unknown"))),
             ("node_count", str(len(nodes))), ("edge_count", str(len(edges)))],
        )
        node_ids: set[str] = set()
        batch: list[tuple] = []
        for item in nodes:
            nid = str(item["id"])
            node_ids.add(nid)
            batch.append((nid, str(item.get("type", "unknown")), str(item.get("region", "unknown"))))
            if len(batch) >= _BATCH:
                conn.executemany("INSERT INTO nodes(id, type, region) VALUES(?, ?, ?)", batch)
                batch = []
        if batch:
            conn.executemany("INSERT INTO nodes(id, type, region) VALUES(?, ?, ?)", batch)
        kept = dropped = 0
        batch = []
        for item in edges:
            src, tgt = str(item["source"]), str(item["target"])
            if src not in node_ids or tgt not in node_ids:
                dropped += 1
                continue
            batch.append((src, tgt, float(item.get("weight", 1.0)), str(item.get("sign", "unknown"))))
            kept += 1
            if len(batch) >= _BATCH:
                conn.executemany("INSERT INTO edges(source, target, weight, sign) VALUES(?, ?, ?, ?)", batch)
                batch = []
        if batch:
            conn.executemany("INSERT INTO edges(source, target, weight, sign) VALUES(?, ?, ?, ?)", batch)
        conn.execute("CREATE INDEX idx_edges_source ON edges(source)")
        conn.commit()
    finally:
        conn.close()
    return {"nodes": len(nodes), "edges_kept": kept, "edges_dropped": dropped,
            "bytes": dest.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, help="Input catalog JSON path")
    ap.add_argument("--out", default="", help="Output .db path (default: same basename)")
    args = ap.parse_args()

    src = Path(args.json)
    print(f"reading {src} …", flush=True)
    raw = json.loads(src.read_bytes().decode("utf-8"))
    checksum_payload = json.dumps(
        {k: v for k, v in raw.items() if k != "checksum"},
        sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(checksum_payload).hexdigest()
    if str(raw.get("checksum", checksum)) != checksum:
        print("checksum MISMATCH — refusing to convert", flush=True)
        return 1
    dest = Path(args.out) if args.out else src.with_suffix(".db")
    print(f"converting {len(raw.get('nodes', []))} nodes / {len(raw.get('edges', []))} edges …", flush=True)
    stats = build_sqlite(
        raw.get("nodes", []), raw.get("edges", []),
        {"source": raw.get("source", "unknown"), "version": raw.get("version", "unknown"), "checksum": checksum},
        dest,
    )
    print(f"wrote {dest} ({stats['bytes'] / 1e6:.0f} MB): {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
