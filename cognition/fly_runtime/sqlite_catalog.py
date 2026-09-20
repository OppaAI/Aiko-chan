"""Indexed on-disk connectome catalog: hot cells in RAM, cold graph on disk.

The full JSON catalog parses into ~6.5M Python objects (~1GB+ heap) that an
8GB Jetson cannot swallow without a multi-minute swap stall. This module
keeps the same query interface as :class:`ConnectomeCatalog` but stores the
graph in SQLite:

* node id/type/region rows (~211k) load into memory once (~tens of MB),
* 6.3M edges stay on disk behind ``CREATE INDEX edges(source)``,
* each evaluation materialises only its bounded subgraph (<= budget cells).

Build the file once, on a big machine, with
``cognition/fly_runtime/tools/build_catalog_sqlite.py`` — never on the Jetson.
"""
from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from .catalog import Edge, Node

_SCHEMA_VERSION = 1


class _NodeView(Mapping):
    """Read-only dict-like view over the in-memory node table."""

    def __init__(self, nodes: dict[str, Node]):
        self._nodes = nodes

    def __getitem__(self, key: str) -> Node:
        return self._nodes[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)


class SqliteCatalog:
    """ConnectomeCatalog-compatible catalog backed by an indexed SQLite file."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        # check_same_thread=False + own lock: evaluations may come from
        # background threads for different users.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            meta = {row["key"]: row["value"] for row in self._conn.execute("SELECT key, value FROM meta")}
        schema = meta.get("schema")
        if schema != str(_SCHEMA_VERSION):
            raise ValueError(f"{self._db_path} has schema={schema!r}, need {_SCHEMA_VERSION}")
        self.source = meta.get("source", "unknown")
        self.version = meta.get("version", "unknown")
        self.checksum = meta.get("checksum", "unknown")
        with self._lock:
            nodes = {row["id"]: Node(str(row["id"]), str(row["type"]), str(row["region"]))
                     for row in self._conn.execute("SELECT id, type, region FROM nodes")}
        self._node_table = nodes
        self.nodes: Mapping[str, Node] = _NodeView(nodes)
        self._by_type: dict[str, list[str]] = defaultdict(list)
        for node_id, node in nodes.items():
            self._by_type[node.type].append(node_id)
        for members in self._by_type.values():
            members.sort()

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass

    # ── seed resolution (same contract as ConnectomeCatalog) ────────────

    def ids_for(self, *, types: Iterable[str] = (), regions: Iterable[str] = ()) -> list[str]:
        ids = set()
        for value in types:
            ids.update(self._by_type.get(value, ()))
        if regions:
            wanted = set(regions)
            ids.update(nid for nid, node in self._node_table.items() if node.region in wanted)
        return sorted(ids)

    def ids_matching_prefixes(self, prefixes: Iterable[str]) -> list[str]:
        wanted = [str(p).lower() for p in prefixes if str(p)]
        if not wanted:
            return []
        ids: set[str] = set()
        for cell_type, members in self._by_type.items():
            lowered = cell_type.lower()
            if any(lowered.startswith(prefix) for prefix in wanted):
                ids.update(members)
        return sorted(ids)

    # ── bounded path query (edges fetched per BFS level, indexed) ──────

    def _level_edges(self, node_ids: list[str]) -> dict[str, list[Edge]]:
        """One indexed query per chunk (not per node): ~hops queries per eval."""
        out: dict[str, list[Edge]] = defaultdict(list)
        with self._lock:
            for start in range(0, len(node_ids), 500):
                chunk = node_ids[start:start + 500]
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT source, target, weight, sign FROM edges WHERE source IN ({marks})", chunk)
                for r in rows:
                    if r["target"] in self._node_table:
                        out[str(r["source"])].append(
                            Edge(str(r["source"]), str(r["target"]), float(r["weight"]), str(r["sign"])))
        return out

    def subgraph(self, seeds: Iterable[str], *, budget: int = 20000, max_hops: int = 4) -> dict:
        """Same return shape as ConnectomeCatalog.subgraph."""
        valid_seeds = sorted(set(seeds).intersection(self._node_table))
        if budget < 1:
            return {
                "nodes": [], "edges": [], "truncated": bool(valid_seeds),
                "source": self.source, "version": self.version, "checksum": self.checksum,
            }
        selected: set[str] = set()
        selected_order: list[str] = []
        scheduled: set[str] = set()
        adjacency: dict[str, list[Edge]] = {}
        frontier: list[str] = []
        truncated = False
        for seed in valid_seeds:
            if len(scheduled) >= budget:
                truncated = True
                continue
            scheduled.add(seed)
            frontier.append(seed)
        depth = 0
        while frontier and depth <= max_hops:
            level = self._level_edges(frontier)
            adjacency.update(level)
            next_frontier: list[str] = []
            for node_id in frontier:
                selected.add(node_id)
                selected_order.append(node_id)
                if depth < max_hops:
                    for edge in level.get(node_id, ()):
                        if edge.target in scheduled:
                            continue
                        if len(scheduled) >= budget:
                            truncated = True
                            continue
                        scheduled.add(edge.target)
                        next_frontier.append(edge.target)
            frontier = next_frontier
            depth += 1
        edges = [edge for node_id in selected_order
                 for edge in adjacency.get(node_id, ()) if edge.target in selected]
        return {
            "nodes": [self._node_table[node_id] for node_id in sorted(selected)],
            "edges": edges,
            "truncated": truncated,
            "source": self.source, "version": self.version, "checksum": self.checksum,
        }

    def summary(self) -> dict:
        with self._lock:
            edge_count = self._conn.execute("SELECT COUNT(*) AS n FROM edges").fetchone()["n"]
        return {"nodes": len(self._node_table), "edges": int(edge_count),
                "source": self.source, "version": self.version, "checksum": self.checksum}
