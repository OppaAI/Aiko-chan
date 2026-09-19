"""Source-attributed, read-only connectome catalog and bounded path queries."""
from __future__ import annotations

import json
import mmap
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Node:
    id: str
    type: str = "unknown"
    region: str = "unknown"


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    weight: float = 1.0
    sign: str = "unknown"


class ConnectomeCatalog:
    """An immutable catalog with deterministic, bounded breadth-first queries.

    JSON is deliberately the interchange format for this first vertical slice.
    ``from_path`` reads it through an mmap so a future generated MaleCNS extract
    can be kept off the Python heap until it is indexed.  Unknown biological
    fields remain ``unknown`` rather than being fabricated.
    """

    def __init__(self, nodes: Iterable[Node], edges: Iterable[Edge], *, source: str = "unknown", version: str = "unknown"):
        self.nodes = {node.id: node for node in nodes}
        self.edges = tuple(edge for edge in edges if edge.source in self.nodes and edge.target in self.nodes)
        self.source, self.version = source, version
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._by_type: dict[str, list[str]] = defaultdict(list)
        self._by_region: dict[str, list[str]] = defaultdict(list)
        for node in self.nodes.values():
            self._by_type[node.type].append(node.id)
            self._by_region[node.region].append(node.id)
        for edge in self.edges:
            self._out[edge.source].append(edge)
        for values in (*self._out.values(), *self._by_type.values(), *self._by_region.values()):
            values.sort(key=lambda item: item.target if isinstance(item, Edge) else item)

    @classmethod
    def from_path(cls, path: str | Path) -> ConnectomeCatalog:
        path = Path(path)
        with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            raw = json.loads(mapped.read().decode("utf-8"))
        nodes = [Node(str(n["id"]), str(n.get("type", "unknown")), str(n.get("region", "unknown"))) for n in raw.get("nodes", [])]
        edges = [Edge(str(e["source"]), str(e["target"]), float(e.get("weight", 1.0)), str(e.get("sign", "unknown"))) for e in raw.get("edges", [])]
        return cls(nodes, edges, source=str(raw.get("source", "unknown")), version=str(raw.get("version", "unknown")))

    def ids_for(self, *, types: Iterable[str] = (), regions: Iterable[str] = ()) -> list[str]:
        ids = set()
        for value in types:
            ids.update(self._by_type.get(value, ()))
        for value in regions:
            ids.update(self._by_region.get(value, ()))
        return sorted(ids)

    def subgraph(self, seeds: Iterable[str], *, budget: int = 20000, max_hops: int = 4) -> dict:
        """Return a deterministic forward path query, never exceeding *budget*."""
        if budget < 1:
            return {"nodes": [], "edges": [], "truncated": bool(tuple(seeds)), "source": self.source, "version": self.version}
        selected: set[str] = set()
        queue = deque((seed, 0) for seed in sorted(set(seeds)) if seed in self.nodes)
        while queue and len(selected) < budget:
            node_id, depth = queue.popleft()
            if node_id in selected:
                continue
            selected.add(node_id)
            if depth < max_hops:
                for edge in self._out.get(node_id, ()):
                    if edge.target not in selected:
                        queue.append((edge.target, depth + 1))
        edges = [edge for edge in self.edges if edge.source in selected and edge.target in selected]
        return {
            "nodes": [self.nodes[node_id] for node_id in sorted(selected)],
            "edges": edges,
            "truncated": bool(queue),
            "source": self.source,
            "version": self.version,
        }

    def summary(self) -> dict:
        return {"nodes": len(self.nodes), "edges": len(self.edges), "source": self.source, "version": self.version}
