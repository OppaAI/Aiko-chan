"""Source-attributed, read-only connectome catalog and bounded path queries."""
from __future__ import annotations

import hashlib
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

    def __init__(self, nodes: Iterable[Node], edges: Iterable[Edge], *, source: str = "unknown", version: str = "unknown", checksum: str = "unknown"):
        self.nodes = {node.id: node for node in nodes}
        self.edges = tuple(edge for edge in edges if edge.source in self.nodes and edge.target in self.nodes)
        self.source, self.version, self.checksum = source, version, checksum
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
            payload = mapped.read()
        raw = json.loads(payload.decode("utf-8"))
        checksum_payload = json.dumps({key: value for key, value in raw.items() if key != "checksum"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        checksum = hashlib.sha256(checksum_payload).hexdigest()
        nodes = [Node(str(n["id"]), str(n.get("type", "unknown")), str(n.get("region", "unknown"))) for n in raw.get("nodes", [])]
        edges = [Edge(str(e["source"]), str(e["target"]), float(e.get("weight", 1.0)), str(e.get("sign", "unknown"))) for e in raw.get("edges", [])]
        declared = str(raw.get("checksum", checksum))
        if declared != checksum:
            raise ValueError("catalog checksum does not match its declared SHA-256")
        return cls(nodes, edges, source=str(raw.get("source", "unknown")), version=str(raw.get("version", "unknown")), checksum=checksum)

    def ids_for(self, *, types: Iterable[str] = (), regions: Iterable[str] = ()) -> list[str]:
        ids = set()
        for value in types:
            ids.update(self._by_type.get(value, ()))
        for value in regions:
            ids.update(self._by_region.get(value, ()))
        return sorted(ids)

    def ids_matching_prefixes(self, prefixes: Iterable[str]) -> list[str]:
        """Case-insensitive cell-type prefix match.

        The MaleCNS catalog labels cells anatomically (``DNp01``, ``MBON01``,
        ``AMMC-A1`` …) while callers seed functionally (``sensory``, ``visual``).
        This bridges the two without hard-coding full type names.
        """
        wanted = [str(p).lower() for p in prefixes if str(p)]
        if not wanted:
            return []
        ids: set[str] = set()
        for cell_type, members in self._by_type.items():
            lowered = cell_type.lower()
            if any(lowered.startswith(prefix) for prefix in wanted):
                ids.update(members)
        return sorted(ids)

    def subgraph(self, seeds: Iterable[str], *, budget: int = 20000, max_hops: int = 4) -> dict:
        """Return a deterministic forward path query, never exceeding *budget*."""
        valid_seeds = sorted(set(seeds).intersection(self.nodes))
        if budget < 1:
            return {
                "nodes": [], "edges": [], "truncated": bool(valid_seeds),
                "source": self.source, "version": self.version, "checksum": self.checksum,
            }
        selected: set[str] = set()
        selected_order: list[str] = []
        scheduled: set[str] = set()
        queue = deque()
        truncated = False
        for seed in valid_seeds:
            if len(scheduled) >= budget:
                truncated = True
                continue
            scheduled.add(seed)
            queue.append((seed, 0))
        while queue:
            node_id, depth = queue.popleft()
            selected.add(node_id)
            selected_order.append(node_id)
            if depth < max_hops:
                for edge in self._out.get(node_id, ()):
                    if edge.target in scheduled:
                        continue
                    if len(scheduled) >= budget:
                        truncated = True
                        continue
                    scheduled.add(edge.target)
                    queue.append((edge.target, depth + 1))
        edges = [edge for node_id in selected_order for edge in self._out.get(node_id, ()) if edge.target in selected]
        return {
            "nodes": [self.nodes[node_id] for node_id in sorted(selected)],
            "edges": edges,
            "truncated": truncated,
            "source": self.source,
            "version": self.version,
            "checksum": self.checksum,
        }

    def summary(self) -> dict:
        return {"nodes": len(self.nodes), "edges": len(self.edges), "source": self.source, "version": self.version, "checksum": self.checksum}
