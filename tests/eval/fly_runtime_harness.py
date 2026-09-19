#!/usr/bin/env python3
"""Jetson-oriented active-subgraph smoke benchmark; no model server required."""
from __future__ import annotations

import time

from cognition.fly_runtime import ConnectomeCatalog, Edge, Node, get_fly_runtime


def main() -> int:
    count = 2000
    nodes = [Node(f"n{i}", "DN" if i == count - 1 else ("sensory" if i == 0 else "relay")) for i in range(count)]
    catalog = ConnectomeCatalog(nodes, [Edge(f"n{i}", f"n{i + 1}") for i in range(count - 1)], source="synthetic-benchmark")
    started = time.perf_counter()
    trace = get_fly_runtime("eval", catalog).activate(["n0"], {"n0": 1.0}, budget=count, max_hops=count)
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"active_nodes={trace['active_nodes']} active_edges={trace['active_edges']} elapsed_ms={elapsed_ms:.2f}")
    return 0 if trace["active_nodes"] == count else 2


if __name__ == "__main__":
    raise SystemExit(main())
