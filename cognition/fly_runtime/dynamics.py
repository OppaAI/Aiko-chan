"""Small deterministic rate dynamics for one active graph, never the full catalog."""
from __future__ import annotations

from collections.abc import Iterable
from cognition.neural_state import _finite

from .catalog import Edge, Node


class ActiveDynamics:
    """Leaky bounded rate update; signs remain neutral when the catalog lacks them."""

    def __init__(self, nodes: Iterable[Node], edges: Iterable[Edge], *, leak: float = 0.35) -> None:
        self.rates = {node.id: 0.0 for node in nodes}
        self.edges = tuple(edges)
        self.leak = max(0.0, min(1.0, _finite(leak, 0.35)))

    def step(self, drive: dict[str, float]) -> dict[str, float]:
        incoming = {node_id: max(0.0, min(1.0, _finite(value, 0.0))) for node_id, value in drive.items() if node_id in self.rates}
        for edge in self.edges:
            sign = -1.0 if edge.sign == "inhibitory" else 1.0
            incoming[edge.target] = incoming.get(edge.target, 0.0) + self.rates[edge.source] * _finite(edge.weight, 0.0) * sign
        for node_id, old in self.rates.items():
            target = max(0.0, min(1.0, _finite(incoming.get(node_id, 0.0), 0.0)))
            self.rates[node_id] = max(0.0, min(1.0, old * (1.0 - self.leak) + target * self.leak))
        return dict(self.rates)
