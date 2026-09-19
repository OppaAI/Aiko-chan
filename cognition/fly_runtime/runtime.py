"""Identity-scoped active graph execution and explainable traces."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .catalog import ConnectomeCatalog
from .dynamics import ActiveDynamics


@dataclass
class FlyRuntime:
    catalog: ConnectomeCatalog
    user_id: str | None = None
    trace: dict = field(default_factory=dict)

    def activate(self, seeds: list[str], drive: dict[str, float], *, budget: int = 20000, max_hops: int = 4, mode: str = "live", observations: list[dict] | None = None) -> dict:
        if mode not in {"shadow", "live"}:
            raise ValueError("mode must be 'shadow' or 'live'")
        active = self.catalog.subgraph(seeds, budget=budget, max_hops=max_hops)
        dynamics = ActiveDynamics(active["nodes"], active["edges"])
        # One bounded update per eligible hop lets a sensory drive reach an
        # output node without ever turning this into an unbounded full-graph
        # simulation.
        rates = {}
        # Cap settling passes independently from graph depth so a malformed
        # request cannot turn one interaction into O(nodes * hops) work.
        for _ in range(min(8, max(1, max_hops + 1))):
            rates = dynamics.step(drive)
        outputs = {node_id: value for node_id, value in rates.items() if self.catalog.nodes[node_id].type in {"DN", "VNC", "output"}}
        self.trace = {
            "at": time.time(), "user_id": self.user_id, "seeds": sorted(seeds),
            "budget": budget, "active_nodes": len(active["nodes"]), "active_edges": len(active["edges"]),
            "truncated": active["truncated"], "source": active["source"], "version": active["version"],
            "checksum": active["checksum"], "mode": mode,
            "observations": list(observations or []),
            "rates": {key: round(value, 5) for key, value in rates.items() if value > 0},
            "outputs": {key: round(value, 5) for key, value in outputs.items()},
            "nodes": [
                {"id": node.id, "type": node.type, "region": node.region, "rate": round(rates.get(node.id, 0.0), 5)}
                for node in active["nodes"]
            ],
            "edges": [
                {"source": edge.source, "target": edge.target, "weight": edge.weight, "sign": edge.sign}
                for edge in active["edges"]
            ],
        }
        if mode == "live":
            self._publish(rates, outputs)
        return dict(self.trace)

    def _publish(self, rates: dict[str, float], outputs: dict[str, float]) -> None:
        """Translate graph readouts to the existing, bounded NeuralState bus."""
        try:
            from cognition.neural_state import get_neural_state
            state = get_neural_state(self.user_id)
            sensory = max((v for node, v in rates.items() if self.catalog.nodes[node].type in {"sensory", "AL", "T4", "T5"}), default=0.0)
            vigor = max(outputs.values(), default=0.5)
            state.sensory_gain = max(0.5, min(1.5, 0.5 + sensory))
            state.motion_salience = max(0.0, min(1.0, sensory))
            state.publish_dn(arousal=vigor, rate_mult=0.8 + 0.4 * vigor, source="active_subgraph")
        except Exception:
            # The runtime remains usable for offline evaluation without Aiko's bus.
            return


_lock = threading.RLock()
_runtimes: dict[str, FlyRuntime] = {}


def _key(user_id: str | None) -> str:
    try:
        from cognition.fly_registry import _norm_id
        return _norm_id(user_id)
    except Exception:
        return (user_id or "").strip() or "default"


def get_fly_runtime(user_id: str | None, catalog: ConnectomeCatalog) -> FlyRuntime:
    key = _key(user_id)
    with _lock:
        runtime = _runtimes.get(key)
        if runtime is None or runtime.catalog is not catalog:
            runtime = FlyRuntime(catalog=catalog, user_id=user_id)
            _runtimes[key] = runtime
        return runtime


def peek_fly_runtime(user_id: str | None = None) -> FlyRuntime | None:
    with _lock:
        return _runtimes.get(_key(user_id))
