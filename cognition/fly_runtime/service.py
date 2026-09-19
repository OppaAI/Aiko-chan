"""Lazy application wiring for an operator-provided MaleCNS catalog."""
from __future__ import annotations

import os
import threading
from pathlib import Path

from .adapters import SensoryObservation
from .catalog import ConnectomeCatalog
from .runtime import get_fly_runtime

_lock = threading.RLock()
_catalog: ConnectomeCatalog | None = None
_catalog_path: str | None = None
_background_lock = threading.Lock()
_background_inflight: set[str] = set()


def _configured_catalog() -> ConnectomeCatalog | None:
    global _catalog, _catalog_path
    path = (os.getenv("AIKO_FLY_CATALOG_PATH", "") or "").strip()
    if not path:
        return None
    resolved = str(Path(path).expanduser().resolve())
    with _lock:
        if _catalog is None or _catalog_path != resolved:
            _catalog = ConnectomeCatalog.from_path(resolved)
            _catalog_path = resolved
        return _catalog


def observe(user_id: str | None, observation: SensoryObservation, *, seed_types: tuple[str, ...]) -> dict | None:
    """Evaluate one consented observation, or return ``None`` when disabled."""
    mode = (os.getenv("AIKO_FLY_RUNTIME_MODE", "off") or "off").strip().lower()
    if mode not in {"shadow", "live"} or not observation.consented:
        return None
    catalog = _configured_catalog()
    if catalog is None:
        return None
    seed_limit = max(1, min(256, int(os.getenv("AIKO_FLY_SEED_LIMIT", "64"))))
    seeds = catalog.ids_for(types=seed_types)[:seed_limit]
    if not seeds:
        return None
    # Keep normal chat telemetry bounded; explicit evaluation can raise this.
    budget = max(1, min(20000, int(os.getenv("AIKO_FLY_ACTIVE_BUDGET", "4000"))))
    max_hops = max(1, min(8, int(os.getenv("AIKO_FLY_ACTIVE_HOPS", "4"))))
    return get_fly_runtime(user_id, catalog).activate(
        seeds, {seed: max(0.0, min(1.0, observation.salience)) for seed in seeds},
        budget=budget, max_hops=max_hops,
        mode=mode, observations=[observation.as_trace()],
    )


def observe_background(user_id: str | None, observation: SensoryObservation, *, seed_types: tuple[str, ...]) -> None:
    """Evaluate optional telemetry off the conversational hot path."""
    mode = (os.getenv("AIKO_FLY_RUNTIME_MODE", "off") or "off").strip().lower()
    if mode not in {"shadow", "live"} or not observation.consented:
        return
    key = (user_id or "").strip() or "default"
    with _background_lock:
        if key in _background_inflight:
            return
        _background_inflight.add(key)

    def run() -> None:
        try:
            observe(user_id, observation, seed_types=seed_types)
        except Exception:
            pass
        finally:
            with _background_lock:
                _background_inflight.discard(key)

    threading.Thread(target=run, name="aiko-fly-runtime", daemon=True).start()
