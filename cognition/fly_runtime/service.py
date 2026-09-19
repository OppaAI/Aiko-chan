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
    seeds = catalog.ids_for(types=seed_types)
    if not seeds:
        return None
    budget = max(1, min(20000, int(os.getenv("AIKO_FLY_ACTIVE_BUDGET", "20000"))))
    return get_fly_runtime(user_id, catalog).activate(
        seeds, observation.drive(seeds[0]), budget=budget, mode=mode, observations=[observation.as_trace()],
    )
