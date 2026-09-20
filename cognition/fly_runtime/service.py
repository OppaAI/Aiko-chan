"""Lazy application wiring for an operator-provided MaleCNS catalog."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from .adapters import SensoryObservation
from .catalog import ConnectomeCatalog
from .runtime import get_fly_runtime

log = logging.getLogger("aiko.fly_runtime")

_lock = threading.RLock()
_catalog: ConnectomeCatalog | None = None
_catalog_path: str | None = None
_background_lock = threading.Lock()
_background_inflight: set[str] = set()


# Functional seed label -> anatomical cell-type prefixes in the MaleCNS
# catalog (verified against male-cns-v1.0-w5.json cell types such as VS,
# MBON01, AMMC-A1, DNp01, ER5). Callers seed functionally ("visual");
# the catalog labels anatomically — this table bridges the two.
_SEED_ALIASES: dict[str, tuple[str, ...]] = {
    "sensory": ("ORN", "OSN", "DA1", "VA7", "DP1", "DM", "VS", "HS",
                "Mi", "Tm", "L1", "L2", "L3", "L4", "L5", "R7", "R8",
                "LoV", "MeV", "AVLP", "PVLP", "PLP", "OCG", "GNG",
                "VES", "SMP", "AMMC", "DCH", "VCH", "CT1"),
    "visual": ("VS", "HS", "HSE", "HSN", "HSS", "H2", "Dm", "Mi", "Tm",
               "L1", "L2", "L3", "L4", "L5", "R7", "R8", "LoV", "MeV",
               "LPi", "DCH", "VCH", "CT1", "Am", "Li"),
    "auditory": ("AMMC", "JO", "aSP", "AVLP"),
    "input": ("ORN", "OSN", "DA1", "VA7", "DP1", "DM", "VS", "HS",
              "Mi", "Tm", "L1", "L2", "R7", "R8", "LoV", "MeV",
              "AVLP", "PVLP", "GNG", "AMMC"),
    "olfactory": ("ORN", "OSN", "DA1", "VA7", "DP1", "lPN", "adPN",
                  "OA-AL", "OA-V"),
    "memory": ("MBON", "KC", "PAM", "PPL", "DAN", "APL"),
    "decide": ("ER", "ExR", "EPG", "PFN", "PEN", "CX", "CB", "PB"),
    "action": ("DN", "MN"),
    "arousal": ("OA", "5-HT", "DA", "DH44", "AstA", "sNPF", "VUM", "VPM"),
}


def _resolve_seeds(catalog: ConnectomeCatalog, seed_types: tuple[str, ...]) -> list[str]:
    """Resolve functional seed labels to catalog cell ids.

    Tries, in order: exact cell-type match (legacy), then anatomical-prefix
    aliases (case-insensitive). Returns [] when nothing matches so callers
    fail silent-but-logged instead of evaluating noise.
    """
    seed_limit = max(1, min(256, int(os.getenv("AIKO_FLY_SEED_LIMIT", "64"))))
    exact = catalog.ids_for(types=tuple(seed_types))
    if exact:
        return exact[:seed_limit]
    lowered = {str(label).lower() for label in seed_types if str(label)}
    prefixes: list[str] = []
    for label in lowered:
        prefixes.extend(_SEED_ALIASES.get(label, ()))
    if not prefixes:
        log.warning("fly runtime: no seed alias for labels=%s", sorted(lowered))
        return []
    matched = catalog.ids_matching_prefixes(prefixes)
    if not matched:
        log.warning("fly runtime: aliases matched 0 cells for labels=%s", sorted(lowered))
        return []
    log.debug("fly runtime: labels=%s -> %d seed cells", sorted(lowered), len(matched))
    return matched[:seed_limit]


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
    seeds = _resolve_seeds(catalog, seed_types)
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
            result = observe(user_id, observation, seed_types=seed_types)
            if result is not None:
                log.info(
                    "fly runtime: evaluated trace user=%s nodes=%s edges=%s",
                    (user_id or "").strip() or "default",
                    result.get("active_nodes"), result.get("edges") and len(result["edges"]),
                )
            else:
                log.debug("fly runtime: background observation produced no trace")
        except Exception as exc:
            log.warning("fly runtime: background evaluation failed: %s", exc)
        finally:
            with _background_lock:
                _background_inflight.discard(key)

    threading.Thread(target=run, name="aiko-fly-runtime", daemon=True).start()
