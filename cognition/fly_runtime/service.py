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
_bg_counts: dict[str, int] = {}


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


def _configured_catalog() -> "ConnectomeCatalog | SqliteCatalog | None":
    """Load the connectome, preferring the indexed SQLite build.

    ``AIKO_FLY_CATALOG_PATH`` points at the canonical JSON; when a
    same-basename ``.db`` sits beside it (built once via
    tools/build_catalog_sqlite.py), the SQLite backend is used instead —
    hot cells in RAM, cold graph on disk, no 1GB parse spike. The JSON
    path remains the fallback (dev machines, tests).
    """
    global _catalog, _catalog_path
    path = (os.getenv("AIKO_FLY_CATALOG_PATH", "") or "").strip()
    if not path:
        return None
    resolved = str(Path(path).expanduser().resolve())
    with _lock:
        if _catalog is None or _catalog_path != resolved:
            db_path = str(Path(resolved).with_suffix(".db"))
            if Path(db_path).exists():
                try:
                    from .sqlite_catalog import SqliteCatalog

                    _catalog = SqliteCatalog(db_path)
                    _catalog_path = resolved
                    log.info("fly runtime: using indexed catalog %s", db_path)
                except Exception as exc:
                    log.warning("fly runtime: indexed catalog unusable, JSON fallback: %s", exc)
                    _catalog = ConnectomeCatalog.from_path(resolved)
                    _catalog_path = resolved
            else:
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
    """Evaluate optional telemetry off the conversational hot path.

    Throttled by AIKO_FLY_BG_EVERY_N (default 1 = every call): background
    threads share the GIL, so each CPU-bound evaluation measurably slows
    token streaming on small boxes. Raising the stride (e.g. 3) keeps the
    trace fresh at a fraction of the contention.
    """
    mode = (os.getenv("AIKO_FLY_RUNTIME_MODE", "off") or "off").strip().lower()
    if mode not in {"shadow", "live"} or not observation.consented:
        return
    try:
        stride = max(1, int(os.getenv("AIKO_FLY_BG_EVERY_N", "1")))
    except (TypeError, ValueError):
        stride = 1
    key = (user_id or "").strip() or "default"
    with _background_lock:
        if key in _background_inflight:
            return
        count = _bg_counts.get(key, 0) + 1
        _bg_counts[key] = count
        if (count - 1) % stride != 0:
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
