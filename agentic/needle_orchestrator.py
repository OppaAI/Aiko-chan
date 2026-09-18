"""Deterministic, bounded orchestration for independent Needle 2 workers.

Workers are configured explicitly and must point at separate Needle servers (or
at server sessions that the Needle deployment documents as isolated).  This
module only aggregates constrained tool proposals; Aiko's normal ReAct loop
continues to validate, approve, and execute every proposed call.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from agentic.needle import NeedleClient, NeedleError, NeedleResponse

try:
    from system.log import get_logger as _get_logger
    log = _get_logger(__name__)
except Exception:
    import logging as _logging
    log = _logging.getLogger("aiko.needle_orchestrator")

try:
    from system.config import env_str as _env_str
except Exception:
    _env_str = None


def _flycx_mode() -> str:
    try:
        if _env_str is None:
            return "off"
        return _env_str("MEMORY_FLYCX_MODE", "off").strip().lower()
    except Exception:
        return "off"


def _flycx_cadence(task: str, user_id: str | None = None) -> str:
    """parallel|sequential crew cadence from compass sleep pressure.

    Shares the identity-scoped compass with cognition.attention via
    fly_registry. Drowsy crews run sequentially (same coverage, calmer
    cadence). Shadow only logs the would-be choice.
    """
    mode = _flycx_mode()
    if mode not in ("shadow", "live"):
        return "parallel"
    try:
        from cognition.fly_registry import get_flycx, get_flycx_lock
        from cognition.flymemory import text_features
        if user_id is None:
            try:
                from system.userspace import current_user_id
                user_id = current_user_id() or None
            except Exception:
                user_id = None
        cx = get_flycx(user_id)
        if cx is None:
            return "parallel"
        with get_flycx_lock(user_id):
            out = cx.step(text_features(task or ""), fatigue=0.0)
    except Exception as exp:
        log.debug("flycx cadence failed: %s", exp)
        return "parallel"
    choice = "sequential" if out["sleep_pressure"] > 0.5 else "parallel"
    log.debug("flycx cadence mode=%s sleep=%.2f -> %s", mode, out["sleep_pressure"], choice)
    return choice if mode == "live" else "parallel"


@dataclass(frozen=True)
class NeedleWorkerSpec:
    name: str
    client: NeedleClient
    tools_filter: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None


@dataclass(frozen=True)
class NeedleWorkerResult:
    name: str
    response: NeedleResponse | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.response is not None and self.error is None


class NeedleOrchestrator:
    """Run several Needle workers and merge their structured proposals."""

    def __init__(self, workers: list[NeedleWorkerSpec]) -> None:
        if not workers:
            raise ValueError("NeedleOrchestrator requires at least one worker")
        self.workers = list(workers)

    def _run_one(
        self,
        worker: NeedleWorkerSpec,
        task: str,
        tools: list[dict[str, Any]],
    ) -> NeedleWorkerResult:
        try:
            filtered = worker.tools_filter(tools) if worker.tools_filter else tools
            if not filtered:
                return NeedleWorkerResult(worker.name, None, error="no tools after filter")
            resp = worker.client.complete(task, filtered)
            return NeedleWorkerResult(worker.name, resp)
        except NeedleError as exc:
            return NeedleWorkerResult(worker.name, None, error=str(exc))
        except Exception as exp:
            return NeedleWorkerResult(worker.name, None, error=str(exp))

    def complete(self, task: str, tools: list[dict[str, Any]]) -> tuple[NeedleWorkerResult, ...]:
        """Run workers concurrently (or sequentially when drowsy), preserving configured order."""
        cadence = _flycx_cadence(task)
        max_workers = 1 if cadence == "sequential" else len(self.workers)
        try:
            from cognition.fly_behavior import maintenance_level
            uid = None
            try:
                from system.userspace import current_user_id
                uid = current_user_id() or None
            except Exception:
                uid = None
            level = maintenance_level(uid)
            if level == "maintenance":
                max_workers = 1
                log.debug("flysleep maintenance → sequential needle user=%s", uid or "default")
            elif level == "reduced":
                max_workers = min(max_workers, max(1, (len(self.workers) + 1) // 2))
                log.debug("flysleep reduced → max_workers=%d user=%s", max_workers, uid or "default")
        except Exception:
            pass
        results: dict[str, NeedleWorkerResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="needle-worker") as pool:
            futures = {pool.submit(self._run_one, worker, task, tools): worker for worker in self.workers}
            for fut in as_completed(futures):
                worker = futures[fut]
                try:
                    results[worker.name] = fut.result()
                except Exception as exp:
                    results[worker.name] = NeedleWorkerResult(worker.name, None, error=str(exp))
        ordered = tuple(results.get(w.name) or NeedleWorkerResult(w.name, None, error="missing") for w in self.workers)
        return ordered


def merge_needle_results(results: tuple[NeedleWorkerResult, ...]) -> list[dict[str, Any]]:
    """Merge tool proposals from workers; later workers do not override earlier ones by name."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for r in results:
        if not r.ok or r.response is None:
            continue
        for item in getattr(r.response, "tool_calls", None) or []:
            name = (item.get("name") if isinstance(item, dict) else None) or ""
            if not name or name in seen:
                continue
            seen.add(name)
            merged.append(item)
    return merged
