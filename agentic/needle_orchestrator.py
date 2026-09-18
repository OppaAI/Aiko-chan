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
except Exception:  # logging must never break orchestration
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
        from cognition.fly_registry import get_flycx
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
        out = cx.step(text_features(task or ""), fatigue=0.0)
    except Exception as exc:
        log.debug("flycx cadence failed: %s", exc)
        return "parallel"
    choice = "sequential" if out["sleep_pressure"] > 0.5 else "parallel"
    log.debug("flycx cadence mode=%s sleep=%.2f -> %s", mode, out["sleep_pressure"], choice)
    return choice if mode == "live" else "parallel"


@dataclass(frozen=True)
class NeedleWorkerSpec:
    """One explicitly configured Needle worker with a least-privilege tool set."""

    id: str
    role: str
    base_url: str
    allowed_tools: tuple[str, ...]
    confidence_threshold: float = 0.85
    timeout: float = 15.0


@dataclass(frozen=True)
class NeedleWorkerResult:
    worker_id: str
    role: str
    response: NeedleResponse | None = None
    error: str | None = None


class NeedleOrchestrator:
    """Fan-out independent Needle workers and collect constrained proposals."""

    def __init__(
        self,
        workers: tuple[NeedleWorkerSpec, ...] | list[NeedleWorkerSpec],
        client_factory: Callable[..., NeedleClient] | None = None,
    ) -> None:
        self.workers = tuple(workers)
        self._client_factory = client_factory or NeedleClient

    @staticmethod
    def _tools_for_worker(worker: NeedleWorkerSpec, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed = set(worker.allowed_tools)
        return [t for t in tools if (t.get("function", {}).get("name") or "") in allowed]

    @staticmethod
    def _worker_prompt(worker: NeedleWorkerSpec, task: str) -> str:
        return f"Role: {worker.role}.\nTask: {task}"

    def _run_one(self, worker: NeedleWorkerSpec, task: str, tools: list[dict[str, Any]]) -> NeedleWorkerResult:
        worker_tools = self._tools_for_worker(worker, tools)
        if not worker_tools:
            return NeedleWorkerResult(worker.id, worker.role, error="no allowed tools are available for this turn")
        try:
            client = self._client_factory(
                worker.base_url,
                timeout=worker.timeout,
                confidence_threshold=worker.confidence_threshold,
            )
            return NeedleWorkerResult(
                worker.id,
                worker.role,
                response=client.complete(self._worker_prompt(worker, task), worker_tools),
            )
        except NeedleError as exc:
            return NeedleWorkerResult(worker.id, worker.role, error=str(exc))

    def complete(self, task: str, tools: list[dict[str, Any]]) -> tuple[NeedleWorkerResult, ...]:
        """Run workers concurrently (or sequentially when drowsy), preserving configured order."""
        cadence = _flycx_cadence(task)
        max_workers = 1 if cadence == "sequential" else len(self.workers)
        results: dict[str, NeedleWorkerResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="needle-worker") as pool:
            futures = {pool.submit(self._run_one, worker, task, tools): worker for worker in self.workers}
            for future in as_completed(futures):
                worker = futures[future]
                try:
                    results[worker.id] = future.result()
                except Exception as exc:  # defensive: one worker cannot abort the team
                    results[worker.id] = NeedleWorkerResult(worker.id, worker.role, error=str(exc))
        ordered = tuple(results[worker.id] for worker in self.workers)
        if not any(result.response is not None for result in ordered):
            details = "; ".join(f"{result.worker_id}: {result.error}" for result in ordered)
            raise NeedleError(f"all Needle workers failed or were ineligible: {details}")
        return ordered
