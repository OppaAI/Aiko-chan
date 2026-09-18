"""Fly Circuit Studio API — read-only NeuralState + mode snapshot."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/state")
def fly_state(request: Request) -> dict:
    uid = getattr(request.state, "user_id", None)
    try:
        from cognition.neural_state import get_neural_state
        snap = get_neural_state(uid).snapshot()
    except Exception as exc:
        snap = {"error": str(exc)}
    try:
        from system.config import env_str
        modes = {
            "MEMORY_FLYMB_MODE": env_str("MEMORY_FLYMB_MODE", "off"),
            "MEMORY_FLYCX_MODE": env_str("MEMORY_FLYCX_MODE", "off"),
            "MEMORY_FLYLH_MODE": env_str("MEMORY_FLYLH_MODE", "off"),
            "MEMORY_FLYGF_MODE": env_str("MEMORY_FLYGF_MODE", "off"),
            "MEMORY_FLYSLEEP_MODE": env_str("MEMORY_FLYSLEEP_MODE", "off"),
            "MEMORY_FLYDN_MODE": env_str("MEMORY_FLYDN_MODE", "off"),
        }
    except Exception:
        modes = {}
    return {"user_id": uid, "neural_state": snap, "modes": modes}
